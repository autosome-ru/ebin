import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from functools import partial
from dataclasses import dataclass
from scipy.stats import poisson as scp_poisson
from copy import deepcopy
import warnings
import numpy as np
import gc

from .compound_start import fit_nb
from .gauss_rules import compute_nodes_and_logweights, Rules


@dataclass(frozen=True)
class LatentCountsResult:
    params: dict
    responsibilities: np.ndarray
    counts: np.ndarray
    prospenity: np.ndarray
    prospenity_var: np.ndarray
    loglik: float
    num_nodes: int
    

MAX_POI = 100. 

_prepared_data = tuple[list[list[tuple[jnp.ndarray, jnp.ndarray]]], jnp.ndarray]


@jax.custom_vjp
def log_einsum_exp(log_A, log_B):
    """
    Computes log(A @ B) safely and optimally. 
    Memory usage avoids instantiation of the 4D broadcast array.
    """
    max_A = jax.lax.stop_gradient(jnp.max(log_A, axis=-1, keepdims=True))
    max_B = jax.lax.stop_gradient(jnp.max(log_B, axis=0, keepdims=True))
    
    max_A_safe = jnp.where(jnp.isneginf(max_A), 0.0, max_A)
    max_B_safe = jnp.where(jnp.isneginf(max_B), 0.0, max_B)
    
    A_lin = jnp.exp(log_A - max_A_safe)
    B_lin = jnp.exp(log_B - max_B_safe)
    
    C_lin = jnp.einsum('nkt,tv->nkv', A_lin, B_lin)
    
    eps_val = 1e-300 if log_A.dtype == jnp.float64 else 1e-38
    C_safe = jnp.maximum(C_lin, eps_val)
    
    res = jnp.log(C_safe) + max_A_safe + max_B_safe
    is_neginf = jnp.isneginf(max_A) | jnp.isneginf(max_B) | (C_lin == 0)
    res = jnp.where(is_neginf, -jnp.inf, res)
    return res

def log_einsum_exp_fwd(log_A, log_B):
    return log_einsum_exp(log_A, log_B), (log_A, log_B)

def log_einsum_exp_bwd(res_tuple, g):
    log_A, log_B = res_tuple
    
    max_A = jax.lax.stop_gradient(jnp.max(log_A, axis=-1, keepdims=True))
    max_B = jax.lax.stop_gradient(jnp.max(log_B, axis=0, keepdims=True))
    
    max_A_safe = jnp.where(jnp.isneginf(max_A), 0.0, max_A)
    max_B_safe = jnp.where(jnp.isneginf(max_B), 0.0, max_B)
    
    A_lin = jnp.exp(log_A - max_A_safe)
    B_lin = jnp.exp(log_B - max_B_safe)
    
    C_lin = jnp.einsum('nkt,tv->nkv', A_lin, B_lin)
    
    # Clip limits to safely bound the highest possible gradient ratio
    clip_val = 1e-200 if log_A.dtype == jnp.float64 else 1e-20
    C_safe_bwd = jnp.maximum(C_lin, clip_val)
    
    g_over_C = g / C_safe_bwd
    
    d_log_A = A_lin * jnp.einsum('nkv,tv->nkt', g_over_C, B_lin)
    d_log_B = B_lin * jnp.einsum('nkv,nkt->tv', g_over_C, A_lin)
    
    return d_log_A, d_log_B

log_einsum_exp.defvjp(log_einsum_exp_fwd, log_einsum_exp_bwd)



def safe_nb_logpmf(x, n, p):
    """
    A safe version of the Negative Binomial logpmf that handles n=0 correctly.
    """
    # The problematic case for the gradient is when n=0.
    # The NB distribution with n=0 is a point mass at 0.
    # Its logpmf is 0 if x=0, and -inf if x>0.
    # The gradient wrt n of this branch is 0, which is correct.
    is_n_zero = (n < 1e-14)
    log_pmf_at_zero_n = jnp.where(x == 0, 0.0, -jnp.inf)
    
    # Calculate the standard logpmf for the non-zero n case.
    # We use a trick here: where n is zero, we pass n=1 to the function
    # to avoid a NaN, but this value will be masked out by the outer `jnp.where`.
    safe_n = jnp.where(is_n_zero, 1.0, n)
    log_pmf_at_nonzero_n = jax.scipy.stats.nbinom.logpmf(x, n=safe_n, p=p)

    return jnp.where(is_n_zero, log_pmf_at_zero_n, log_pmf_at_nonzero_n)

def safe_nb_logpmf_precomp(x, logp, log1p, rtau, gammaln_rtau, is_rtau_zero):

    log_pmf_at_zero_n = jnp.where(x == 0, 0.0, -jnp.inf)
    logpmf = gammaln(x + rtau) - gammaln_rtau - gammaln(x + 1) + logp
    logpmf = logpmf +  log1p * x

    return jnp.where(is_rtau_zero, log_pmf_at_zero_n, logpmf)


@partial(jax.jit, static_argnames=('batch_size',  'num_nodes', 'eps',
                                   'high_precision', 'fixed_beta'))
def marginal_loglik(processed_data: list[list[tuple[jnp.ndarray, jnp.ndarray]]], r: jnp.ndarray, p: jnp.ndarray,
                    theta: jnp.ndarray,  moments=None, lambda_moment: int = 0,
                    batch_size=48, num_nodes=25, eps=1e-9, 
                    high_precision: bool = True, fixed_beta: float = None):
    if high_precision:
        prec_int = jnp.int64
        prec_float = jnp.float64
    else:
        prec_int = jnp.int32
        prec_float = jnp.float32
    processed_data = processed_data[0]
    B, M, K = r.shape
    N = len(processed_data[0][0][-1])
    
    alpha = theta[0]
    if fixed_beta is None:
        beta = theta[1]
    else:
        beta = fixed_beta
        
    phi, logweights = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha - 1 + lambda_moment,
                                                   rule=Rules.GenLaguerre)
    lambda_div = M + 1 / beta
    phi_hat = phi / lambda_div
    log_phi_hat = jnp.log(phi) - jnp.log(M + 1 / beta)

    right = scp_poisson.isf(eps, (num_nodes * 4 + 2 * MAX_POI - 2) / M,)
    taus = jnp.arange(0, right + 1)
    loggammataus = gammaln(taus + 1)
    logtaus = jnp.log(taus)
    log_phi_poi = -loggammataus[..., None] + taus[..., None] * log_phi_hat
    
    rtau = r[..., None] * taus 
    is_rtau_zero = rtau == 0
    rtau = jnp.where(is_rtau_zero, 1.0, rtau)
    gammaln_rtau = gammaln(rtau)
    logp = jnp.log(p)[..., None] * rtau
    log1p = jnp.log1p(-p)[..., None]
    
    def calc_marginal_nb_precomp_old(xs, b, m, moment=None):
        logp_ = logp[b,m]
        log1p_ = log1p[b, m]
        rtau_ = rtau[b, m]
        gammaln_rtau_ = gammaln_rtau[b, m]
        is_rtau_zero_ = is_rtau_zero[b, m]
        if moment is not None:
            moment = moment[m]
        
        xs = xs.reshape(-1, 1, 1)
        
        logpmf = safe_nb_logpmf_precomp(xs, logp=logp_, log1p=log1p_, rtau=rtau_,
                                        gammaln_rtau=gammaln_rtau_, is_rtau_zero=is_rtau_zero_)[..., None] # N_uniq x K x batch_size x 1
        if moment is not None:
            safe_logtaus = jnp.where(taus == 0, 0.0, logtaus)
            t = safe_logtaus * moment
            t = jnp.where((taus == 0) & (moment > 0), -jnp.inf, t)
            logpmf = logpmf + t[..., None]
        # logsumexp
        res = logsumexp((logpmf + log_phi_poi), axis=-2)
        return res
    
    def calc_marginal_nb_precomp(xs, b, m, moment=None):
        logp_ = logp[b,m]
        log1p_ = log1p[b, m]
        rtau_ = rtau[b, m]
        gammaln_rtau_ = gammaln_rtau[b, m]
        is_rtau_zero_ = is_rtau_zero[b, m]
        if moment is not None:
            moment = moment[m]
        
        xs = xs.reshape(-1, 1, 1)

        logpmf = safe_nb_logpmf_precomp(xs, logp=logp_, log1p=log1p_, rtau=rtau_,
                                        gammaln_rtau=gammaln_rtau_, is_rtau_zero=is_rtau_zero_)
                                        
        if moment is not None:
            safe_logtaus = jnp.where(taus == 0, 0.0, logtaus)
            t = safe_logtaus * moment
            t = jnp.where((taus == 0) & (moment > 0), -jnp.inf, t)
            logpmf = logpmf + t
            
        return log_einsum_exp(logpmf, log_phi_poi)
        

    loglik = jnp.zeros((B, N, K), dtype=prec_float)
    const = logsumexp(logweights + phi_hat * M)

    const = const + jnp.log(M + 1 / beta) * lambda_moment
    if not high_precision:
        const = const.astype(jnp.float32)
        logweights = logweights.astype(jnp.float32)
        log_phi_poi = log_phi_poi.astype(jnp.float32)
        logtaus = logtaus.astype(jnp.float32)
        taus = taus.astype(jnp.float32)
        r = r.astype(jnp.float32)
        p = p.astype(jnp.float32)
        
        rtau = rtau.astype(jnp.float32)
        gammaln_rtau = gammaln_rtau.astype(jnp.float32)
        logp = logp.astype(jnp.float32)
        log1p = log1p.astype(jnp.float32)
        
        if moments is not None:
            moments = moments.astype(jnp.float32)

    for b in range(B):
        marg_nb = jnp.zeros((N, K, num_nodes), dtype=prec_float)
        for m in range(M):
            x, inv_ind = processed_data[b][m]
            x = x.astype(prec_int)
            marg_nb_s = calc_marginal_nb_precomp(x, b, m, moment=moments)
            marg_nb = marg_nb + marg_nb_s[inv_ind]
        marg_nb = marg_nb + logweights[None, None]
        loglik = loglik.at[b].set(logsumexp(marg_nb, axis=-1)) 
    return loglik - const


def prepare_data(data: jnp.ndarray) -> _prepared_data:
    res = list()
    zeros = np.all(data == 0, axis=1)
    for bin_matrix in data:
        lt = list()
        for sample_matrix in bin_matrix:
            unique_vals, inverse = np.unique(sample_matrix, return_inverse=True)
            lt.append((jnp.asarray(unique_vals).astype(jnp.int32),
                        jnp.asarray(inverse).astype(int)))
        res.append(lt)
    return res, jnp.array(zeros)



def update_weights(T: jnp.ndarray, C: jnp.ndarray, eps=1e-12) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs the M-step update for mixture weights (w) and zero-inflation probs (pi).

    This function maximizes the Q-function with respect to w and pi.

    Args:
        T (jnp.ndarray): The responsibility matrix for component assignment.
                         Shape: (N, K), where N is objects, K is components.
                         T_nk = P(Z_n=k | X_n, params_old)
        C (jnp.ndarray): The responsibility matrix for zero-inflation.
                         Shape: (B, N), where B is individuals.
                         C_nb = P(Phi_nb=1 | X_n, params_old)

    Returns:
        A tuple containing:
        - new_weights (jnp.ndarray): Updated mixture weights. Shape: (K,).
        - new_pi (jnp.ndarray): Updated zero-inflation probabilities. Shape: (B,).
    """

    N_from_T = T.shape[0]
    N_from_C = C.shape[1]
    assert N_from_T == N_from_C, f"Inconsistent number of objects: {N_from_T} from T vs {N_from_C} from C"
    N = N_from_T


    weights_numerator = jnp.sum(T, axis=0)
    new_weights = weights_numerator / N


    pi_numerator = jnp.sum(C, axis=1)
    new_pi = pi_numerator / N


    new_weights = jnp.clip(new_weights, eps, 1.0 - eps)
    new_weights /= jnp.sum(new_weights)

    new_pi = jnp.clip(new_pi, eps, 1.0 - eps)

    return new_weights, new_pi

@partial(jax.jit, static_argnames=('num_nodes', 'fixed_beta'))
def compute_responsibilities(processed_data: _prepared_data, 
                             r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray,
                             pi: jnp.ndarray, w: jnp.ndarray, num_nodes: int,
                             eps=1e-12, fixed_beta: float = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the responsibilities for the E-step of the hierarchical zero-inflated mixture model.


    Returns:
        A tuple containing:
        - T (jnp.ndarray): Posterior probabilities P(Z_n=k | X_n). Shape (N, K).
        - C (jnp.ndarray): Posterior probabilities P(Phi_nb=1 | X_n). Shape (B, N).
        - R (jnp.ndarray): Joint posterior P(Z_n=k, Phi_nb=0 | X_n). Shape (B, N, K).
    """
    # B: bins, M: replicates, N: sequences, K: components
    logP = marginal_loglik(processed_data, 
                           num_nodes=num_nodes, r=r, p=p, theta=theta,
                           high_precision=True, fixed_beta=fixed_beta)
    
    M = len(processed_data[0])
    B, N, K = logP.shape
    
    # I_nb: Indicator matrix where I[b, n] is 1 if all replicates for individual b
    # in object n are zero. Shape: (B, N)
    I_nb = processed_data[1]

    log_pi = jnp.log(pi)
    log_one_minus_pi = jnp.log1p(-pi)
    w = jnp.append(w, jnp.clip(1.0 - w.sum(), eps, 1.0 - eps))
    log_weights = jnp.log(w)

    # 2. Calculate log P(X_b | Z_n=k)
    # Sum the log probabilities over the M replicates.
    # log_L shape: (B, N, K)
    # log_L_nbk = jnp.sum(logP, axis=1)

    # 3. Calculate the core term for the marginal likelihood.
    # This term is log( pi_b * I_nb + (1-pi_b) * L_nbk )
    
    # Reshape for broadcasting: pi -> (B, 1, 1), I_nb -> (B, N, 1)
    log_pi_b = log_pi[:, None, None]
    log_one_minus_pi_b = log_one_minus_pi[:, None, None]
    
    # log(I_nb) will be -inf where I_nb is 0, correctly zeroing out the term.
    log_I_nb = jnp.log(I_nb[:, :, None])

    # Calculate the two terms inside the sum
    log_term1 = log_pi_b + log_I_nb
    log_term2 = log_one_minus_pi_b + logP
    
    # log_term_per_indiv_and_comp shape: (B, N, K)
    log_term_per_indiv_and_comp = jnp.logaddexp(log_term1, log_term2)

    # 4. Calculate log P(X_n | Z_n=k)
    # This is the product over individuals 'b', so we sum in log-space.
    # log_P_Xn_given_Znk shape: (N, K)
    log_P_Xn_given_Znk = jnp.sum(log_term_per_indiv_and_comp, axis=0)

    # 5. Calculate log P(X_n), the marginal log-likelihood for each object n.
    # This involves weighting by w_k and summing over components 'k'.
    # log(P(X_n)) = log( sum_k w_k * P(X_n | Z_n=k) )
    
    # log_joint_likelihoods shape: (N, K)
    log_joint_likelihoods = log_weights + log_P_Xn_given_Znk
    
    # log_P_Xn shape: (N,)
    log_P_Xn = logsumexp(log_joint_likelihoods, axis=1)

    # 6. Compute T_nk = P(Z_n=k | X_n)
    # T_nk = (w_k * P(X_n | Z_n=k)) / P(X_n)
    # In log space: log(T_nk) = log(w_k) + log(P(X_n|Z_n=k)) - log(P(X_n))
    log_T = log_joint_likelihoods - log_P_Xn[:, None] # Broadcast log_P_Xn
    T = jnp.exp(log_T)

    # 7. Compute R_nbk = P(Z_n=k, Phi_nb=0 | X_n)
    # R_nbk = T_nk * P(Phi_nb=0 | X_n, Z_n=k)
    # P(Phi_nb=0 | X_n, Z_n=k) = ( (1-pi_b) * L_nbk ) / ( pi_b*I_nb + (1-pi_b)*L_nbk )
    # So, R_nbk = T_nk * (1-pi_b) * L_nbk / term_bk
    # In log space: log(R) = log(T) + log(1-pi) + log(L) - log(term)
    
    # Broadcast T: (1, N, K), pi: (B, 1, 1) to match (B, N, K) shapesweights
    log_R = log_T[None, :, :] + log_one_minus_pi_b + logP - log_term_per_indiv_and_comp
    R = jnp.exp(log_R)

    # 8. Compute C_nb = P(Phi_nb=1 | X_n)
    # C_nb = 1 - sum_k P(Z_n=k, Phi_nb=0 | X_n)
    # C_nb = 1 - sum_k R_nbk
    C = 1.0 - jnp.sum(R, axis=2)

    return T, C, R, log_P_Xn.sum() / (B * M * N)


def get_starting_values(data: jnp.ndarray, K: int, theta, r, p, w, pi, fixed_beta: float = None):
    B, M, N = data.shape
    if r is None:
        r = jnp.arange(1, K + 1, dtype=float).reshape(1, 1, -1)
        r = jnp.repeat(r, B, axis=0)
        r = jnp.repeat(r, M, axis=1)
    
    if theta is None:
        if fixed_beta is None:
            theta = jnp.array([7.5, 1.0])
            mean_tau = theta[0] / theta[1]
        else:
            theta = jnp.array([7.5, ])
            mean_tau = theta[0] / fixed_beta
        r = r / mean_tau
    else:
        if fixed_beta is None and len(theta) == 1:
            theta = jnp.array([theta[0], 1.0])

    if p is None:
        p = jnp.linspace(1e-2, 0.1, num=K)[::-1].reshape(1, 1, -1)
        p = jnp.repeat(p, B, axis=0)
        p = jnp.repeat(p, M, axis=1) 
    if w is None:
        w = jnp.arange(1, K + 1)
        w = w / w.sum()
        w = w.at[:-1].get()
    elif len(w) == K:
        w = jnp.asarray(w).at[:-1].get()
    if pi is None:
        pi = jnp.all(data == 0, axis=1).mean(axis=-1) / 2
    return theta, r, p, w, pi
    

@partial(jax.jit, static_argnames=('num_nodes', 'high_precision', 'fixed_beta'))
def Q_dist(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, R: jnp.ndarray,
           processed_data: _prepared_data, num_nodes: int,
           high_precision: bool = True, fixed_beta: float = None):
    logP = marginal_loglik(processed_data, 
                           r=r, p=p, theta=theta, num_nodes=num_nodes,
                           high_precision=high_precision, fixed_beta=fixed_beta)
    
    return -(logP * R).mean()
    
@partial(jax.jit, static_argnames=('num_nodes', 'high_precision', 'fixed_beta'))
def negloglik(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, w: jnp.ndarray, pi: jnp.ndarray,
           processed_data: _prepared_data, num_nodes: int,  R=None,
           high_precision: bool = True, fixed_beta: float = None):
    M = len(processed_data[0][0])
    eps = 1e-15 if high_precision else 1e-9
    # logP shape: (B, N, K)
    logP = marginal_loglik(processed_data, 
                           r=r, p=p, theta=theta, num_nodes=num_nodes,
                           high_precision=high_precision, fixed_beta=fixed_beta)
    B, N, K = logP.shape
    

    log_pi_b = jnp.log(jnp.clip(pi, eps, 1.0 - eps))[:, None, None]
    log_one_minus_pi_b = jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))[:, None, None]
    

    I_nb = processed_data[-1]
    log_I_nb = jnp.log(I_nb)[:, :, None]
    

    log_term1 = log_pi_b + log_I_nb
    log_term2 = log_one_minus_pi_b + logP
    log_term_per_indiv_and_comp = jnp.logaddexp(log_term1, log_term2)  # Shape: (B, N, K)
    

    log_P_Xn_given_Znk = jnp.sum(log_term_per_indiv_and_comp, axis=0)  # Shape: (N, K)
    
    if len(w):

        w_full = jnp.append(w, jnp.clip(1.0 - w.sum(), eps, 1.0 - eps))
        log_weights = jnp.log(w_full)
        

        log_joint_likelihoods = log_weights + log_P_Xn_given_Znk
        log_P_Xn = logsumexp(log_joint_likelihoods, axis=1) # Shape: (N,)
    else:

        log_P_Xn = log_P_Xn_given_Znk[:, 0]
        
    return -log_P_Xn.sum() / (B * M * N)


def param_slicer(x: jnp.ndarray, names: list[str], shapes: list[tuple[int]], mult: bool = False, **kwargs):
    def mult_parmas(r, p):
        K = r.shape[-1]
        r_mult = x[-2 *K: -K]
        p_mult = x[-K:]
        t = p /(1 - p) * p_mult
        p = t / (t + 1)
        r = r * r_mult
        return r, p

    params = dict()
    n = 0
    x = jnp.asarray(x)
    for name, shape in zip(names, shapes):
        m = n + np.prod(np.asarray(shape))
        params[name] = x.at[n:m].get().reshape(shape)
        n = m
    if mult:
        if 'r' in kwargs:
            r = kwargs['r']
            p = kwargs['p']
            r, p = mult_parmas(r, p)
            kwargs['r'] = r
            kwargs['p'] = p
        elif 'r' in params:
            r, p = params['r'], params['p']
            r, p = mult_parmas(r, p)
            params['r'] = r
            params['p'] = p
    return params, kwargs

@partial(jax.jit, static_argnames=('f', 'names', 'shapes', 'mult'))
def param_slicer_wrapper(x: jnp.ndarray, f, names: list[str], shapes: list[tuple[int]],
                         mult: int = 0, warmup_vec: jnp.array = None, warmup_x: jnp.array = None,
                         **kwargs):
    if warmup_vec is not None:
        x = jnp.where(warmup_vec, warmup_x, x)
    params, kwargs = param_slicer(x, names, shapes, mult, **kwargs)
    return f(**params, **kwargs)
    

def get_bounds_():
    base = {'r': [[1e-5, None]], 'p': [[1e-9, 0.99]],
            'w': [[0.0, 1.0]], 'pi': [[0.0, 1.0]]}
    return base

def get_bounds(names, shapes):
    bounds = list()
    base = get_bounds_()
    for name, shape in zip(names, shapes):
        t = np.prod(np.asarray(shape))
        if name == 'theta':
            bounds.extend([[1e-3, MAX_POI],])
            if t > 1:
                bounds.extend([[1e-3, None],])
        else:
            b = base[name]
            bounds.extend(b * t)
    return bounds
    
def optimize(fun, data, processed_data: _prepared_data, params: dict, num_iters: int,
             R=None, ftol: float = 1e-9, 
             return_funs: bool = False, fun_aux=None, maxiter: int = 2500,
             optimizer='TNC', use_mults=False, warmup: bool = False,
             ):
    params = deepcopy(params)
    names = tuple(sorted(params.keys()))
    shapes = tuple([params[name].shape for name in names])
    bounds = get_bounds(names, shapes)
    if use_mults:
        num_mults = params['r'].shape[-1]
        bounds.extend([(0.1, 10)] * (2 * num_mults) )
    else:
        num_mults = 0
    slicer = partial(param_slicer, names=names, shapes=shapes, mult=num_mults)
    
    if fun_aux is None:
        fun_raw = partial(param_slicer_wrapper, f=fun, names=names, shapes=shapes,
                          mult=num_mults, processed_data=processed_data)
        
        grad_fn = jax.grad(fun_raw, argnums=0)
        
        @jax.jit
        def fun_val_grad(x, R=None, warmup_vec=None, warmup_x=None):
            return fun_raw(x, R=R, warmup_vec=warmup_vec, warmup_x=warmup_x), grad_fn(x, R=R, warmup_vec=warmup_vec, warmup_x=warmup_x)
        
        @jax.jit
        def hessp_fd(x, v, R=None, warmup_vec=None, warmup_x=None):
            """
            Finite difference approximation of the Hessian-Vector Product.
            H*v ~ (g(x + h*v) - g(x - h*v)) / (2h)
            """
            v_cast = jnp.asarray(v, dtype=x.dtype)
            v_norm = jnp.linalg.norm(v_cast)
            
            # Define a small step size, safely scaling by the vector's norm
            eps = 1e-5
            h = jnp.where(v_norm > 0.0, eps / v_norm, eps)
            
            def g(x_inner):
                return grad_fn(x_inner, R=R, warmup_vec=warmup_vec, warmup_x=warmup_x)
            
            g_plus = g(x + h * v_cast)
            g_minus = g(x - h * v_cast)
            return (g_plus - g_minus) / (2.0 * h)
            
        fun_aux = (fun_val_grad, hessp_fd)
        
    if isinstance(fun_aux, tuple):
        fun_opt, hessp_opt = fun_aux
    else:
        fun_opt = fun_aux
        hessp_opt = None

    x0 = list()
    for name in names:
        x0.extend(jnp.asarray(params[name]).flatten())
    x0 = jnp.array(x0)
    warmup_vec = list()
    for name in names:
        if warmup and name == 'theta':
            warmup_vec.extend(jnp.ones_like(params[name], dtype=bool).flatten())
        else:
            warmup_vec.extend(jnp.zeros_like(params[name], dtype=bool).flatten())
    
    x0 = jnp.append(x0, jnp.ones(2 * num_mults, dtype=float))
    
    warmup_vec.extend(jnp.zeros(2 * num_mults, dtype=bool))
    warmup_vec = jnp.asarray(warmup_vec, dtype=bool)
    
    with warnings.catch_warnings(action="ignore"):
        if warmup is not None:
            fun_to_use = partial(fun_opt, R=R, warmup_vec=warmup_vec, warmup_x=x0)
            if hessp_opt is not None:
                hessp_to_use = partial(hessp_opt, R=R, warmup_vec=warmup_vec, warmup_x=x0)
        else:
            fun_to_use = partial(fun_opt, R=R)
            if hessp_opt is not None:
                hessp_to_use = partial(hessp_opt, R=R)

        options = {'maxiter': maxiter, 'ftol': ftol}
                 
        opt_method = optimizer
        if optimizer.lower() == 'newton':

            opt_method = 'trust-constr'
            options['xtol'] = ftol
            options['gtol'] = ftol

            if 'ftol' in options:
                del options['ftol']
        elif optimizer == 'TNC':
            options['maxfun'] = maxiter
            
        try:
            kwargs = {'jac': True, 'method': opt_method, 'options': options, 'bounds': bounds}
            if optimizer.lower() == 'newton' and hessp_opt is not None:
                kwargs['hessp'] = hessp_to_use
                
            res = minimize(fun_to_use, x0, **kwargs)
        except ValueError as e:
            raise e
            print(f'ERROR: {optimizer} failed, probably NaNs. Trying L-BFGS-B instead.')
            if 'maxfun' in options:
                del options['maxfun']
            res = minimize(fun_to_use, x0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': maxiter // 2, 'gtol': 1e-12, 'ftol': 1e-12},
                           bounds=bounds)
            print(res)
    if not res.success:
        if 'limit reached' not in res.message:
            print('Failed to converge for theta. Optimizer message:')
            print(res)
    p, kw = slicer(res.x,)
    for p_name, v in p.items():
        params[p_name] = v
    if return_funs:
        return params, res.fun, fun_aux
    return params, res.fun

from jax.flatten_util import ravel_pytree
from time import time

def find_lambda(A, ones, target_norm_sq, eye):
    tol = 1e-8
    low = 0.0
    A_base = A + 1e-8 * eye
    inv_low = jnp.linalg.solve(A_base, ones)
    norm_low_sq = jnp.sum(inv_low ** 2)
    if norm_low_sq <= target_norm_sq + tol:
        return low
    
    high = 1.0
    while True:
        inv_high = jnp.linalg.solve(A + high * eye + 1e-8 * eye, ones)
        norm_high_sq = jnp.sum(inv_high ** 2)
        if norm_high_sq < target_norm_sq + tol:
            break
        high *= 2.0
    
    for _ in range(30):
        mid = (low + high) / 2
        inv_mid = jnp.linalg.solve(A + mid * eye + 1e-8 * eye, ones)
        norm_mid_sq = jnp.sum(inv_mid ** 2)
        if norm_mid_sq > target_norm_sq:
            low = mid
        else:
            high = mid
    
    return mid

def anderson_acceleration(history_x, history_g, m_aa, unravel, bounds=None, dampen=True, s=0, alpha=1.2, kappa=25, max_neg=3):
    """Apply Anderson acceleration to accelerate convergence."""
    if len(history_x) < 2:
        return None, False, s
    
    mk = min(len(history_x), m_aa)
    F = jnp.stack([g - x for g, x in zip(history_g[-mk:], history_x[-mk:])]).T  # (dim, mk)
    A = F.T @ F
    ones = jnp.ones(mk)
    
    lambda_reg = 0.0
    if dampen:
        delta = 1 / (1 + alpha ** (kappa - s))
        inv_A_ones_ls = jnp.linalg.solve(A + 1e-8 * jnp.eye(mk), ones)
        norm_ls_sq = jnp.sum(inv_A_ones_ls ** 2)
        target_norm_sq = delta * norm_ls_sq
        lambda_reg = find_lambda(A, ones, target_norm_sq, jnp.eye(mk))
    
    A_reg = A + lambda_reg * jnp.eye(mk) + 1e-8 * jnp.eye(mk)
    inv_A_ones = jnp.linalg.solve(A_reg, ones)
    denom = jnp.dot(ones, inv_A_ones)
    alpha = inv_A_ones / denom
    extrapolated_flat = sum(alpha[i] * history_g[-mk + i] for i in range(mk))
    full_extra = unravel(extrapolated_flat)
    
    if bounds is not None:
        for key in full_extra:
            if key in bounds:
                bound = bounds[key]
                if isinstance(bound, list) and isinstance(bound[0], list):
                    lb, ub = bound[0]
                else:
                    lb, ub = bound
                if lb is not None:
                    full_extra[key] = jnp.maximum(full_extra[key], lb)
                if ub is not None:
                    full_extra[key] = jnp.minimum(full_extra[key], ub)
    
    return full_extra, True, s


        

def fit_em(data: jnp.ndarray, num_mixtures: int, 
           num_iters: int = 100, ftol: float = 1e-6,num_nodes: int = 20, theta=None, r=None, p=None, w=None,
           pi=None, max_nodes=125, nodes_add: int = 10, max_nodes_true: int = None, 
           dampen_aa: bool = False, bounds: dict = None,
           num_warmup: int = 3, high_precision: bool = True, fixed_beta: float = None):
    theta, r, p, w, pi = get_starting_values(data, num_mixtures, theta, r, p, w, pi, fixed_beta=fixed_beta)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta}
    # optimizers = ['TNC', 'SLSQP']
    optimizers = ['SLSQP', 'TNC']
    optimizer_warmup = 'SLSQP'
    swapped = 0
    
    Q_fun = partial(Q_dist, num_nodes=num_nodes, high_precision=high_precision, fixed_beta=fixed_beta)
    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                    num_nodes=max_nodes_true if max_nodes_true else num_nodes, fixed_beta=fixed_beta)
    prev_loglik = -float('inf')
    funs = None
    last_good_params = None 
    history_x = []
    history_g = []
    bounds2 = get_bounds_()
    m_aa = 5
    s_aa = 1.5
    alpha_aa = 1.2
    kappa_aa = 60
    max_neg_aa = 3
    
    for n_iter in range(num_iters):
        is_warmup = n_iter < num_warmup
        try:
            if is_warmup:
                print(f'\tEM iteration: {n_iter+1}/{num_iters} [Warmup]')
            else:
                print(f'\tEM iteration: {n_iter+1}/{num_iters}')
            
            if n_iter == 0:
                print('First iteration always takes much longer than later ones due to the necessity of JIT.')
                print('If you are waiting for 5 minutes just for a first iteration to finish, that\'s probably fine.')
            t0 = time()
            print('E-step...')
            full_x = {**params, 'w': w, 'pi': pi}
            flat_x, unravel = ravel_pytree(full_x)
            T, C, R, loglik = calc_responsibilities(pi=pi, w=w, **params)
            if n_iter > 0:
                print(loglik, prev_loglik)
            w, pi = update_weights(T, C)
            w = w[:-1]
            print(f"theta = {params['theta']}, w = {w}")
            

            if loglik < prev_loglik - 1e-5:
                if last_good_params is not None:
                    params = last_good_params
                prev_loglik = -float('inf')
                if num_nodes > max_nodes:
                    print('EM: stopped due to inability to decrease negloglik.')
                    break
                num_nodes = num_nodes + nodes_add
                print(f'Failure to maximize likelihood. Increasing number of quadrature nodes to {num_nodes}.')
                Q_fun = partial(Q_dist, num_nodes=num_nodes, high_precision=high_precision, fixed_beta=fixed_beta)
                if not max_nodes_true:
                    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                                    num_nodes=num_nodes, fixed_beta=fixed_beta)
                funs = None
                history_x = []
                history_g = []
                s_aa = 0
                continue
            last_good_params = params
  
            if not is_warmup and ((loglik - prev_loglik) < ftol):
                if (swapped == len(optimizers)):
                    print('EM converged.', loglik - prev_loglik)
                    break
                swapped = swapped + 1
                optimizers = [optimizers[-1]] + optimizers[:-1]
                print(f'EM seems to be converging. Trying optimization with {optimizers[0]}.')
            else:
                swapped = 0
            prev_loglik = loglik
            print('M-step...')
           
            params, _, funs = optimize(Q_fun, data, processed_data, params=params, num_iters=10 + n_iter * 10,
                                       R=R, fun_aux=funs, return_funs=True,
                                       optimizer=optimizer_warmup if is_warmup else optimizers[0],
                                       warmup=n_iter < num_warmup)
            full_g = {**params, 'w': w, 'pi': pi}
            flat_g = ravel_pytree(full_g)[0]
            history_x.append(flat_x)
            history_g.append(flat_g)
            if len(history_x) > m_aa:
                history_x.pop(0)
                history_g.pop(0)
            full_extra, used_aa, s_aa = anderson_acceleration(history_x, history_g, m_aa, unravel, bounds=bounds2,
                                                               dampen=dampen_aa, s=s_aa, alpha=alpha_aa,
                                                               kappa=kappa_aa, max_neg=max_neg_aa)
            if used_aa:
                T, C, R, extra_loglik = calc_responsibilities(pi=full_extra['pi'], w=full_extra['w'],
                                                              **{k: v for k, v in full_extra.items() if k not in ['w', 'pi']})
                if (extra_loglik >= loglik):
                    params = {k: v for k, v in full_extra.items() if k not in ['w', 'pi']}
                    w = full_extra['w']
                    pi = full_extra['pi']
                    loglik = extra_loglik
                    print('Used Anderson acceleration.')
                    if dampen_aa:
                        s_aa += 1
                else:
                    if dampen_aa:
                        s_aa = max(s_aa - 1, -max_neg_aa)
        
            last_good_params = params
            if n_iter > 0:
                t = time() - t0
                print(f'Took {t:.3f}')
        except KeyboardInterrupt:
            print('EM algorithm stopped.')
            break
    
    T, C, R, loglik = calc_responsibilities(**params, pi=pi, w=w)
    w, pi = update_weights(T, C)
    params['w'] = w
    params['pi'] = pi
    return params, R, loglik, num_nodes

def fit_mle(data: jnp.ndarray, num_mixtures: int, 
           ftol: float = 1e-5, num_nodes: int = 20, theta=None, r=None, p=None, w=None,
           pi=None,  high_precision: bool = True, fixed_beta: float = None):
    theta, r, p, w, pi = get_starting_values(data, num_mixtures, theta, r, p, w, pi, fixed_beta=fixed_beta)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta, 'w': w, 'pi': pi}
    
    fun = partial(negloglik, num_nodes=num_nodes, high_precision=high_precision, fixed_beta=fixed_beta)
    fun_aux = None

    t = float('inf')
    for i in range(3):
        for opt in ['SLSQP', 'TNC', 'L-BFGS-B']:
            print(i+1, opt)
            params_c, t_c, fun_aux = optimize(fun=fun, data=data, processed_data=processed_data,
                                              ftol=ftol, params=params, num_iters=10000, return_funs=True,
                                              fun_aux=fun_aux,
                                              optimizer=opt,)
            if t_c < t:
                params = params_c
                t = t_c
                print(i+1, t_c, '[improved]')
            else:
                print(i+1, t_c)
    T, C, R, loglik = compute_responsibilities(processed_data=processed_data,
                                               num_nodes=num_nodes,
                                               fixed_beta=fixed_beta,
                                               **params)
    w = params['w']
    if len(w) == num_mixtures - 1:
        w = jnp.append(w, 1.0 - w.sum())
        params['w'] = w
    return params, R, loglik, num_nodes


def calc_tau_means(data, params, responsibilities, 
                   batch_size=32, num_nodes=100,
                   zero_inflation: bool = False, fixed_beta: float = None):
    if not zero_inflation:
        responsibilities = responsibilities / responsibilities.sum(axis=-1, keepdims=True)
    theta, r, p = params['theta'], params['r'], params['p']
    B, M, N = data.shape
    processed = prepare_data(data)
    momfun = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, 
                      r=r, p=p, theta=theta, high_precision=True, fixed_beta=fixed_beta)
    
    moments = jnp.identity(M)
    moments = jnp.vstack((jnp.zeros((1, M)), moments))
    logmoments = jax.lax.map(lambda m: momfun(moments=m), moments)
    moments = jnp.exp(logmoments[1:] - logmoments[:1])
    return (responsibilities * moments).sum(axis=-1).transpose(1, 0, 2)

def estimate_lambda(data, params, responsibilities, batch_size: int = 32,
                    num_nodes: int = 100, fixed_beta: float = None) -> tuple[np.ndarray, np.ndarray]:
    B, M, N = data.shape
    theta, r, p = params['theta'], params['r'], params['p']
    processed = prepare_data(data)
    momfun_ = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, 
                      r=r, p=p, theta=theta, high_precision=True, fixed_beta=fixed_beta)
    momfun = lambda x: momfun_(lambda_moment=x)
    logmarginal = momfun(0)
    first_moment = jnp.exp(momfun(1) - logmarginal)
    second_moment = jnp.exp(momfun(2) - logmarginal)
    mean = (first_moment * responsibilities).sum(axis=-1)
    var = (responsibilities * (second_moment - first_moment ** 2)).sum(axis=-1)
    return mean, var

@partial(jax.jit, static_argnames=('sample_ind', 'normalized', 'fixed_beta'))
def _estimate_log_posterior_jit(tau, sample_ind, X, r, p, w, pi, 
                                phi, logweights, taus_quad, normalized,
                                theta, fixed_beta=None):
    B, M, N = X.shape
    K = r.shape[-1]
    
    alpha = theta[0]
    if fixed_beta is None:
        beta = theta[1]
    else:
        beta = fixed_beta
        
    lambda_div = M + 1 / beta
    log_phi_hat = jnp.log(phi) - jnp.log(lambda_div)
    
    loggammataus = gammaln(taus_quad + 1)
    log_phi_poi = -loggammataus[..., None] + taus_quad[..., None] * log_phi_hat
    
    num_nodes = phi.shape[0]
    log_A_sum = jnp.zeros((B, N, K, num_nodes))
    
    for m in range(M):
        if m == sample_ind:
            continue
            
        rtau = r[:, m, :, None] * taus_quad[None, None, :]
        is_rtau_zero = rtau == 0
        rtau = jnp.where(is_rtau_zero, 1.0, rtau)
        gammaln_rtau = gammaln(rtau)
        logp = jnp.log(p[:, m, :, None]) * rtau
        log1p = jnp.log1p(-p[:, m, :, None])
        
        rtau_exp = rtau[:, None, :, :]
        gammaln_rtau_exp = gammaln_rtau[:, None, :, :]
        is_rtau_zero_exp = is_rtau_zero[:, None, :, :]
        logp_exp = logp[:, None, :, :]
        log1p_exp = log1p[:, None, :, :]
        
        xs = X[:, m, :][:, :, None, None]
        
        logpmf = safe_nb_logpmf_precomp(xs, logp_exp, log1p_exp, rtau_exp, gammaln_rtau_exp, is_rtau_zero_exp)
        logpmf_flat = logpmf.reshape(B * N, K, len(taus_quad))
        
        log_A_m_flat = log_einsum_exp(logpmf_flat, log_phi_poi)
        log_A_m = log_A_m_flat.reshape(B, N, K, num_nodes)
        
        log_A_sum = log_A_sum + log_A_m

    tau_jnp = jnp.asarray(tau)
    T_user = len(tau_jnp)

    rtau_s = r[:, sample_ind, :, None] * tau_jnp[None, None, :]
    is_rtau_zero_s = rtau_s == 0
    rtau_s = jnp.where(is_rtau_zero_s, 1.0, rtau_s)
    gammaln_rtau_s = gammaln(rtau_s)
    logp_s = jnp.log(p[:, sample_ind, :, None]) * rtau_s
    log1p_s = jnp.log1p(-p[:, sample_ind, :, None])

    rtau_s_exp = rtau_s[:, None, :, :]
    gammaln_rtau_s_exp = gammaln_rtau_s[:, None, :, :]
    is_rtau_zero_s_exp = is_rtau_zero_s[:, None, :, :]
    logp_s_exp = logp_s[:, None, :, :]
    log1p_s_exp = log1p_s[:, None, :, :]

    xs_s = X[:, sample_ind, :][:, :, None, None]

    logpmf_s = safe_nb_logpmf_precomp(xs_s, logp_s_exp, log1p_s_exp, rtau_s_exp, gammaln_rtau_s_exp, is_rtau_zero_s_exp)

    # Calculate probability map specific to passed tau values under Gamma nodes
    log_phi_poi_user = -gammaln(tau_jnp + 1)[..., None] + tau_jnp[..., None] * log_phi_hat

    # 3. Combine replicates, specific sample, and Laguerre nodes 
    log_A_sum_w = log_A_sum + logweights[None, None, None, :]
    log_A_sum_w_flat = log_A_sum_w.reshape(B * N, K, num_nodes)

    log_B_mat = log_phi_poi_user.T

    log_marginal_lambda_flat = log_einsum_exp(log_A_sum_w_flat, log_B_mat)
    log_marginal_lambda = log_marginal_lambda_flat.reshape(B, N, K, T_user)

    log_D = logpmf_s + log_marginal_lambda

    # Marginalize over components K
    log_D_w = log_D + jnp.log(w)[None, None, :, None]
    log_E = logsumexp(log_D_w, axis=2)
    
    # Include non zero-inflation term (1 - pi)
    log_E = log_E + jnp.log1p(-pi)[:, None, None]

    # 4. Integrate zero-inflation considerations (acting broadly over sample condition X == 0)
    I_nb = jnp.all(X == 0, axis=1)
    zi_val = jnp.log(pi)[:, None] + jnp.where(I_nb, 0.0, -jnp.inf)
    
    zero_mask = (tau_jnp == 0)
    zi_expanded = jnp.where(zero_mask[None, None, :], zi_val[:, :, None], -jnp.inf)
    
    log_posterior = jnp.logaddexp(log_E, zi_expanded)


    if normalized:
        log_posterior = log_posterior - logsumexp(log_posterior, axis=-1, keepdims=True)

    # Adjust to `G x B x len(tau)` 
    # Current is (B, G, T)
    return log_posterior.transpose(1, 0, 2)


def estimate_log_posterior(tau: jnp.ndarray, sample_ind: int, result: LatentCountsResult, X: np.ndarray, normalized: bool = False,
                           right=None) -> np.ndarray:
    params = result.params
    num_nodes = result.num_nodes
    
    r = jnp.asarray(params['r'])
    p = jnp.asarray(params['p'])
    theta = jnp.asarray(params['theta'])
    
    if len(theta) == 1:
        fixed_beta = 1.0
    else:
        fixed_beta = None
        

    w = jnp.asarray(params['w'])
    if len(w) == r.shape[-1] - 1:
        w = jnp.append(w, jnp.clip(1.0 - w.sum(), 1e-12, 1.0))
        

    if 'pi' in params:
        pi = jnp.asarray(params['pi'])
    else:
        pi = jnp.zeros(X.shape[0])
    pi = jnp.clip(pi, 1e-12, 1.0 - 1e-12)
    
    X_jnp = jnp.asarray(X)
    B, M, N = X_jnp.shape
    

    alpha = float(theta[0])
    phi, logweights = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha - 1, rule=Rules.GenLaguerre)
    

    eps = 1e-9
    
    if right is None:

        max_lambda = (num_nodes * 4 + 2 * MAX_POI - 2) / M
        
  
        p_np, r_np = np.asarray(p), np.asarray(r)
        C = max_lambda * (p_np ** r_np)  # Shape: B x M x K
    
        X_max = np.max(X, axis=2)[:, :, None]  
        
        C_flat = C.flatten()
        r_flat = r_np.flatten()
        X_flat = np.broadcast_to(X_max, C.shape).flatten()
        
        tau = C_flat + X_flat / r_flat + 1.0  # Safe initial guess
        for _ in range(10):  
            term1 = 1.0 + X_flat / (r_flat * tau + 1e-12)
            F = tau - C_flat * (term1 ** r_flat)
            F_prime = 1.0 + C_flat * X_flat / (tau**2 + 1e-12) * (term1 ** (r_flat - 1.0))
            
            tau = np.maximum(tau - F / F_prime, C_flat) # tau cannot be smaller than C
            
        max_tau_star = float(np.max(tau))
        
        safe_mu = max_tau_star * 1.05 + 1.0
        right = int(scp_poisson.isf(eps, safe_mu))
    
    taus_quad = jnp.arange(0, right + 1)
    
    log_posterior = _estimate_log_posterior_jit(
        tau=jnp.asarray(tau),
        sample_ind=sample_ind,
        X=X_jnp,
        r=r, p=p, w=w, pi=pi,
        phi=phi, logweights=logweights, taus_quad=taus_quad,
        normalized=normalized,
        theta=theta, fixed_beta=fixed_beta
    )
    
    return np.asarray(log_posterior)


def infer_latent_counts(data: np.ndarray, num_mixture_components: int,
                        em_max_iter=40, 
                        em_warmup_iters=0,
                        mle_warmup: bool = True,
                        ftol=1e-6, use_cuda: bool=True,
                        num_nodes: int = 30, max_nodes: int = 120,
                        max_nodes_true: int = None, 
                        high_precision: bool = True,
                        multi_precision: bool = False,
                        prev_filename: str = None,
                        beta: float = None) -> LatentCountsResult:
    jax.config.update('jax_enable_x64', True)
    if use_cuda:
        jax.config.update('jax_platforms', 'cuda,cpu') 
    else:
        jax.config.update('jax_platforms', 'cpu') 
    if prev_filename is not None:
        from dill import load
        with open(prev_filename, 'rb') as f:
            prev_res : LatentCountsResult = load(f)
            params = prev_res.params
            if beta == 1.0 and len(params['theta']) == 2:
                beta = None
    else:
        params, _ = fit_nb(data, num_mixtures=num_mixture_components, verbose=False, max_iter=100)
    if multi_precision:
        precisions = [False, True]
    elif high_precision:
        precisions = [True]
    else:
        precisions = [False]
    for prec in precisions:
        if mle_warmup:
            params, R, loglik, num_nodes = fit_mle(data, num_mixtures=num_mixture_components,
                                                   ftol=ftol, 
                                                   num_nodes=num_nodes,
                                                   high_precision=prec,
                                                   fixed_beta=beta,
                                                   **params)
        if em_max_iter:
            params, R, loglik, num_nodes = fit_em(data, num_mixtures=num_mixture_components, 
                                                   num_iters=em_max_iter, ftol=ftol, 
                                                   num_warmup=em_warmup_iters *(not mle_warmup),
                                                   num_nodes=num_nodes, max_nodes=max_nodes, max_nodes_true=max_nodes_true,
                                                   high_precision=prec, fixed_beta=beta,
                                                   **params)
        if len(precisions) > 1 and prec==False:    
            print('-' * 10)
            print('Running for high precision.')
            print('-' * 10)
        
        
    if max_nodes_true:
        num_nodes = max_nodes_true
    taus = calc_tau_means(data, params, R, num_nodes=num_nodes, fixed_beta=beta)
    taus = np.asarray(taus)
    lambdas, lambdas_var = estimate_lambda(data, params, R, fixed_beta=beta)
    R = np.asarray(R)
    loglik = float(loglik)
    params = {n: np.asarray(v) for n, v in params.items()}
    return LatentCountsResult(params=params, 
                              responsibilities=R,
                              counts=taus,
                              loglik=loglik,
                              num_nodes=num_nodes,
                              prospenity=lambdas,
                              prospenity_var=lambdas_var)
