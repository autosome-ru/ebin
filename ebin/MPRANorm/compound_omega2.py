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
    responsibilities_omega: np.ndarray
    counts: np.ndarray
    prospenity: np.ndarray
    prospenity_var: np.ndarray
    loglik: float
    num_nodes: int
    

MAX_POI = 60. 

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

def compute_charlier_bounds(num_genlag_nodes: int, eps: float, M: int, max_alpha: float = None,
                            safe_alpha=3.0,):
    from scipy.special import roots_genlaguerre
    if max_alpha is None:
        max_alpha = MAX_POI
    
    min_nodes = roots_genlaguerre(num_genlag_nodes, 0.0)[0] / M
    j = np.searchsorted(min_nodes, safe_alpha )
    max_nodes = roots_genlaguerre(num_genlag_nodes, max_alpha)[0] / M
    right = scp_poisson.isf(eps, (num_genlag_nodes * 4 + 2 * max_alpha - 2) / M,)
    num_c = int(2 * np.log2(right) + 1)
    right = scp_poisson.isf(eps, max_nodes[j])
    num = max(right, num_c)
    return int(num), j

def safe_nb_logpmf(x, n, p):
    is_n_zero = (n < 1e-14)
    log_pmf_at_zero_n = jnp.where(x == 0, 0.0, -jnp.inf)
    safe_n = jnp.where(is_n_zero, 1.0, n)
    log_pmf_at_nonzero_n = jax.scipy.stats.nbinom.logpmf(x, n=safe_n, p=p)

    return jnp.where(is_n_zero, log_pmf_at_zero_n, log_pmf_at_nonzero_n)

def safe_nb_logpmf_precomp(x, logp, log1p, rtau, gammaln_rtau, is_rtau_zero):
    log_pmf_at_zero_n = jnp.where(x == 0, 0.0, -jnp.inf)
    logpmf = gammaln(x + rtau) - gammaln_rtau - gammaln(x + 1) + logp
    logpmf = logpmf +  log1p * x
    return jnp.where(is_rtau_zero, log_pmf_at_zero_n, logpmf)

@partial(jax.jit, static_argnames=('num_nodes', 'eps',
                                   'high_precision', 'num_extra_omega_components', 'omega_poisson'))
def marginal_loglik_charlier(processed_data: list[list[tuple[jnp.ndarray, jnp.ndarray]]], r: jnp.ndarray, p: jnp.ndarray,
                             theta: jnp.ndarray, lambda_poi: jnp.ndarray = jnp.array([]), moments=None, lambda_moment: int = 0,
                             num_nodes=25, eps=1e-6,  make_valid_mixture=None,
                             high_precision: bool = True, num_extra_omega_components: int = 0, omega_poisson: bool = False):
    if num_extra_omega_components > 0:
        raise NotImplementedError("Charlier rules are not yet implemented with extra omega components.")
    # Standard logic falls back here for 0 extra components
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
    beta = 1.0
    
    phi, logweights = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha - 1 + lambda_moment,
                                                   rule=Rules.GenLaguerre, norm=True)
    lambda_div = M + 1 / beta
    phi_hat = phi / lambda_div

    c_nodes, j = compute_charlier_bounds(num_nodes, eps, M, max_alpha=MAX_POI)
    taus, tau_logweights = jax.vmap(compute_nodes_and_logweights, in_axes=(None, 0, None))(c_nodes, phi_hat[j:], Rules.Charlier)
    taus_ = jnp.arange(0, c_nodes)[None]
    tau_logweights_ = jax.scipy.stats.poisson.logpmf(taus_, phi_hat[:j][:, None])
    taus = jnp.append(jnp.repeat(taus_, j, axis=0), taus,  axis=-2)
    tau_logweights = jnp.append(tau_logweights_, tau_logweights,  axis=-2)

    loglik = jnp.zeros((B, K, N), dtype=prec_float)
    const = logsumexp(logweights + phi_hat * M) + jnp.log(M + 1 / beta) * lambda_moment
    
    if not high_precision:
        const = const.astype(jnp.float32)
        logweights = logweights.astype(jnp.float32)
        taus = taus.astype(jnp.float32)
        tau_logweights = tau_logweights.astype(jnp.float32)
        r = r.astype(jnp.float32)
        p = p.astype(jnp.float32)
        if moments is not None:
            moments = moments.astype(jnp.float32)

    def calc_marginal_nb(x, r, p, taus, logweights):
        rtau = r * taus
        return logsumexp(safe_nb_logpmf(x, rtau, p) + logweights)
    
    logpmf = jax.vmap(calc_marginal_nb, in_axes=(None, None, None, 0, 0))
    logpmf = jax.vmap(logpmf, in_axes=(0, None, None, None, None,))
    logpmf = jax.vmap(logpmf, in_axes=(None, 0, 0, None, None,)) 
    
    for b in range(B):
        marg_nb = jnp.zeros((K, N, num_nodes), dtype=prec_float)
        for m in range(M):
            x, inv_ind = processed_data[b][m]
            x = x.astype(prec_int)
            marg_nb_s = logpmf(x, r[b, m], p[b, m], taus, tau_logweights) + phi_hat[None, None]
            marg_nb = marg_nb + marg_nb_s[:, inv_ind]
        marg_nb = marg_nb + logweights[None, None]
        loglik = loglik.at[b].set(logsumexp(marg_nb, axis=-1)) 
        
    return jnp.transpose(loglik - const, (0, 2, 1))[..., None]

@partial(jax.jit, static_argnames=('batch_size',  'num_nodes', 'eps',
                                   'make_valid_mixture', 'charlier',
                                   'high_precision', 'num_extra_omega_components', 'omega_poisson'))
def marginal_loglik(processed_data: list[list[tuple[jnp.ndarray, jnp.ndarray]]], r: jnp.ndarray, p: jnp.ndarray,
                    theta: jnp.ndarray, lambda_poi: jnp.ndarray = jnp.array([]), moments=None, lambda_moment: int = 0,
                    batch_size=48, num_nodes=25, eps=1e-9, 
                    make_valid_mixture: bool = False, charlier: bool = False,
                    high_precision: bool = True, num_extra_omega_components: int = 0, omega_poisson: bool = False):
    if charlier:
        return marginal_loglik_charlier(processed_data, r, p, theta, lambda_poi, moments, lambda_moment=lambda_moment,
                                        batch_size=batch_size, 
                                        num_nodes=num_nodes, eps=eps, make_valid_mixture=make_valid_mixture,
                                        high_precision=high_precision, num_extra_omega_components=num_extra_omega_components,
                                        omega_poisson=omega_poisson)
    if high_precision:
        prec_int = jnp.int64
        prec_float = jnp.float64
    else:
        prec_int = jnp.int32
        prec_float = jnp.float32
        
    processed_data = processed_data[0]
    B, M, K = r.shape
    N = len(processed_data[0][0][-1])
    C = 1 + num_extra_omega_components
    beta = 1.0
    lambda_div = M + 1 / beta
    
    log_phi_hats = []
    log_weights = []
    consts = []
    sizes = []
    
    # 1. Base Gamma-Poisson component
    alpha_base = theta[0]
    phi_base, lw_base = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha_base - 1 + lambda_moment, rule=Rules.GenLaguerre)
    phi_hat_base = phi_base / lambda_div
    log_phi_hats.append(jnp.log(phi_hat_base))
    log_weights.append(lw_base)
    sizes.append(num_nodes)
    if make_valid_mixture:
        consts.append(logsumexp(lw_base + phi_hat_base * M) + jnp.log(lambda_div) * lambda_moment)
    else:
        consts.append(jnp.log(M * beta + 1) * alpha_base + gammaln(alpha_base) + jnp.log(lambda_div) * lambda_moment)

    # 2. Extra components
    for c in range(num_extra_omega_components):
        if omega_poisson:
            lam = lambda_poi[c]
            log_phi_hats.append(jnp.array([jnp.log(lam)]))
            log_weights.append(jnp.array([-M * lam]))
            consts.append(jnp.array(0.0) + lambda_moment * jnp.log(lam))
            sizes.append(1)
        else:
            alpha_extra = theta[c + 1]
            phi_extra, lw_extra = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha_extra - 1 + lambda_moment, rule=Rules.GenLaguerre)
            phi_hat_extra = phi_extra / lambda_div
            log_phi_hats.append(jnp.log(phi_hat_extra))
            log_weights.append(lw_extra)
            sizes.append(num_nodes)
            if make_valid_mixture:
                consts.append(logsumexp(lw_extra + phi_hat_extra * M) + jnp.log(lambda_div) * lambda_moment)
            else:
                consts.append(jnp.log(M * beta + 1) * alpha_extra + gammaln(alpha_extra) + jnp.log(lambda_div) * lambda_moment)

    log_phi_hat_all = jnp.concatenate(log_phi_hats)
    log_weights_all = jnp.concatenate(log_weights)
    total_nodes = sum(sizes)

    right = scp_poisson.isf(eps, (num_nodes * 4 + 2 * MAX_POI - 2) / M,)
    taus = jnp.arange(0, right + 1)
    loggammataus = gammaln(taus + 1)
    logtaus = jnp.log(taus)
    log_phi_poi = -loggammataus[..., None] + taus[..., None] * log_phi_hat_all
    
    rtau = r[..., None] * taus 
    is_rtau_zero = rtau == 0
    rtau = jnp.where(is_rtau_zero, 1.0, rtau)
    gammaln_rtau = gammaln(rtau)
    logp = jnp.log(p)[..., None] * rtau
    log1p = jnp.log1p(-p)[..., None]
    
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
        
    if not high_precision:
        log_weights_all = log_weights_all.astype(jnp.float32)
        log_phi_poi = log_phi_poi.astype(jnp.float32)
        logtaus = logtaus.astype(jnp.float32)
        taus = taus.astype(jnp.float32)
        r = r.astype(jnp.float32)
        p = p.astype(jnp.float32)
        rtau = rtau.astype(jnp.float32)
        gammaln_rtau = gammaln_rtau.astype(jnp.float32)
        logp = logp.astype(jnp.float32)
        log1p = log1p.astype(jnp.float32)
        consts = [c.astype(jnp.float32) for c in consts]
        if moments is not None:
            moments = moments.astype(jnp.float32)

    for b in range(B):
        marg_nb = jnp.zeros((N, K, total_nodes), dtype=prec_float)
        for m in range(M):
            x, inv_ind = processed_data[b][m]
            x = x.astype(prec_int)
            marg_nb_s = calc_marginal_nb_precomp(x, b, m, moment=moments)
            marg_nb = marg_nb + marg_nb_s[inv_ind]
            
        marg_nb_w = marg_nb + log_weights_all[None, None, :]
        
        loglik_all = []
        start = 0
        for i in range(C):
            end = start + sizes[i]
            ll_c = logsumexp(marg_nb_w[..., start:end], axis=-1) - consts[i]
            loglik_all.append(ll_c)
            start = end
            
        if b == 0:
            loglik_out = jnp.zeros((B, N, K, C), dtype=prec_float)
            
        loglik_out = loglik_out.at[b].set(jnp.stack(loglik_all, axis=-1))

    return loglik_out


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



def update_weights(T: jnp.ndarray, C: jnp.ndarray, T_Y: jnp.ndarray = None, eps=1e-12):
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

    if T_Y is not None and T_Y.shape[1] > 0:
        new_omega = jnp.mean(T_Y, axis=0)
        new_omega = jnp.clip(new_omega, eps, 1.0 - eps)
        omega_sum = jnp.sum(new_omega)
        new_omega = jnp.where(omega_sum >= 1.0, new_omega / (omega_sum + eps) * (1.0 - eps), new_omega)
        return new_weights, new_pi, new_omega

    return new_weights, new_pi, jnp.array([])

@partial(jax.jit, static_argnames=('num_nodes', 'valid_mixture', 'num_extra_omega_components', 'omega_poisson'))
def compute_responsibilities(processed_data: _prepared_data, 
                             r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray,
                             pi: jnp.ndarray, w: jnp.ndarray, num_nodes: int,
                             lambda_poi: jnp.ndarray = jnp.array([]),
                             omega: jnp.ndarray = jnp.array([]),
                             valid_mixture: bool = False,
                             eps=1e-12, num_extra_omega_components: int = 0, omega_poisson: bool = False):
    logP_all = marginal_loglik(processed_data, make_valid_mixture=valid_mixture,
                           num_nodes=num_nodes, r=r, p=p, theta=theta, lambda_poi=lambda_poi,
                           high_precision=True, num_extra_omega_components=num_extra_omega_components,
                           omega_poisson=omega_poisson) # Shape: (B, N, K, C)
    
    M = len(processed_data[0])
    B, N, K, C_components = logP_all.shape
    I_nb = processed_data[1]

    log_pi = jnp.log(jnp.clip(pi, eps, 1.0 - eps))
    log_one_minus_pi = jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))
    w = jnp.append(w, jnp.clip(1.0 - w.sum(), eps, 1.0 - eps))
    log_weights = jnp.log(w)
    
    if num_extra_omega_components > 0:
        omega_clipped = jnp.clip(omega, eps, 1.0 - eps)
        omega_sum = jnp.sum(omega_clipped)
        omega_clipped = jnp.where(omega_sum >= 1.0, omega_clipped / (omega_sum + eps) * (1.0 - eps), omega_clipped)
        omega_sum = jnp.sum(omega_clipped)
        
        log_omega_0 = jnp.log(jnp.clip(1.0 - omega_sum, eps, 1.0))
        log_omega_all = jnp.concatenate([jnp.array([log_omega_0]), jnp.log(omega_clipped)])
    else:
        log_omega_all = jnp.array([0.0])

    log_pi_b = log_pi[:, None, None, None]
    log_one_minus_pi_b = log_one_minus_pi[:, None, None, None]
    log_I_nb = jnp.log(I_nb[:, :, None, None])

    log_term1 = log_pi_b + log_I_nb
    log_term2_all = log_one_minus_pi_b + logP_all
    
    log_term_per_indiv_and_comp_all = jnp.logaddexp(log_term1, log_term2_all)
    log_P_Xn_given_Znkc = jnp.sum(log_term_per_indiv_and_comp_all, axis=0) # Shape: (N, K, C)

    log_P_Xn_given_Znk = logsumexp(log_omega_all[None, None, :] + log_P_Xn_given_Znkc, axis=2)
    log_joint_likelihoods = log_weights + log_P_Xn_given_Znk
    log_P_Xn = logsumexp(log_joint_likelihoods, axis=1)

    log_T = log_joint_likelihoods - log_P_Xn[:, None] 
    T = jnp.exp(log_T)

    log_R_all = (log_omega_all[None, None, None, :] + log_one_minus_pi_b + logP_all + 
                 log_weights[None, None, :, None] - log_P_Xn[None, :, None, None] + 
                 log_P_Xn_given_Znkc[None, :, :, :] - log_term_per_indiv_and_comp_all)
    R_all = jnp.exp(log_R_all)

    C_zi = 1.0 - jnp.sum(R_all, axis=(2, 3))
    
    if num_extra_omega_components > 0:
        log_P_Ync_Xn = log_omega_all + logsumexp(log_weights[:, None] + log_P_Xn_given_Znkc, axis=1)
        T_Y_all = jnp.exp(log_P_Ync_Xn - log_P_Xn[:, None])
        T_Y = T_Y_all[:, 1:]
    else:
        T_Y = jnp.zeros((N, 0))

    return T, C_zi, R_all, T_Y, log_P_Xn.sum() / (B * M * N)


def get_starting_values(data: jnp.ndarray, K: int, theta, r, p, w, pi, omega=None, lambda_poi=None, 
                        num_extra_omega_components: int = 0, omega_poisson: bool = False):
    B, M, N = data.shape
    if r is None:
        r = jnp.arange(1, K + 1, dtype=float).reshape(1, 1, -1)
        r = jnp.repeat(r, B, axis=0)
        r = jnp.repeat(r, M, axis=1)
    
    if theta is None:
        if not omega_poisson:
            theta = jnp.array([7.5] + [1.0] * num_extra_omega_components)
        else:
            theta = jnp.array([7.5])
    else:
        theta = jnp.atleast_1d(jnp.asarray(theta))
        
    mean_tau = theta[0]
    r = r / mean_tau

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
        
    if num_extra_omega_components > 0:
        if omega is None:
            omega = jnp.array([1e-2] * num_extra_omega_components)
        else:
            omega = jnp.atleast_1d(jnp.asarray(omega))
            
        if omega_poisson:
            if lambda_poi is None:
                lambda_poi = jnp.array([1.0] * num_extra_omega_components)
            else:
                lambda_poi = jnp.atleast_1d(jnp.asarray(lambda_poi))
        else:
            lambda_poi = jnp.array([])
    else:
        omega = jnp.array([])
        lambda_poi = jnp.array([])
        
    return theta, r, p, w, pi, omega, lambda_poi
    

@partial(jax.jit, static_argnames=('num_nodes', 'high_precision',
                                   'valid_mixture', 'num_extra_omega_components', 'omega_poisson'))
def Q_dist(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, R: jnp.ndarray,
           processed_data: _prepared_data, num_nodes: int, lambda_poi: jnp.ndarray = jnp.array([]),
           valid_mixture: bool = False, high_precision: bool = True, 
           num_extra_omega_components: int = 0, omega_poisson: bool = False):
    
    logP_all = marginal_loglik(processed_data, 
                           r=r, p=p, theta=theta, lambda_poi=lambda_poi, num_nodes=num_nodes,
                           make_valid_mixture=valid_mixture,
                           high_precision=high_precision, num_extra_omega_components=num_extra_omega_components,
                           omega_poisson=omega_poisson)
    
    return -(logP_all * R).sum(axis=-1).mean()
    
@partial(jax.jit, static_argnames=('num_nodes', 'high_precision',
                                   'valid_mixture', 'num_extra_omega_components', 'omega_poisson'))
def negloglik(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, w: jnp.ndarray, 
              pi: jnp.ndarray, processed_data: _prepared_data, num_nodes: int,  R=None,
              omega: jnp.ndarray = jnp.array([]), lambda_poi: jnp.ndarray = jnp.array([]),
           valid_mixture: bool = False, high_precision: bool = True,
           num_extra_omega_components: int = 0, omega_poisson: bool = False):
    
    M = len(processed_data[0][0])
    eps = 1e-15 if high_precision else 1e-9

    logP_all = marginal_loglik(processed_data, 
                           r=r, p=p, theta=theta, lambda_poi=lambda_poi, num_nodes=num_nodes,
                           make_valid_mixture=valid_mixture, num_extra_omega_components=num_extra_omega_components,
                           omega_poisson=omega_poisson)
    B, N, K, C = logP_all.shape
    
    log_pi_b = jnp.log(jnp.clip(pi, eps, 1.0 - eps))[:, None, None, None]
    log_one_minus_pi_b = jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))[:, None, None, None]
    
    I_nb = processed_data[-1]
    log_I_nb = jnp.log(I_nb)[:, :, None, None]
    
    log_term1 = log_pi_b + log_I_nb
    log_term2_all = log_one_minus_pi_b + logP_all
    
    log_term_per_indiv_and_comp_all = jnp.logaddexp(log_term1, log_term2_all)  
    log_P_Xn_given_Znkc = jnp.sum(log_term_per_indiv_and_comp_all, axis=0)  
    
    if num_extra_omega_components > 0:
        omega_clipped = jnp.clip(omega, eps, 1.0 - eps)
        omega_sum = jnp.sum(omega_clipped)
        omega_clipped = jnp.where(omega_sum >= 1.0, omega_clipped / (omega_sum + eps) * (1.0 - eps), omega_clipped)
        omega_sum = jnp.sum(omega_clipped)
        
        log_omega_0 = jnp.log(jnp.clip(1.0 - omega_sum, eps, 1.0))
        log_omega_all = jnp.concatenate([jnp.array([log_omega_0]), jnp.log(omega_clipped)])
    else:
        log_omega_all = jnp.array([0.0])
    
    log_P_Xn_given_Znk = logsumexp(log_omega_all[None, None, :] + log_P_Xn_given_Znkc, axis=2)

    if len(w):
        w_full = jnp.append(w, jnp.clip(1.0 - w.sum(), eps, 1.0 - eps))
        log_weights = jnp.log(w_full)
        log_joint_likelihoods = log_weights + log_P_Xn_given_Znk
        log_P_Xn = logsumexp(log_joint_likelihoods, axis=1) 
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
            'w': [[0.0, 1.0]], 'pi': [[0.0, 1.0]],
            'omega': [[0.0, 1.0]], 'lambda_poi': [[1e-5, MAX_POI]]}
    return base

def get_bounds(names, shapes):
    bounds = list()
    base = get_bounds_()
    for name, shape in zip(names, shapes):
        t = np.prod(np.asarray(shape))
        if name == 'theta':
            bounds.extend([[1e-3, MAX_POI]] * t)
        else:
            bounds.extend(base.get(name, []) * t)
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
        

def coordinate_descent(fun, data, processed_data: _prepared_data, params: dict, num_iters: int,
                       R=None, ftol: float = 1e-6, 
                       return_funs: bool = False, funs=None, maxiter: int = 250, verbose: bool = False):
    params = deepcopy(params)
    if funs is not None:
        rp_funs, (t_fun, t_names, t_shapes, t_bounds, t_slicer) = funs
        t_names = tuple(filter(lambda x: x not in ('r', 'p'), params.keys()))
        t_shapes = tuple([params[name].shape for name in t_names])
        t_bounds = get_bounds(t_names, t_shapes)
        num_mults = params['r'].shape[-1]
        t_bounds.extend([(0.1, 10)] * (2 * num_mults) )
        t_slicer = partial(param_slicer, names=t_names, shapes=t_shapes, mult=num_mults)
    else:
        rp_funs = list()
        f_rp_base = None
        data = data == 0
        for b, bin_list in enumerate(processed_data[0]):
            p = {'r': params['r'][b:b+1], 'p': params['p'][b:b+1] }
            if 'pi' in params:
                p['pi'] = params['pi'][b:b+1]
            names = tuple(p.keys())
            shapes = tuple([p[name].shape for name in names])
            tpr = [processed_data[0][b]], processed_data[1][b:b+1]
            
            if f_rp_base is None:
                f_rp = partial(param_slicer_wrapper, f=fun, names=names, shapes=shapes)
                f_rp = jax.jit(jax.value_and_grad(f_rp, argnums=0))
                f_rp_base = f_rp
            f_rp = partial(f_rp_base, processed_data=tpr)
            
            slicer = partial(param_slicer, names=names, shapes=shapes)
            rp_funs.append((f_rp, slicer, get_bounds(names, shapes), names, (b, slice(None))))
    
        t_names = tuple(filter(lambda x: x not in ('r', 'p', 'pi'), params.keys()))
        t_shapes = tuple([params[name].shape for name in t_names])
        t_bounds = get_bounds(t_names, t_shapes)
        num_mults = params['r'].shape[-1]
        t_bounds.extend([(0.1, 10)] * (2 * num_mults))
        t_slicer = partial(param_slicer, names=t_names, shapes=t_shapes, mult=num_mults)
        t_fun = partial(param_slicer_wrapper, f=fun, names=t_names, shapes=t_shapes,
                          mult=num_mults, processed_data=processed_data)
        t_fun = jax.jit(jax.value_and_grad(t_fun, argnums=0))
    
    prev_fun = float('inf')
    from tqdm import tqdm
    for n in range(num_iters):
        try:
            if verbose:
                print(f'\tCD iteration: {n+1}/{num_iters}')
                
            x0 = list()
            for name in t_names:
                x0.extend(jnp.asarray(params[name]).flatten())
            x0 = jnp.array(x0)
            x0 = jnp.append(x0, jnp.ones(2 * num_mults, dtype=float))
            fixed_params = {'r': params['r'], 'p': params['p']}
            if 'pi' in params:
                fixed_params['pi'] = params['pi']
            if R is None:
                fun = partial(t_fun, **fixed_params)
            else:
                fun = partial(t_fun, R=R, **fixed_params)
            with warnings.catch_warnings(action="ignore"):
                res = minimize(fun, x0, jac=True, method='SLSQP', 
                               options={'maxiter': maxiter, 
                                        'ftol': 1e-10
                                        },
                               bounds=t_bounds)
            if not res.success:
                if 'limit reached' not in res.message:
                    print('CD failed to converge for theta. Optimizer message:')
                    print(res)
            cur_fun = -res.fun
            p, kw = t_slicer(res.x, **fixed_params)
            for p, v in p.items():
                params[p] = v
            params['r'] = kw['r']
            params['p'] = kw['p']
            gc.collect()
            for funr, slicer, bounds, names, (b, m_slc) in tqdm(rp_funs):
                x0 = list()    
                for name in names:
                    try:
                        x0.extend(params[name][b:b+1, m_slc].flatten())
                    except IndexError:
                        assert name == 'pi'
                        x0.extend(params[name][b:b+1].flatten())
                x0 = jnp.asarray(x0)
                fixed_params = {'theta': params['theta']}
                if 'lambda_poi' in params:
                    fixed_params['lambda_poi'] = params['lambda_poi']
                if 'w' in params:
                    fixed_params['w'] = params['w']
                if 'omega' in params:
                    fixed_params['omega'] = params['omega']
                if R is None:
                    fun = partial(funr, **fixed_params)
                else:
                    fun = partial(funr, R=R[b:b+1], **fixed_params)
                with warnings.catch_warnings(action="ignore"):
                    res = minimize(fun, x0, jac=True, method='TNC',
                                   bounds=bounds,
                                   options={'maxiter': maxiter, 
                                            'ftol': 1e-10
                                            })
                p, _ = slicer(res.x)
                gc.collect()
                for n in names:
                    try:
                        params[n] = jnp.asarray(params[n]).at[b:b+1, m_slc].set(p[n])
                    except IndexError:
                        assert n == 'pi'
                        params[n] = jnp.asarray(params[n]).at[b:b+1].set(p[n])
            if verbose:
                print(f'Cur fun: {cur_fun:6f}\tPrev fun: {prev_fun:6f}')
            if np.abs(prev_fun - cur_fun) < ftol:
                if verbose:
                    print('CD converged.')
                break
            prev_fun = cur_fun
        except KeyboardInterrupt as e:
            if verbose:
                print('CD stopped.')
                break
            else:
                raise e
    if return_funs:
        funs = rp_funs, (t_fun, t_names, t_shapes, t_bounds, t_slicer)
        return params, res.fun, funs
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
           num_iters: int = 100, ftol: float = 1e-6, cd: bool = False, num_nodes: int = 20, theta=None, r=None, p=None, w=None,
           pi=None, omega=None, lambda_poi=None, max_nodes=125, nodes_add: int = 10, max_nodes_true: int = None, 
           valid_mixture: bool = True, dampen_aa: bool = False, bounds: dict = None,
           num_warmup: int = 3, high_precision: bool = True, num_extra_omega_components: int = 0, omega_poisson: bool = False):
    theta, r, p, w, pi, omega, lambda_poi = get_starting_values(data, num_mixtures, theta, r, p, w, pi, omega, lambda_poi, 
                                                                num_extra_omega_components, omega_poisson)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta}
    if num_extra_omega_components > 0 and omega_poisson:
        params['lambda_poi'] = lambda_poi
        
    optimizers = ['SLSQP', 'TNC']
    optimizer_warmup = 'SLSQP'
    swapped = 0
    
    Q_fun = partial(Q_dist, num_nodes=num_nodes,
                    valid_mixture=valid_mixture, high_precision=high_precision,
                    num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                    valid_mixture=valid_mixture, 
                                    num_nodes=max_nodes_true if max_nodes_true else num_nodes,
                                    num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
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
            t0 = time()
            print('E-step...')
            
            full_x = {**params, 'w': w, 'pi': pi}
            if num_extra_omega_components > 0:
                full_x['omega'] = omega
            flat_x, unravel = ravel_pytree(full_x)
            
            T, C, R_all, T_Y, loglik = calc_responsibilities(
                pi=pi, w=w, omega=omega, lambda_poi=lambda_poi,
                **{k: v for k, v in params.items() if k not in ['lambda_poi', 'omega']}
            )
            if n_iter > 0:
                print(loglik, prev_loglik)
            w, pi, omega = update_weights(T, C, T_Y)
            w = w[:-1]
            if num_extra_omega_components > 0:
                print(f"theta = {params['theta']}, omega = {omega}, w = {w}")
            else:
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
                Q_fun = partial(Q_dist, num_nodes=num_nodes,
                                valid_mixture=valid_mixture, 
                                num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
                if not max_nodes_true:
                    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                                    num_nodes=num_nodes,
                                                    valid_mixture=True,
                                                    num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
                funs = None
                history_x = []
                history_g = []
                s_aa = 0
                continue
            last_good_params = params
  
            if not is_warmup and ((loglik - prev_loglik) < ftol):
                if (swapped == len(optimizers) or cd):
                    print('EM converged.', loglik - prev_loglik)
                    break
                swapped = swapped + 1
                optimizers = [optimizers[-1]] + optimizers[:-1]
                print(f'EM seems to be converging. Trying optimization with {optimizers[0]}.')
            else:
                swapped = 0
            prev_loglik = loglik
            print('M-step...')
            if cd:
                params, _, funs = coordinate_descent(Q_fun, data, processed_data, params=params, num_iters=1, 
                                                     R=R_all, funs=funs, return_funs=True)
            else:
                params, _, funs = optimize(Q_fun, data, processed_data, params=params, num_iters=10 + n_iter * 10,
                                           R=R_all, fun_aux=funs, return_funs=True,
                                           optimizer=optimizer_warmup if is_warmup else optimizers[0],
                                           warmup=n_iter < num_warmup)
                
            full_g = {**params, 'w': w, 'pi': pi}
            if num_extra_omega_components > 0:
                full_g['omega'] = omega
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
                T, C, R_all, T_Y, extra_loglik = calc_responsibilities(
                    pi=full_extra['pi'], w=full_extra['w'],
                    omega=full_extra.get('omega', omega), 
                    lambda_poi=full_extra.get('lambda_poi', lambda_poi),
                    **{k: v for k, v in full_extra.items() if k not in ['w', 'pi', 'omega', 'lambda_poi']}
                )
                if (extra_loglik >= loglik):
                    params = {k: v for k, v in full_extra.items() if k not in ['w', 'pi', 'omega']}
                    w = full_extra['w']
                    pi = full_extra['pi']
                    if 'omega' in full_extra:
                        omega = full_extra['omega']
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
    
    T, C, R_all, T_Y, loglik = calc_responsibilities(
        pi=pi, w=w, omega=omega, lambda_poi=lambda_poi,
        **{k: v for k, v in params.items() if k not in ['lambda_poi', 'omega']}
    )
    w, pi, omega = update_weights(T, C, T_Y)
    params['w'] = w
    params['pi'] = pi
    if num_extra_omega_components > 0:
        params['omega'] = omega
    return params, R_all, T_Y, loglik, num_nodes

def fit_mle(data: jnp.ndarray, num_mixtures: int, 
           num_cd_iters: int = 100, ftol: float = 1e-5, num_nodes: int = 20, theta=None, r=None, p=None, w=None,
           pi=None, omega=None, lambda_poi=None, valid_mixture=True, cd: bool = False, high_precision: bool = True,
           num_extra_omega_components: int = 0, omega_poisson: bool = False):
    theta, r, p, w, pi, omega, lambda_poi = get_starting_values(data, num_mixtures, theta, r, p, w, pi, omega, lambda_poi, 
                                                                num_extra_omega_components, omega_poisson)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta, 'w': w, 'pi': pi}
    if num_extra_omega_components > 0:
        params['omega'] = omega
        if omega_poisson:
            params['lambda_poi'] = lambda_poi
    
    fun = partial(negloglik, num_nodes=num_nodes,
                  valid_mixture=valid_mixture, high_precision=high_precision,
                  num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    if cd:
        params, t  = coordinate_descent(fun=fun, data=data, processed_data=processed_data,
                                        ftol=ftol,
                                        params=params, num_iters=num_cd_iters,
                                        verbose=True)
    else:
        params, t = optimize(fun=fun, data=data, processed_data=processed_data,
                             ftol=ftol,
                                        params=params, num_iters=10000,
                                        optimizer='SLSQP',)
    
    T, C, R_all, T_Y, loglik = compute_responsibilities(valid_mixture=True,  processed_data=processed_data,
                                               num_nodes=num_nodes, num_extra_omega_components=num_extra_omega_components, 
                                               omega_poisson=omega_poisson, lambda_poi=lambda_poi,
                                               **params)
    w = params['w']
    if len(w) == num_mixtures - 1:
        w = jnp.append(w, 1.0 - w.sum())
        params['w'] = w
    return params, R_all, T_Y, loglik, num_nodes
    


def calc_tau_means(data, params, responsibilities, 
                   batch_size=32, num_nodes=100,
                   zero_inflation: bool = False, num_extra_omega_components: int = 0, omega_poisson: bool = False):
    theta, r, p = params['theta'], params['r'], params['p']
    lambda_poi = params.get('lambda_poi', jnp.array([]))
    B, M, N = data.shape
    processed = prepare_data(data)
    momfun = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, make_valid_mixture=True,
                      r=r, p=p, theta=theta, lambda_poi=lambda_poi, high_precision=True,
                      num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    
    moments = jnp.identity(M)
    moments = jnp.vstack((jnp.zeros((1, M)), moments))
    logmoments_all = jax.lax.map(lambda m: momfun(moments=m), moments)
    
    moments_all = jnp.exp(logmoments_all[1:] - logmoments_all[:1])
    R_all = responsibilities
    
    tau_means = (R_all[None, ...] * moments_all).sum(axis=(-1, -2)).transpose(1, 0, 2)
    return tau_means

def estimate_lambda(data, params, responsibilities, batch_size: int = 32,
                    num_nodes: int = 100, num_extra_omega_components: int = 0, omega_poisson: bool = False) -> tuple[np.ndarray, np.ndarray]:
    B, M, N = data.shape
    theta, r, p = params['theta'], params['r'], params['p']
    lambda_poi = params.get('lambda_poi', jnp.array([]))
    processed = prepare_data(data)
    momfun_ = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, make_valid_mixture=True,
                      r=r, p=p, theta=theta, lambda_poi=lambda_poi, high_precision=True,
                      num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    momfun = lambda x: momfun_(lambda_moment=x)
    
    logmarginal_all = momfun(0)
    
    log1_all = momfun(1)
    first_moment_all = jnp.exp(log1_all - logmarginal_all)
    
    log2_all = momfun(2)
    second_moment_all = jnp.exp(log2_all - logmarginal_all)
    
    R_all = responsibilities
    mean = (first_moment_all * R_all).sum(axis=(-1, -2))
    var = (R_all * (second_moment_all - first_moment_all ** 2)).sum(axis=(-1, -2))
    return mean, var

@partial(jax.jit, static_argnames=('sample_ind', 'normalized'))
def _estimate_log_posterior_jit(tau, sample_ind, X, r, p, w, pi, 
                                log_phi_hat_all, logweights_all, taus_quad, normalized):
    B, M, N = X.shape
    K = r.shape[-1]
    
    loggammataus = gammaln(taus_quad + 1)
    log_phi_poi = -loggammataus[..., None] + taus_quad[..., None] * log_phi_hat_all
    
    num_nodes_all = log_phi_hat_all.shape[0]
    log_A_sum = jnp.zeros((B, N, K, num_nodes_all))
    
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
        log_A_m = log_A_m_flat.reshape(B, N, K, num_nodes_all)
        
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

    log_phi_poi_user = -gammaln(tau_jnp + 1)[..., None] + tau_jnp[..., None] * log_phi_hat_all
    
    log_A_sum_w = log_A_sum + logweights_all[None, None, None, :]
    log_A_sum_w_flat = log_A_sum_w.reshape(B * N, K, num_nodes_all)

    log_B_mat = log_phi_poi_user.T

    log_marginal_lambda_flat = log_einsum_exp(log_A_sum_w_flat, log_B_mat)
    log_marginal_lambda = log_marginal_lambda_flat.reshape(B, N, K, T_user)

    log_D = logpmf_s + log_marginal_lambda

    log_D_w = log_D + jnp.log(w)[None, None, :, None]
    log_E = logsumexp(log_D_w, axis=2)
    
    log_E = log_E + jnp.log1p(-pi)[:, None, None]

    I_nb = jnp.all(X == 0, axis=1)
    zi_val = jnp.log(pi)[:, None] + jnp.where(I_nb, 0.0, -jnp.inf)
    
    zero_mask = (tau_jnp == 0)
    zi_expanded = jnp.where(zero_mask[None, None, :], zi_val[:, :, None], -jnp.inf)
    
    log_posterior = jnp.logaddexp(log_E, zi_expanded)

    if normalized:
        log_posterior = log_posterior - logsumexp(log_posterior, axis=-1, keepdims=True)

    return log_posterior.transpose(1, 0, 2)


def estimate_log_posterior(tau: jnp.ndarray, sample_ind: int, result: LatentCountsResult, X: np.ndarray, 
                           normalized: bool = False, num_extra_omega_components: int = 0, omega_poisson: bool = False) -> np.ndarray:
    params = result.params
    num_nodes = result.num_nodes
    
    r = jnp.asarray(params['r'])
    p = jnp.asarray(params['p'])
    theta = jnp.asarray(params['theta'])
    lambda_poi = jnp.asarray(params.get('lambda_poi', []))

    w = jnp.asarray(params['w'])
    if len(w) == r.shape[-1] - 1:
        w = jnp.append(w, jnp.clip(1.0 - w.sum(), 1e-12, 1.0))
        
    if 'pi' in params:
        pi = jnp.asarray(params['pi'])
    else:
        pi = jnp.zeros(X.shape[0])
    pi = jnp.clip(pi, 1e-12, 1.0 - 1e-12)
    
    omega = jnp.asarray(params.get('omega', []))
    
    X_jnp = jnp.asarray(X)
    B, M, N = X_jnp.shape
    
    beta = 1.0
    lambda_div = M + 1 / beta
    
    log_phi_hats = []
    log_weights = []
    
    alpha_base = float(theta[0])
    phi_base, lw_base = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha_base - 1, rule=Rules.GenLaguerre)
    log_phi_hats.append(jnp.log(phi_base) - jnp.log(lambda_div))
    
    if num_extra_omega_components > 0:
        omega_sum = jnp.sum(omega)
        log_omega_0 = jnp.log(1.0 - omega_sum)
        log_weights.append(lw_base + log_omega_0)
        
        for c in range(num_extra_omega_components):
            if omega_poisson:
                lam = float(lambda_poi[c])
                log_phi_hats.append(jnp.array([jnp.log(lam)]))
                log_weights.append(jnp.array([-M * lam + jnp.log(omega[c])]))
            else:
                alpha_extra = float(theta[c + 1])
                phi_extra, lw_extra = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha_extra - 1, rule=Rules.GenLaguerre)
                log_phi_hats.append(jnp.log(phi_extra) - jnp.log(lambda_div))
                log_weights.append(lw_extra + jnp.log(omega[c]))
    else:
        log_weights.append(lw_base)
        
    log_phi_hat_all = jnp.concatenate(log_phi_hats)
    logweights_all = jnp.concatenate(log_weights)
    
    eps = 1e-9
    right = int(scp_poisson.isf(eps, (num_nodes * 4 + 2 * MAX_POI - 2) / M))
    taus_quad = jnp.arange(0, right + 1)
    
    log_posterior = _estimate_log_posterior_jit(
        tau=jnp.asarray(tau),
        sample_ind=sample_ind,
        X=X_jnp,
        r=r, p=p, w=w, pi=pi,
        log_phi_hat_all=log_phi_hat_all, logweights_all=logweights_all,
        taus_quad=taus_quad,
        normalized=normalized
    )
    
    return np.asarray(log_posterior)


def infer_latent_counts(data: np.ndarray, num_mixture_components: int,
                        em_max_iter=40, 
                        em_warmup_iters=1,
                        mle_warmup: bool = True,
                        ftol=1e-6, use_cuda: bool=True,
                        num_nodes: int = 30, max_nodes: int = 120,
                        max_nodes_true: int = None, 
                        valid_mixture: bool = True,
                        high_precision: bool = True,
                        multi_precision: bool = False,
                        prev_filename: str = None,
                        num_extra_omega_components: int = 0,
                        omega_poisson: bool = False) -> LatentCountsResult:
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
            
            if num_extra_omega_components > 0:
                if 'theta' in params and jnp.asarray(params['theta']).size == 1 and not omega_poisson:
                    params['theta'] = jnp.concatenate([params['theta'], jnp.ones(num_extra_omega_components)])
                if 'omega' not in params:
                    params['omega'] = jnp.array([1e-2] * num_extra_omega_components)
                if omega_poisson and 'lambda_poi' not in params:
                    params['lambda_poi'] = jnp.array([1.0] * num_extra_omega_components)
    else:
        params, _ = fit_nb(data, num_mixtures=num_mixture_components, verbose=False, max_iter=100)
        
    if num_extra_omega_components > 0 and 'omega' not in params:
        params['omega'] = jnp.array([1e-2] * num_extra_omega_components)
        if omega_poisson and 'lambda_poi' not in params:
            params['lambda_poi'] = jnp.array([1.0] * num_extra_omega_components)
        
    if multi_precision:
        precisions = [False, True]
    elif high_precision:
        precisions = [True]
    else:
        precisions = [False]
    for prec in precisions:
        if mle_warmup:
            params, R_all, T_Y, loglik, num_nodes = fit_mle(data, num_mixtures=num_mixture_components,
                                                   num_cd_iters=em_max_iter, ftol=ftol, 
                                                   cd=False,
                                                   num_nodes=num_nodes,
                                                   valid_mixture=valid_mixture,
                                                   high_precision=prec,
                                                   num_extra_omega_components=num_extra_omega_components,
                                                   omega_poisson=omega_poisson,
                                                   **params)
        if em_max_iter:
            params, R_all, T_Y, loglik, num_nodes = fit_em(data, num_mixtures=num_mixture_components, 
                                                   num_iters=em_max_iter, ftol=ftol, 
                                                   num_warmup=em_warmup_iters *(not mle_warmup),
                                                   num_nodes=num_nodes, max_nodes=max_nodes, max_nodes_true=max_nodes_true,
                                                   valid_mixture=valid_mixture,
                                                   high_precision=prec,
                                                   num_extra_omega_components=num_extra_omega_components,
                                                   omega_poisson=omega_poisson,
                                                   **params)
        if len(precisions) > 1 and prec==False:    
            print('-' * 10)
            print('Running for high precision.')
            print('-' * 10)
        
        
    if max_nodes_true:
        num_nodes = max_nodes_true
    taus = calc_tau_means(data, params, R_all, num_nodes=num_nodes, 
                          num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    taus = np.asarray(taus)
    lambdas, lambdas_var = estimate_lambda(data, params, R_all, num_nodes=num_nodes, 
                                           num_extra_omega_components=num_extra_omega_components, omega_poisson=omega_poisson)
    
    R_total = np.asarray(R_all.sum(axis=-1))
    
    loglik = float(loglik)
    params = {n: np.asarray(v) for n, v in params.items()}
    return LatentCountsResult(params=params, 
                              responsibilities=R_total,
                              responsibilities_omega=np.asarray(T_Y),
                              counts=taus,
                              loglik=loglik,
                              num_nodes=num_nodes,
                              prospenity=lambdas,
                              prospenity_var=lambdas_var)
