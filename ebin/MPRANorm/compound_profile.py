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
    
MAX_POI = 50. 
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
    
    clip_val = 1e-200 if log_A.dtype == jnp.float64 else 1e-20
    C_safe_bwd = jnp.maximum(C_lin, clip_val)
    
    g_over_C = g / C_safe_bwd
    
    d_log_A = A_lin * jnp.einsum('nkv,tv->nkt', g_over_C, B_lin)
    d_log_B = B_lin * jnp.einsum('nkv,nkt->tv', g_over_C, A_lin)
    
    return d_log_A, d_log_B

log_einsum_exp.defvjp(log_einsum_exp_fwd, log_einsum_exp_bwd)



def safe_nb_logpmf_precomp(x, logp, log1p, rtau, gammaln_rtau, is_rtau_zero):
    log_pmf_at_zero_n = jnp.where(x == 0, 0.0, -jnp.inf)
    logpmf = gammaln(x + rtau) - gammaln_rtau - gammaln(x + 1) + logp
    logpmf = logpmf +  log1p * x
    return jnp.where(is_rtau_zero, log_pmf_at_zero_n, logpmf)


@partial(jax.jit, static_argnames=('batch_size',  'num_nodes', 'eps', 'high_precision', 'min_M'))
def marginal_loglik(processed_data: list[list[tuple[jnp.ndarray, jnp.ndarray]]], r: jnp.ndarray, p: jnp.ndarray,
                    theta: jnp.ndarray, U: jnp.ndarray, moments=None, lambda_moment: int = 0,
                    batch_size=48, num_nodes=25, eps=1e-9, 
                    high_precision: bool = True, min_M: float = 1.0):
    if high_precision:
        prec_int = jnp.int64
        prec_float = jnp.float64
    else:
        prec_int = jnp.int32
        prec_float = jnp.float32
        
    processed_data = processed_data[0]
    B, M, K = r.shape
    V = U.shape[0]
    N = len(processed_data[0][0][-1])
    alpha, = theta
    beta = 1.0
    phi, logweights = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha - 1 + lambda_moment,
                                                   rule=Rules.GenLaguerre)
    lambda_div = M + 1 / beta
    phi_hat = phi / lambda_div
    log_phi_hat = jnp.log(phi_hat)

    right = scp_poisson.isf(eps, (num_nodes * 4 + 2 * MAX_POI - 2) / min_M)
    taus = jnp.arange(0, right + 1)
    loggammataus = gammaln(taus + 1)
    logtaus = jnp.log(taus)
    
    # Pre-compute log_phi_poi safely incorporating profile scales per sample
    log_phi_hat_v = log_phi_hat[None, None, :] + jnp.log(U)[:, :, None] # (V, M, num_nodes)
    log_phi_poi = -loggammataus[None, None, :, None] + taus[None, None, :, None] * log_phi_hat_v[:, :, None, :]
    log_phi_poi_flat = log_phi_poi.transpose(1, 2, 0, 3).reshape(M, len(taus), V * num_nodes)
    
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
            
        res_flat = log_einsum_exp(logpmf, log_phi_poi_flat[m])
        return res_flat.reshape(-1, K, V, num_nodes)
        
    loglik = jnp.zeros((B, N, K, V), dtype=prec_float)
    const = logsumexp(logweights + phi_hat * M)
    const = const + jnp.log(lambda_div) * lambda_moment
    
    if not high_precision:
        const = const.astype(jnp.float32)
        logweights = logweights.astype(jnp.float32)

    for b in range(B):
        marg_nb = jnp.zeros((N, K, V, num_nodes), dtype=prec_float)
        for m in range(M):
            x, inv_ind = processed_data[b][m]
            x = x.astype(prec_int)
            marg_nb_s = calc_marginal_nb_precomp(x, b, m, moment=moments)
            marg_nb = marg_nb + marg_nb_s[inv_ind]
        marg_nb = marg_nb + logweights[None, None, None, :]
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


def update_weights(T_K: jnp.ndarray, T_V: jnp.ndarray, C: jnp.ndarray, eps=1e-12) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = T_K.shape[0]
    
    new_w = jnp.sum(T_K, axis=0) / N
    new_pi = jnp.sum(C, axis=1) / N

    new_w = jnp.clip(new_w, eps, 1.0 - eps)
    new_w /= jnp.sum(new_w)
    new_pi = jnp.clip(new_pi, eps, 1.0 - eps)

    if T_V.shape[1] > 1:
        new_omega = jnp.sum(T_V, axis=0) / N
        new_omega = jnp.clip(new_omega, eps, 1.0 - eps)
        new_omega /= jnp.sum(new_omega)
    else:
        new_omega = jnp.ones(1)

    return new_w, new_pi, new_omega


@partial(jax.jit, static_argnames=('num_nodes', 'min_M'))
def compute_responsibilities(processed_data: _prepared_data, 
                             r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray,
                             pi: jnp.ndarray, w: jnp.ndarray, omega: jnp.ndarray, U: jnp.ndarray,
                             num_nodes: int, eps=1e-12, min_M: float = 1.0):
    logP = marginal_loglik(processed_data, num_nodes=num_nodes, r=r, p=p, theta=theta, U=U, high_precision=True, min_M=min_M)
    M = len(processed_data[0])
    B, N, K, V = logP.shape
    I_nb = processed_data[1]

    log_pi = jnp.log(jnp.clip(pi, eps, 1.0 - eps))
    log_one_minus_pi = jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))
    
    # Strictly enforce simplex normalization
    if len(w) == K - 1:
        w = jnp.append(w, jnp.clip(1.0 - jnp.sum(w), eps, 1.0 - eps))
    else:
        w = jnp.clip(w, eps, 1.0 - eps)
    w = w / jnp.sum(w)
        
    if len(omega) == V - 1:
        omega = jnp.append(omega, jnp.clip(1.0 - jnp.sum(omega), eps, 1.0 - eps))
    else:
        omega = jnp.clip(omega, eps, 1.0 - eps)
    omega = omega / jnp.sum(omega)
        
    log_weights = jnp.log(w)[:, None] + jnp.log(omega)[None, :]

    log_pi_b = log_pi[:, None, None, None]
    log_one_minus_pi_b = log_one_minus_pi[:, None, None, None]
    log_I_nb = jnp.log(I_nb[:, :, None, None])

    log_term1 = log_pi_b + log_I_nb
    log_term2 = log_one_minus_pi_b + logP
    log_term_per_indiv_and_comp = jnp.logaddexp(log_term1, log_term2)

    log_P_Xn_given_Znk_Vn = jnp.sum(log_term_per_indiv_and_comp, axis=0)
    log_joint_likelihoods = log_weights[None, :, :] + log_P_Xn_given_Znk_Vn
    log_P_Xn = logsumexp(log_joint_likelihoods, axis=(1, 2))

    log_T = log_joint_likelihoods - log_P_Xn[:, None, None]
    T_joint = jnp.exp(log_T)

    log_R = log_T[None, :, :, :] + log_one_minus_pi_b + logP - log_term_per_indiv_and_comp
    R = jnp.exp(log_R)

    C = 1.0 - jnp.sum(R, axis=(2, 3))
    return T_joint, C, R, log_P_Xn.sum() / (B * M * N)


def get_starting_values(data: jnp.ndarray, K: int, theta, r, p, w, pi, 
                        num_profiles: int = 0, batch_map=None, custom_profiles=None, omega=None, profiles=None):
    B, M, N = data.shape
    if r is None:
        r = jnp.arange(1, K + 1, dtype=float).reshape(1, 1, -1)
        r = jnp.repeat(r, B, axis=0)
        r = jnp.repeat(r, M, axis=1)
    
    if theta is None:
        theta = jnp.array([7.5, ])
        mean_tau = jnp.prod(theta)
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
        
    if omega is None:
        if num_profiles > 0 and batch_map is not None:
            omega = jnp.ones(num_profiles + 1)
            omega = omega.at[0].set(0.8)
            omega = omega.at[1:].set(0.2 / num_profiles)
            omega = omega[:-1]
    else:
        # Enforce exactly num_profiles to trim bloated loads from previous pickles
        if num_profiles > 0:
            if len(omega) > num_profiles:
                omega = jnp.asarray(omega)[:num_profiles]
            elif len(omega) < num_profiles:
                diff = num_profiles - len(omega)
                omega = jnp.concatenate([jnp.asarray(omega), jnp.ones(diff) * 0.01])
            
    if profiles is None and num_profiles > 0 and batch_map is not None:
        if custom_profiles is not None:
            profiles = jnp.asarray(custom_profiles)
        else:
            from sklearn.cluster import KMeans
            from scipy.stats import spearmanr
            counts_mn = np.sum(data, axis=0)
            corr_matrix, _ = spearmanr(counts_mn, axis=1)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            N_c = batch_map.sum(axis=0)
            corr_C = (batch_map.T @ corr_matrix @ batch_map) / np.outer(N_c, N_c)
            
            num_p_actual = min(num_profiles, batch_map.shape[1])
            kmeans = KMeans(n_clusters=num_p_actual, n_init=10, random_state=0).fit(corr_C)
            labels = kmeans.labels_
            
            profiles_np = np.ones((num_profiles, batch_map.shape[1]))
            for p_idx in range(num_p_actual):
                profiles_np[p_idx, labels == p_idx] = 2.0
                profiles_np[p_idx, labels != p_idx] = 0.5
            profiles = jnp.asarray(profiles_np)
    elif profiles is not None and num_profiles > 0 and batch_map is not None:
        # Prevent stack inflation: ensure profiles is robustly cut down to raw parameter shape
        profiles = jnp.asarray(profiles)
        if profiles.shape[0] > num_profiles:
            profiles = profiles[-num_profiles:] 
        elif profiles.shape[0] < num_profiles:
            diff = num_profiles - profiles.shape[0]
            profiles = jnp.vstack([profiles, jnp.ones((diff, profiles.shape[1]))])
            
    return theta, r, p, w, pi, omega, profiles

def compute_U(raw_profiles, batch_map, M):
    if batch_map is None:
        return jnp.ones((1, M))
    base = jnp.ones((1, batch_map.shape[1]))
    if raw_profiles is not None:
        full_prof = jnp.vstack([base, raw_profiles])
    else:
        full_prof = base
    N_c = batch_map.sum(axis=0)
    sums = jnp.sum(full_prof * N_c, axis=1, keepdims=True)
    full_prof = full_prof / sums * M
    return full_prof @ batch_map.T

def get_full_profiles(raw_profiles, batch_map, M):
    base = np.ones((1, batch_map.shape[1]))
    if raw_profiles is not None:
        full_prof = np.vstack([base, np.asarray(raw_profiles)])
    else:
        full_prof = base
    N_c = batch_map.sum(axis=0)
    sums = np.sum(full_prof * N_c, axis=1, keepdims=True)
    return full_prof / sums * M

@partial(jax.jit, static_argnames=('num_nodes', 'high_precision', 'min_M'))
def Q_dist(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, R: jnp.ndarray, U: jnp.ndarray,
           processed_data: _prepared_data, num_nodes: int,
           high_precision: bool = True, min_M: float = 1.0):
    logP = marginal_loglik(processed_data, r=r, p=p, theta=theta, U=U, num_nodes=num_nodes,
                           high_precision=high_precision, min_M=min_M)
    
    # Safeguard against -inf * 0.0 generating NaNs during optimization
    safe_logP = jnp.where(jnp.isneginf(logP), -1e10, logP)
    return -(safe_logP * R).mean()

    

@partial(jax.jit, static_argnames=('num_nodes', 'high_precision', 'min_M'))
def negloglik(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, w: jnp.ndarray, pi: jnp.ndarray,
           omega: jnp.ndarray, U: jnp.ndarray, processed_data: _prepared_data, num_nodes: int,  R=None,
           high_precision: bool = True, min_M: float = 1.0):
    M = len(processed_data[0][0])
    eps = 1e-15 if high_precision else 1e-9
    
    logP = marginal_loglik(processed_data, r=r, p=p, theta=theta, U=U, num_nodes=num_nodes, min_M=min_M)
    B, N, K, V = logP.shape

    log_pi_b = jnp.log(jnp.clip(pi, eps, 1.0 - eps))[:, None, None, None]
    log_one_minus_pi_b = jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))[:, None, None, None]
    
    I_nb = processed_data[-1]
    log_I_nb = jnp.log(I_nb)[:, :, None, None]

    log_term1 = log_pi_b + log_I_nb
    log_term2 = log_one_minus_pi_b + logP
    log_term_per_indiv_and_comp = jnp.logaddexp(log_term1, log_term2) 
    
    log_P_Xn_given_Znk_Vn = jnp.sum(log_term_per_indiv_and_comp, axis=0) 
    
    # Strictly enforce simplex normalization to prevent artificial inflation
    if len(w) == K - 1:
        w_full = jnp.append(w, jnp.clip(1.0 - jnp.sum(w), eps, 1.0 - eps))
    else:
        w_full = jnp.clip(w, eps, 1.0 - eps)
    w_full = w_full / jnp.sum(w_full)
        
    if len(omega) == V - 1:
        omega_full = jnp.append(omega, jnp.clip(1.0 - jnp.sum(omega), eps, 1.0 - eps))
    else:
        omega_full = jnp.clip(omega, eps, 1.0 - eps)
    omega_full = omega_full / jnp.sum(omega_full)

    log_weights = jnp.log(w_full)[:, None] + jnp.log(omega_full)[None, :]
    log_joint_likelihoods = log_weights[None, :, :] + log_P_Xn_given_Znk_Vn
    log_P_Xn = logsumexp(log_joint_likelihoods, axis=(1, 2))

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
    params, out_kwargs = param_slicer(x, names, shapes, mult, **kwargs)
    
    batch_map = out_kwargs.pop('batch_map', None)
    fixed_profiles = out_kwargs.pop('fixed_profiles', None)
    M = out_kwargs.pop('M', 1)
    
    # Safely extract and remove 'profiles' from params so it isn't passed to the target function
    raw_prof = params.pop('profiles', fixed_profiles)
    U = compute_U(raw_prof, batch_map, M)
    
    return f(**params, U=U, **out_kwargs)

def get_bounds_():
    base = {'r': [[1e-5, None]], 'p': [[1e-9, 0.99]],
            'w': [[0.0, 1.0]], 'pi': [[0.0, 1.0]]}
    return base

def get_bounds(names, shapes):
    bounds = list()
    base = get_bounds_()
    for name, shape in zip(names, shapes):
        t = int(np.prod(np.asarray(shape)))
        if name == 'theta':
            bounds.extend([[1e-3, MAX_POI],])
        elif name == 'profiles':
            bounds.extend([[1e-3, 1e2]] * t)
        elif name == 'omega':
            bounds.extend([[1e-9, 1.0]] * t)
        else:
            b = base[name]
            bounds.extend(b * t)
    return bounds
    
def optimize(fun, data, processed_data: _prepared_data, params: dict, num_iters: int,
             R=None, ftol: float = 1e-9, 
             return_funs: bool = False, fun_aux=None, maxiter: int = 2500,
             optimizer='TNC', use_mults=False, warmup: bool = False,
             **kwargs):
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
                          mult=num_mults, processed_data=processed_data, **kwargs)
        grad_fn = jax.grad(fun_raw, argnums=0)
        
        @jax.jit
        def fun_val_grad(x, R=None, warmup_vec=None, warmup_x=None):
            return fun_raw(x, R=R, warmup_vec=warmup_vec, warmup_x=warmup_x), grad_fn(x, R=R, warmup_vec=warmup_vec, warmup_x=warmup_x)
        
        @jax.jit
        def hessp_fd(x, v, R=None, warmup_vec=None, warmup_x=None):
            v_cast = jnp.asarray(v, dtype=x.dtype)
            v_norm = jnp.linalg.norm(v_cast)
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
            kw = {'jac': True, 'method': opt_method, 'options': options, 'bounds': bounds}
            if optimizer.lower() == 'newton' and hessp_opt is not None:
                kw['hessp'] = hessp_to_use
            res = minimize(fun_to_use, x0, **kw)
            
            # SAFEGUARD: If numeric drift worsened the Q-function, reject the step
            val_0, _ = fun_to_use(x0)
            if res.fun > val_0:
                res.x = x0
                res.fun = float(val_0)
                
        except ValueError as e:
            raise e
            
    if not res.success and res.fun != float(val_0):
        if 'limit reached' not in res.message:
            print('Failed to converge. Optimizer message:')
            print(res)
            
    p, _ = slicer(res.x,)
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
    F = jnp.stack([g - x for g, x in zip(history_g[-mk:], history_x[-mk:])]).T
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
           num_warmup: int = 3, high_precision: bool = True,
           batch_map=None, num_profiles=0, custom_profiles=None, estimable_profiles=True, omega=None, profiles=None):
           
    theta, r, p, w, pi, omega, profiles = get_starting_values(data, num_mixtures, theta, r, p, w, pi, num_profiles, batch_map, custom_profiles, omega, profiles)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta}
    if estimable_profiles and profiles is not None:
        params['profiles'] = profiles
        
    fixed_profiles = profiles if not estimable_profiles else None
    M = data.shape[1]
    min_M = float(np.min(batch_map.sum(axis=0))) if batch_map is not None else float(M)
    
    optimizers = ['SLSQP', 'TNC']
    optimizer_warmup = 'SLSQP'
    swapped = 0
    
    Q_fun = partial(Q_dist, num_nodes=num_nodes, high_precision=high_precision, min_M=min_M)
    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                    num_nodes=max_nodes_true if max_nodes_true else num_nodes, min_M=min_M)
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
            if num_profiles > 0:
                full_x['omega'] = omega
                
            flat_x, unravel = ravel_pytree(full_x)
            
            U = compute_U(params.get('profiles', fixed_profiles), batch_map, M)
            
            # Explicit parameter unpacking to prevent unexpected 'profiles' kwarg
            T_joint, C, R, loglik = calc_responsibilities(
                pi=pi, w=w, omega=omega, U=U,
                r=params['r'], p=params['p'], theta=params['theta']
            )
            
            if n_iter > 0:
                print(loglik, prev_loglik)
                
            w, pi, omega = update_weights(jnp.sum(T_joint, axis=2), jnp.sum(T_joint, axis=1), C)
            w = w[:-1]
            if omega is not None and len(omega) > 1:
                omega = omega[:-1]
                
            print(f"theta = {params['theta']}, w = {w}")
            if num_profiles > 0:
                print(f"omega = {omega}")
            if batch_map is not None:
                prof_to_print = params.get('profiles', fixed_profiles)
                if prof_to_print is not None:
                    print(f"profiles =\n{prof_to_print}")

            if loglik < prev_loglik - 1e-5:
                if last_good_params is not None:
                    params = last_good_params
                prev_loglik = -float('inf')
                if num_nodes > max_nodes:
                    print('EM: stopped due to inability to decrease negloglik.')
                    break
                num_nodes = num_nodes + nodes_add
                print(f'Failure to maximize likelihood. Increasing number of quadrature nodes to {num_nodes}.')
                Q_fun = partial(Q_dist, num_nodes=num_nodes, min_M=min_M)
                if not max_nodes_true:
                    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                                    num_nodes=num_nodes, min_M=min_M)
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
                                       warmup=n_iter < num_warmup, batch_map=batch_map, fixed_profiles=fixed_profiles, M=M)
                                       
            full_g = {**params, 'w': w, 'pi': pi}
            if num_profiles > 0:
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
                U_aa = compute_U(full_extra.get('profiles', fixed_profiles), batch_map, M)
                params_aa = {k: v for k, v in full_extra.items() if k not in ['w', 'pi', 'omega']}
                
                T_joint, C, R, extra_loglik = calc_responsibilities(
                    pi=full_extra['pi'], w=full_extra['w'], 
                    omega=full_extra.get('omega', jnp.ones(1)), U=U_aa,
                    r=params_aa['r'], p=params_aa['p'], theta=params_aa['theta']
                )
                if (extra_loglik >= loglik):
                    params = params_aa
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
    
    U = compute_U(params.get('profiles', fixed_profiles), batch_map, M)
    T_joint, C, R, loglik = calc_responsibilities(
        pi=pi, w=w, omega=omega, U=U,
        r=params['r'], p=params['p'], theta=params['theta']
    )
    w, pi, omega = update_weights(jnp.sum(T_joint, axis=2), jnp.sum(T_joint, axis=1), C)
    params['w'] = w
    params['pi'] = pi
    if num_profiles > 0:
        params['omega'] = omega
        
    return params, R, loglik, num_nodes

def fit_mle(data: jnp.ndarray, num_mixtures: int, 
           ftol: float = 1e-5, num_nodes: int = 20, theta=None, r=None, p=None, w=None,
           pi=None,  high_precision: bool = True, batch_map=None, num_profiles=0, custom_profiles=None, estimable_profiles=True, omega=None, profiles=None):
           
    theta, r, p, w, pi, omega_start, prof_start = get_starting_values(data, num_mixtures, theta, r, p, w, pi, num_profiles, batch_map, custom_profiles, omega, profiles)
    if omega is None:
        omega = omega_start
    if profiles is None:
        profiles = prof_start
        
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta, 'w': w, 'pi': pi}
    if num_profiles > 0 and batch_map is not None:
        params['omega'] = omega
        if estimable_profiles:
            params['profiles'] = profiles
            
    fixed_profiles = profiles if not estimable_profiles else None
    M = data.shape[1]
    min_M = float(np.min(batch_map.sum(axis=0))) if batch_map is not None else float(M)
    
    fun = partial(negloglik, num_nodes=num_nodes, high_precision=high_precision, min_M=min_M)
    fun_aux = None
    t = float('inf')
    
    for i in range(3):
        for opt in ['SLSQP', 'TNC', 'L-BFGS-B']:
            print(i+1, opt)
            params_c, t_c, fun_aux = optimize(fun=fun, data=data, processed_data=processed_data,
                                              ftol=ftol, params=params, num_iters=10000, return_funs=True,
                                              fun_aux=fun_aux,
                                              optimizer=opt, batch_map=batch_map, fixed_profiles=fixed_profiles, M=M)
            if t_c < t:
                params = params_c
                t = t_c
                print(i+1, t_c, '[improved]')
            else:
                print(i+1, t_c)
                
    U = compute_U(params.get('profiles', fixed_profiles), batch_map, M)
    
    w_f = params['w']
    if len(w_f) == num_mixtures - 1:
        w_f = jnp.append(w_f, 1.0 - w_f.sum())
        
    om_f = params.get('omega', jnp.ones(1))
    if len(om_f) == num_profiles:
        om_f = jnp.append(om_f, 1.0 - om_f.sum())
        
    T_joint, C, R, loglik = compute_responsibilities(processed_data=processed_data,
                                                     num_nodes=num_nodes, min_M=min_M,
                                                     r=params['r'], p=params['p'], theta=params['theta'], pi=params['pi'], w=w_f, omega=om_f, U=U)
                                               
    if len(params['w']) == num_mixtures - 1:
        params['w'] = w_f
    if 'omega' in params and len(params['omega']) == num_profiles:
        params['omega'] = om_f
        
    return params, R, loglik, num_nodes


def calc_tau_means(data, params, responsibilities, 
                   batch_size=32, num_nodes=100,
                   zero_inflation: bool = False, U=None, min_M=1.0, batch_correction: bool = False):
    if not zero_inflation:
        responsibilities = responsibilities / responsibilities.sum(axis=(-1, -2), keepdims=True)
    theta, r, p = params['theta'], params['r'], params['p']
    B, M, N = data.shape
    if U is None:
        U = jnp.ones((1, M))
        
    processed = prepare_data(data)
    momfun = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, 
                      r=r, p=p, theta=theta, U=U, high_precision=True, min_M=min_M)
    
    moments = jnp.identity(M)
    moments = jnp.vstack((jnp.zeros((1, M)), moments))
    logmoments = jax.lax.map(lambda m: momfun(moments=m), moments)
    moments = jnp.exp(logmoments[1:] - logmoments[:1])
    
    if batch_correction and U is not None:
        moments = moments / U.T[:, None, None, None, :]
        
    return (responsibilities * moments).sum(axis=(-1, -2)).transpose(1, 0, 2)

def estimate_lambda(data, params, responsibilities, batch_size: int = 32,
                    num_nodes: int = 100, U=None, min_M=1.0) -> tuple[np.ndarray, np.ndarray]:
    B, M, N = data.shape
    theta, r, p = params['theta'], params['r'], params['p']
    if U is None:
        U = jnp.ones((1, M))
        
    processed = prepare_data(data)
    momfun_ = partial(marginal_loglik, num_nodes=num_nodes, batch_size=batch_size,
                      processed_data=processed, eps=1e-12, 
                      r=r, p=p, theta=theta, U=U, high_precision=True, min_M=min_M)
    momfun = lambda x: momfun_(lambda_moment=x)
    logmarginal = momfun(0)
    first_moment = jnp.exp(momfun(1) - logmarginal)
    second_moment = jnp.exp(momfun(2) - logmarginal)
    mean = (first_moment * responsibilities).sum(axis=(-1, -2))
    var = (responsibilities * (second_moment - first_moment ** 2)).sum(axis=(-1, -2))
    return mean, var

@partial(jax.jit, static_argnames=('sample_ind', 'normalized', 'batch_correction'))
def _estimate_log_posterior_jit(tau, sample_ind, X, r, p, w, pi, omega, U,
                                phi, logweights, taus_quad, normalized, batch_correction):
    B, M, N = X.shape
    K = r.shape[-1]
    V = U.shape[0]
    eps = 1e-12
    
    beta = 1.0
    lambda_div = M + 1 / beta
    log_phi_hat = jnp.log(phi) - jnp.log(lambda_div)
    
    # Strictly enforce simplex normalization
    if len(w) == K - 1:
        w = jnp.append(w, jnp.clip(1.0 - jnp.sum(w), eps, 1.0 - eps))
    else:
        w = jnp.clip(w, eps, 1.0 - eps)
    w = w / jnp.sum(w)
        
    if len(omega) == V - 1:
        omega = jnp.append(omega, jnp.clip(1.0 - jnp.sum(omega), eps, 1.0 - eps))
    else:
        omega = jnp.clip(omega, eps, 1.0 - eps)
    omega = omega / jnp.sum(omega)
        
    num_nodes = phi.shape[0]
    log_A_sum = jnp.zeros((B, N, K, V, num_nodes))
    log_phi_hat_v = log_phi_hat[None, None, :] + jnp.log(U)[:, :, None]
    
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
        
        log_phi_poi_m = -gammaln(taus_quad + 1)[None, :, None] + taus_quad[None, :, None] * log_phi_hat_v[:, m, None, :]
        
        # --- FIXED AXIS SCRAMBLING ---
        log_phi_poi_flat = log_phi_poi_m.transpose(1, 0, 2).reshape(len(taus_quad), V * num_nodes)
        
        log_A_m_flat = log_einsum_exp(logpmf_flat, log_phi_poi_flat)
        log_A_m = log_A_m_flat.reshape(B, N, K, V, num_nodes)
        
        log_A_sum = log_A_sum + log_A_m

    tau_jnp = jnp.asarray(tau)
    T_user = len(tau_jnp)

    if batch_correction:
        r_slice = r[:, sample_ind, :, None, None]            # (B, K, 1, 1)
        U_slice = U[:, sample_ind][None, None, :, None]      # (1, 1, V, 1)
        tau_slice = tau_jnp[None, None, None, :]             # (1, 1, 1, T)
        rtau_s = r_slice * U_slice * tau_slice               # (B, K, V, T)
        
        log_phi_hat_user = log_phi_hat[None, None, :]        # (1, 1, num_nodes)
        log_phi_poi_user = -gammaln(tau_jnp + 1)[None, :, None] + tau_jnp[None, :, None] * log_phi_hat_user
    else:
        r_slice = r[:, sample_ind, :, None, None]            # (B, K, 1, 1)
        tau_slice = tau_jnp[None, None, None, :]             # (1, 1, 1, T)
        rtau_s = r_slice * tau_slice                         # (B, K, 1, T)
        
        log_phi_hat_user = log_phi_hat_v[:, sample_ind, None, :] # (V, 1, num_nodes)
        log_phi_poi_user = -gammaln(tau_jnp + 1)[None, :, None] + tau_jnp[None, :, None] * log_phi_hat_user

    is_rtau_zero_s = rtau_s == 0
    rtau_s = jnp.where(is_rtau_zero_s, 1.0, rtau_s)
    gammaln_rtau_s = gammaln(rtau_s)

    logp_s = jnp.log(p[:, sample_ind, :])[:, :, None, None] * rtau_s
    log1p_s = jnp.log1p(-p[:, sample_ind, :])[:, :, None, None]

    rtau_s_exp = rtau_s[:, None, :, :, :]
    gammaln_rtau_s_exp = gammaln_rtau_s[:, None, :, :, :]
    is_rtau_zero_s_exp = is_rtau_zero_s[:, None, :, :, :]
    logp_s_exp = logp_s[:, None, :, :, :]
    log1p_s_exp = log1p_s[:, None, :, :, :]

    xs_s = X[:, sample_ind, :][:, :, None, None, None]
    logpmf_s = safe_nb_logpmf_precomp(xs_s, logp_s_exp, log1p_s_exp, rtau_s_exp, gammaln_rtau_s_exp, is_rtau_zero_s_exp)

    log_A_sum_w = log_A_sum + logweights[None, None, None, None, :]
    log_B_mat = log_phi_poi_user.transpose(0, 2, 1)

    log_marginal_lambda = jnp.zeros((B, N, K, V, T_user))
    for v in range(V):
        log_A_v_flat = log_A_sum_w[:, :, :, v, :].reshape(B * N, K, num_nodes)
        b_idx = 0 if log_B_mat.shape[0] == 1 else v
        log_B_v = log_B_mat[b_idx]
        log_res = log_einsum_exp(log_A_v_flat, log_B_v).reshape(B, N, K, T_user)
        log_marginal_lambda = log_marginal_lambda.at[:, :, :, v, :].set(log_res)

    log_D = logpmf_s + log_marginal_lambda
    
    log_D_w = log_D + jnp.log(w)[None, None, :, None, None] + jnp.log(omega)[None, None, None, :, None]
    log_E = logsumexp(log_D_w, axis=(2, 3))
    log_E = log_E + jnp.log1p(-jnp.clip(pi, eps, 1.0 - eps))[:, None, None]

    I_nb = jnp.all(X == 0, axis=1)
    zi_val = jnp.log(jnp.clip(pi, eps, 1.0 - eps))[:, None] + jnp.where(I_nb, 0.0, -jnp.inf)
    
    zero_mask = (tau_jnp == 0)
    zi_expanded = jnp.where(zero_mask[None, None, :], zi_val[:, :, None], -jnp.inf)
    
    log_posterior = jnp.logaddexp(log_E, zi_expanded)

    if normalized:
        log_posterior = log_posterior - logsumexp(log_posterior, axis=-1, keepdims=True)
        
    return log_posterior.transpose(1, 0, 2)


def estimate_log_posterior(tau: jnp.ndarray, sample_ind: int, result: LatentCountsResult, X: np.ndarray, normalized: bool = False,
                           right=None, batch_map=None, batch_correction: bool = False) -> np.ndarray:
    params = result.params
    num_nodes = result.num_nodes
    
    r = jnp.asarray(params['r'])
    p = jnp.asarray(params['p'])
    theta = jnp.asarray(params['theta'])
    w = jnp.asarray(params['w'])

    if 'pi' in params:
        pi = jnp.asarray(params['pi'])
    else:
        pi = jnp.zeros(X.shape[0])
    pi = jnp.clip(pi, 1e-12, 1.0 - 1e-12)
    
    if 'omega' in params:
        omega = jnp.asarray(params['omega'])
    else:
        omega = jnp.ones(1)
        
    X_jnp = jnp.asarray(X)
    B, M, N = X_jnp.shape
    
    if batch_map is not None and 'profiles' in params:
        U = get_full_profiles(jnp.asarray(params['profiles']), batch_map, M) @ batch_map.T
    else:
        U = jnp.ones((1, M))
        
    min_M = float(np.min(batch_map.sum(axis=0))) if batch_map is not None else float(M)

    alpha = float(theta[0])
    phi, logweights = compute_nodes_and_logweights(num_nodes=num_nodes, param=alpha - 1, rule=Rules.GenLaguerre)
    
    eps = 1e-9
    if right is None:
        max_lambda = (num_nodes * 4 + 2 * MAX_POI - 2) / min_M
        p_np, r_np = np.asarray(p), np.asarray(r)
        C = max_lambda * (p_np ** r_np) 
    
        X_max = np.max(X, axis=2)[:, :, None]  
        C_flat = C.flatten()
        r_flat = r_np.flatten()
        X_flat = np.broadcast_to(X_max, C.shape).flatten()
        
        tau_val = C_flat + X_flat / r_flat + 1.0  
        for _ in range(10):  
            term1 = 1.0 + X_flat / (r_flat * tau_val + 1e-12)
            F = tau_val - C_flat * (term1 ** r_flat)
            F_prime = 1.0 + C_flat * X_flat / (tau_val**2 + 1e-12) * (term1 ** (r_flat - 1.0))
            tau_val = np.maximum(tau_val - F / F_prime, C_flat) 
            
        max_tau_star = float(np.max(tau_val))
        safe_mu = max_tau_star * 1.05 + 1.0
        right = int(scp_poisson.isf(eps, safe_mu))
    
    taus_quad = jnp.arange(0, right + 1)
    
    log_posterior = _estimate_log_posterior_jit(
        tau=jnp.asarray(tau),
        sample_ind=sample_ind,
        X=X_jnp,
        r=r, p=p, w=w, pi=pi, omega=omega, U=U,
        phi=phi, logweights=logweights, taus_quad=taus_quad,
        normalized=normalized, batch_correction=batch_correction
    )
    
    return np.asarray(log_posterior)


def infer_latent_counts(data: np.ndarray, num_mixture_components: int,
                        em_max_iter=40, 
                        em_warmup_iters=1,
                        mle_warmup: bool = True,
                        ftol=1e-6, use_cuda: bool=True,
                        num_nodes: int = 30, max_nodes: int = 120,
                        max_nodes_true: int = None, 
                        high_precision: bool = True,
                        multi_precision: bool = False,
                        prev_filename: str = None, batch_map=None, num_profiles=0, custom_profiles=None, estimable_profiles=True,
                        batch_correction: bool = True) -> LatentCountsResult:
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
                                                   ftol=ftol, num_nodes=num_nodes, high_precision=prec,
                                                   batch_map=batch_map, num_profiles=num_profiles, custom_profiles=custom_profiles, estimable_profiles=estimable_profiles,
                                                   **params)
        if em_max_iter:
            params, R, loglik, num_nodes = fit_em(data, num_mixtures=num_mixture_components, 
                                                   num_iters=em_max_iter, ftol=ftol, 
                                                   num_warmup=em_warmup_iters *(not mle_warmup),
                                                   num_nodes=num_nodes, max_nodes=max_nodes, max_nodes_true=max_nodes_true,
                                                   high_precision=prec, batch_map=batch_map, num_profiles=num_profiles, custom_profiles=custom_profiles, estimable_profiles=estimable_profiles,
                                                   **params)
        if len(precisions) > 1 and prec==False:    
            print('-' * 10)
            print('Running for high precision.')
            print('-' * 10)
        
    M = data.shape[1]
    min_M = float(np.min(batch_map.sum(axis=0))) if batch_map is not None else float(M)
    
    if batch_map is not None:
        raw_prof = params.get('profiles')
        if raw_prof is None and custom_profiles is not None:
            raw_prof = custom_profiles
        full_profiles = get_full_profiles(raw_prof, batch_map, M)
        U_final = full_profiles @ batch_map.T
    else:
        U_final = np.ones((1, M))

    if max_nodes_true:
        num_nodes = max_nodes_true
        
    taus = calc_tau_means(data, params, R, num_nodes=num_nodes, U=jnp.asarray(U_final), min_M=min_M, batch_correction=batch_correction)
    taus = np.asarray(taus)
    lambdas, lambdas_var = estimate_lambda(data, params, R, U=jnp.asarray(U_final), min_M=min_M)
    R = np.asarray(R)
    loglik = float(loglik)
    params = {n: np.asarray(v) for n, v in params.items()}
    return LatentCountsResult(params=params, responsibilities=R, counts=taus, loglik=loglik,
                              num_nodes=num_nodes, prospenity=lambdas, prospenity_var=lambdas_var)