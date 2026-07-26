import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from functools import partial
from dataclasses import dataclass
from copy import deepcopy
import warnings
import numpy as np
from compound_start import fit_nb

_prepared_data = tuple[list[list[tuple[jnp.ndarray, jnp.ndarray]]], jnp.ndarray]



def poisson_logpmf(x, lam):
    return jnp.log(lam) * x - lam - gammaln(x + 1)

def poisson_forward(x, lam):
    return jnp.log(lam) - jnp.log(x)

def poisson_backward(x, lam):
    return -jnp.log(lam) + jnp.log(x + 1)

def poisson_pr_logmoment(lam, r, p):
    return -lam * (1. - p ** r)


def nb_logpmf(x, r, p):
    return gammaln(x + r) - gammaln(r) - gammaln(x + 1) + jnp.log1p(-p) * x + jnp.log(p) * r

def nb_forward(x, r, p):
    return jnp.log1p(-p) + jnp.log(x + r - 1) - jnp.log(x)

def nb_backward(x, r, p):
    return -jnp.log1p(-p) - jnp.log(x + r) + jnp.log(x + 1)

def nb_pr_logmoment(r0, p0, r, p):
    return r0 * (jnp.log(p0) - jnp.log1p(-p ** r * (1 - p0)))

def get_model_funs(lam: float):
    g = lambda x: poisson_logpmf(x, lam)
    g_forward = lambda x: poisson_forward(x, lam)
    g_backward = lambda x: poisson_backward(x, lam)
    g_pr_logmoment = lambda r, p: poisson_pr_logmoment(lam, r, p)
    g_mean = lam
    return g, g_forward, g_backward, g_pr_logmoment, g_mean


def compute_logR(r: float, p: float, lam: float, max_sz: int, compute_gamma_term=True) -> tuple[int, int, jnp.ndarray]:

    g, g_forward, g_backward, g_pr_logmoment, g_mean = get_model_funs(lam)
    
    def body_forward(carry):
        tau, prev_logg, prev_logp, result = carry
        tau = tau + 1
        logp = prev_logp + logp_r
        logg = prev_logg + g_forward(tau)
        if compute_gamma_term:
            gamma_term = gammaln(r * tau)
        else:
            gamma_term = 0.0
        result = result.at[tau].set(logp + logg - gamma_term)
        return tau, logg, logp, result
        
    def cond_forward(carry):
        tau, prev_logg, prev_logp, result = carry
        return jnp.logical_and(prev_logg > jnp.log(1e-10), tau < max_sz)
    
    def body_backward(carry):
        tau, prev_logg, prev_logp, result = carry
        tau = tau - 1
        logp = prev_logp - logp_r
        logg = prev_logg + g_backward(tau)
        if compute_gamma_term:
            gamma_term = gammaln(r * tau)
        else:
            gamma_term = 0.0
        result = result.at[tau].set(logp + logg - gamma_term)
        return tau, logg, logp, result
    def cond_backward(carry):
        tau, prev_logg, prev_logp, result = carry
        return jnp.logical_and(prev_logg > jnp.log(1e-10), tau > 0)
    c = jnp.asarray(g_mean).astype(int)
    logp_r = jnp.log(p) * r
    result = jnp.zeros(max_sz, dtype=float)
    base_logg = g(c)
    base_logp = logp_r * c
    
    result = result.at[c].set(base_logg + base_logp - gammaln(r * c))
    b, _, _, result = jax.lax.while_loop(cond_forward, body_forward, (c, base_logg, base_logp, result))
    a, _, _, result = jax.lax.while_loop(cond_backward, body_backward, (c, base_logg, base_logp, result))
    return a, b, result


def compound_logpmf(x: float, r: float, p: float, theta,  
                  logR=None, a=None, b=None, batch_size=32, max_sz=10000):
    g, g_forward, g_backward, g_logmoment, g_mean = get_model_funs(theta)
    log_pmf_x0 = g_logmoment(r, p)
    base = jnp.log1p(-p) * x - gammaln(x + 1)
    if logR is None:
        a, b, logR = compute_logR(r, p, g_mean, g, g_forward, g_backward, max_sz, compute_gamma_term=True)
    
    a = jnp.maximum(a, 1)
    num_elements = (b - a + 1)
    num_steps = jnp.ceil(num_elements / batch_size).astype(int)
    
    def summation():
        def body(i, logpmf, mask=False):
            left = a + i * batch_size
            taus = jnp.arange(0, batch_size) + left
            log_R = logR[taus]
            gammas = gammaln(x + r * taus)
            
            log_terms_batch = base + log_R + gammas
            if mask:
                mask = taus <= b
                log_terms_batch =  jnp.where(mask, log_terms_batch, -jnp.inf)
            return jnp.logaddexp(logpmf, logsumexp(log_terms_batch))

        logpmf = -jnp.inf
        logpmf = jax.lax.fori_loop(0, num_steps - 1, body, logpmf)
        logpmf = body(num_steps - 1, logpmf, mask=True)
        return logpmf
    
    def whilenation():
        LOG_EPSILON = jnp.log(1e-10) * 2.3
        start_idx = jnp.clip(jnp.round(g_mean).astype(int), a, b)
        initial_log_term = base + logR[start_idx] + gammaln(x + r * start_idx)
        initial_logpmf = initial_log_term

        def forward_cond(carry):
            i, logpmf, max_log_term_in_batch = carry
            next_batch_start = start_idx + (i - 1) * batch_size + 1
            boundary_cond = next_batch_start <= b

            convergence_cond = max_log_term_in_batch > (logpmf + LOG_EPSILON)
            return jnp.logical_and(boundary_cond, convergence_cond)

        def forward_body(carry, mask=True):
            i, logpmf, _ = carry
            left = start_idx + (i - 1) * batch_size + 1
            taus = jnp.arange(batch_size) + left
            
            log_R_batch = logR[taus]
            gammas = gammaln(x + r * taus)
            log_terms_batch = base + log_R_batch + gammas
            
            if mask:
                mask = taus <= b
                log_terms_batch = jnp.where(mask, log_terms_batch, -jnp.inf)
            
            new_logpmf = jnp.logaddexp(logpmf, logsumexp(log_terms_batch))
            new_max_log_term = jnp.max(log_terms_batch)
            
            return i + 1, new_logpmf, new_max_log_term

        init_carry_fwd = (1, initial_logpmf, initial_log_term)
        _, logpmf_after_fwd, _ = jax.lax.while_loop(forward_cond, forward_body, init_carry_fwd)

        def backward_cond(carry):
            i, logpmf, max_log_term_in_batch = carry
            next_batch_end = start_idx - (i - 1) * batch_size - 1
            boundary_cond = next_batch_end >= a
            convergence_cond = max_log_term_in_batch > (logpmf + LOG_EPSILON)
            return jnp.logical_and(boundary_cond, convergence_cond)

        def backward_body(carry, mask=True):
            i, logpmf, _ = carry
            right = start_idx - (i - 1) * batch_size - 1
            taus = right - jnp.arange(batch_size)
            
            log_R_batch = logR[taus]
            gammas = gammaln(x + r * taus)
            log_terms_batch = base + log_R_batch + gammas
            
            if mask:
                mask = taus >= a
                log_terms_batch = jnp.where(mask, log_terms_batch, -jnp.inf)
            
            new_logpmf = jnp.logaddexp(logpmf, logsumexp(log_terms_batch))
            new_max_log_term = jnp.max(log_terms_batch)
            
            return i + 1, new_logpmf, new_max_log_term

        init_carry_bwd = (1, logpmf_after_fwd, initial_log_term)
        _, final_logpmf, _ = jax.lax.while_loop(backward_cond, backward_body, init_carry_bwd)
        return final_logpmf
    
    # I have tried summation and whilenation, no difference in, whilenation is 2 times faster
    # Hence, whilenation is a correct heuristic
    # logpmf = jax.lax.cond(num_elements > 0, summation, lambda: -jnp.inf)
    logpmf = jax.lax.cond(num_elements > 0, whilenation, lambda: -jnp.inf) 
    return jnp.where(x > 0, logpmf, log_pmf_x0)

        

def prepare_data(data: jnp.ndarray) -> _prepared_data:
    res = list()
    zeros = np.all(data == 0, axis=1)
    for bin_matrix in data:
        lt = list()
        for sample_matrix in bin_matrix:
            unique_vals, inverse = np.unique(sample_matrix, return_inverse=True)
            lt.append((jnp.asarray(unique_vals).astype(int),
                        jnp.asarray(inverse).astype(int)))
        res.append(lt)
    return res, jnp.array(zeros)

@partial(jax.jit, static_argnames=('max_sz'))
def logpmf_vmapped(processed_data: _prepared_data,
                   r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray,
                   max_sz: int, ):
    processed_data = processed_data[0]
    N = len(processed_data[0][0][-1])
    M = len(processed_data[0])
    B = len(processed_data)
    K = r.shape[-1]
    W = len(theta)
    logP = jnp.zeros((B, M, N, K, W), dtype=float)
    logR_base = partial(compute_logR, max_sz=max_sz)
    logpmf_fun_base = partial(compound_logpmf, max_sz=max_sz)
    logpmf_fun_base = jax.vmap(logpmf_fun_base, in_axes=(0, None, None, None, None, None, None))
    
    def block_fun(xs, r, p, theta):
        a, b, logR = logR_base(r, p, theta)
        logpmf = logpmf_fun_base(xs, r, p, theta, logR, a, b)
        return logpmf
    logpmf_fun = jax.vmap(block_fun, in_axes=(None, None, None, 0))
    logpmf_fun = jax.vmap(logpmf_fun, in_axes=(None, 0, 0, None))
    for b in range(B):
        for m in range(M):
            x, inv_inds = processed_data[b][m]
            rs = r[b, m]
            ps = p[b, m]
            logpmfs = logpmf_fun(x, rs, ps, theta).transpose(-1, 0, 1)[inv_inds]
            logP = logP.at[b, m].set(logpmfs)
    return logP

@partial(jax.jit, static_argnames=('eps', ))
def update_weights(T: jnp.ndarray, C: jnp.ndarray, O: jnp.ndarray,  eps=1e-11) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        - new_omega (jnp.ndarray): Updated sub-component weights. Shape: (B, W).
    """

    N_from_T = T.shape[0]
    N_from_C = C.shape[1]
    assert N_from_T == N_from_C, f"Inconsistent number of objects: {N_from_T} from T vs {N_from_C} from C"
    N = N_from_T


    weights_numerator = jnp.sum(T, axis=0)
    new_weights = weights_numerator / N


    pi_numerator = jnp.sum(C, axis=1)
    new_pi = pi_numerator / N


    new_weights = new_weights / jnp.sum(new_weights)
    new_weights = jnp.clip(new_weights, 0.0, 1.0)

    new_pi = jnp.clip(new_pi, 0.0, 1.0)
    
    omega_numerator = O
    omega_denominator = jnp.sum(O, axis=1, keepdims=True)
    # Avoid division by zero if a bin 'b' is always zero-inflated
    safe_denominator = jnp.where(omega_denominator == 0, 1.0, omega_denominator)
    new_omega = omega_numerator / safe_denominator
    
    new_omega = new_omega / jnp.sum(new_omega, axis=1, keepdims=True)
    new_omega = jnp.clip(new_omega, 0.0, 1.0)

    return new_weights, new_pi, new_omega


def compute_responsibilities(processed_data: _prepared_data,
                                 r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray,
                                 pi: jnp.ndarray, w: jnp.ndarray, omega: jnp.ndarray,
                                 max_sz: int,
                                 eps=1e-11) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the responsibilities for the E-step of the hierarchical zero-inflated
    double mixture model.

    Returns:
        A tuple containing:
        - T (jnp.ndarray): Posterior P(Z_n=k | X_n). Shape (N, K).
        - C (jnp.ndarray): Posterior P(Phi_nb=1 | X_n). Shape (B, N).
        - O (jnp.ndarray): Posterior expectation for sub-component choice, summed over n and k.
                           O_bw = sum_{n,k} R_nbkw. Shape (B, W).
        - R (jnp.ndarray): Joint posterior P(Z_n=k, Phi_nb=0, S_nb=w | X_n). Shape (B, N, K, W).
        - avg_log_likelihood (float): The average log-likelihood of the data.
    """
    # B: bins, M: replicates, N: sequences, K: components, W: sub-components
    logP = logpmf_vmapped(processed_data, r=r, p=p, theta=theta, max_sz=max_sz)
    B, M, N, K, W = logP.shape

    I_nb = processed_data[1]

    # Prepare log-probabilities of weights
    log_pi = jnp.log(jnp.clip(pi, eps))
    log_one_minus_pi = jnp.log1p(-pi)#jnp.log(jnp.clip(1.0 - pi, eps))
    w_clipped = jnp.append(w, 1.0 - w.sum())
    w_clipped = jnp.clip(w_clipped / w_clipped.sum(), 0.0, 1.0 )
    log_weights = jnp.log(w_clipped)
    omega_clipped = jnp.append(omega, 1.0 - omega.sum(axis=-1, keepdims=True), axis=-1)
    omega_clipped = omega_clipped / omega_clipped.sum(axis=-1, keepdims=True)
    omega_clipped = jnp.clip(omega_clipped, 0.0, 1.0)
    log_omega = jnp.log(omega_clipped)

    # 1. Sum over replicates M. log_L_nbkw shape: (B, N, K, W)
    log_L_nbkw = jnp.sum(logP, axis=1)

    # 2. Marginalize out sub-component `w`. log_L_nbk_marginalized shape: (B, N, K)
    log_omega_b = log_omega[:, None, None, :]
    log_L_nbk_marginalized = logsumexp(log_omega_b + log_L_nbkw, axis=3)

    # 3. Calculate zero-inflated term. log_term_per_indiv_and_comp shape: (B, N, K)
    log_pi_b = log_pi[:, None, None]
    log_one_minus_pi_b = log_one_minus_pi[:, None, None]
    log_I_nb = jnp.log(I_nb[:, :, None])
    log_term1 = log_pi_b + log_I_nb
    log_term2 = log_one_minus_pi_b + log_L_nbk_marginalized
    log_term_per_indiv_and_comp = jnp.logaddexp(log_term1, log_term2)

    # 4. Sum over bins `b`. log_P_Xn_given_Znk shape: (N, K)
    log_P_Xn_given_Znk = jnp.sum(log_term_per_indiv_and_comp, axis=0)

    # 5. Calculate marginal log-likelihood. log_P_Xn shape: (N,)
    log_joint_likelihoods = log_weights + log_P_Xn_given_Znk
    log_P_Xn = logsumexp(log_joint_likelihoods, axis=1)

    # 6. Compute T_nk = P(Z_n=k | X_n)
    log_T = log_joint_likelihoods - log_P_Xn[:, None]
    T = jnp.exp(log_T)

    # 7. Compute R_nbkw = P(Z_n=k, Phi_nb=0, S_nb=w | X_n)
    log_T_b = log_T[None, :, :, None]
    log_one_minus_pi_b_kw = log_one_minus_pi[:, None, None, None]
    log_omega_b_kw = log_omega[:, None, None, :]
    log_term_b_kw = log_term_per_indiv_and_comp[:, :, :, None]
    log_R = log_T_b + log_one_minus_pi_b_kw + log_omega_b_kw + log_L_nbkw - log_term_b_kw
    R = jnp.exp(log_R)

    # 8. Compute C_nb = P(Phi_nb=1 | X_n)
    C = 1.0 - jnp.sum(R, axis=(2, 3))
    C = jnp.clip(C, 0.0, 1.0)
    # 9. NEW: Compute O_bw = sum_{n,k} R_nbkw for updating omega
    # R has shape (B, N, K, W). Sum over N (axis 1) and K (axis 2).
    O = jnp.sum(R, axis=(1, 2))
    # O = jnp.clip(O, 0.0, 1.0)

    return T, C, O, R, log_P_Xn.sum() / (B * M * N)



def get_starting_values(data: jnp.ndarray, num_mixtures: int, num_poissons: int, theta, r, p, w, pi, omega):
    B, M, N = data.shape
    theta_flag = False
    if theta is None:
        theta_flag = True
        theta = jnp.arange(1, num_poissons + 1, dtype=float) ** 2 * 5
        omega = jnp.arange(1, num_poissons + 1, dtype=float) ** 2 
        omega = omega[::-1] / omega.sum()
        mean = (theta * omega).sum()
        omega = jnp.repeat(omega.reshape(1,-1), B, axis=0)
    if omega.shape[-1] == num_poissons:
        omega = omega.at[..., :-1].get()
    if r is None:
        r = jnp.arange(1, num_mixtures + 1, dtype=float).reshape(1, 1, -1)
        r = jnp.repeat(r, B, axis=0)
        r = jnp.repeat(r, M, axis=1)
    elif theta_flag:
        r = r / mean
        
    if p is None:
        p = jnp.linspace(1e-2, 0.1, num=num_mixtures)[::-1].reshape(1, 1, -1)
        p = jnp.repeat(p, B, axis=0)
        p = jnp.repeat(p, M, axis=1) 
    if w is None:
        w = jnp.arange(1, num_mixtures + 1)
        w = w / w.sum()
        w = w.at[:-1].get()
    elif len(w) == num_mixtures:
        w = jnp.asarray(w).at[:-1].get()
    if pi is None:
        pi = jnp.all(data == 0, axis=1).mean(axis=-1) / 2
    return theta, r, p, w, pi, omega
    

@partial(jax.jit, static_argnames=('max_sz',))
def Q_dist(r: jnp.ndarray, p: jnp.ndarray, theta: jnp.ndarray, R: jnp.ndarray,
           processed_data: _prepared_data, max_sz: int):
    logP = logpmf_vmapped(processed_data, 
                          r=r, p=p, theta=theta, max_sz=max_sz).sum(axis=1)
    
    return -(logP * R).mean()
    

def param_slicer(x: jnp.ndarray, names: list[str], shapes: list[tuple[int]], mult: int = 0, **kwargs):
    params = dict()
    n = 0
    x = jnp.asarray(x)
    for name, shape in zip(names, shapes):
        m = n + np.prod(np.asarray(shape))
        params[name] = x.at[n:m].get().reshape(shape)
        n = m
    if mult > 0:
        kwargs['r'] = kwargs['r'] * x.at[-mult:].get().squeeze()
    return params, kwargs

@partial(jax.jit, static_argnames=('f', 'names', 'shapes', 'mult'))
def param_slicer_wrapper(x: jnp.ndarray, f, names: list[str], shapes: list[tuple[int]],
                         mult: int = 0,
                         **kwargs):
    params, kwargs = param_slicer(x, names, shapes, mult, **kwargs)
    return f(**params, **kwargs)
    
    
def get_bounds(names, shapes):
    bounds = list()
    base = {'r': [[1e-5, None]], 'p': [[1e-7, 0.99]],
            'w': [[0.0, 1.0]], 'pi': [[0.0, 1.0]],
            'theta': [[0.0, 5000]],
            'omega': [[0.0, 1.0]]}
    for name, shape in zip(names, shapes):
        t = np.prod(np.asarray(shape))
        b = base[name]
        bounds.extend(b * t)
    return bounds


def coordinate_descent(fun, data, processed_data: _prepared_data, params: dict, num_iters: int,
                       R=None, ftol: float = 1e-6,  max_iter=1000,
                       return_funs: bool = False, funs=None, verbose: bool = False):
    params = deepcopy(params)
    if funs is not None:
        rp_funs, (t_fun, t_jac, t_names, t_shapes, t_bounds, t_slicer) = funs
        t_names = tuple(filter(lambda x: x not in ('r', 'p'), params.keys()))
        t_shapes = tuple([params[name].shape for name in t_names])
        t_bounds = get_bounds(t_names, t_shapes)
        num_mults = params['r'].shape[-1]
        t_bounds.extend([(0.1, 10)] * num_mults)
        t_slicer = partial(param_slicer, names=t_names, shapes=t_shapes, mult=num_mults)
    else:
        rp_funs = list()
        data = data == 0
        for b, bin_list in enumerate(processed_data[0]):
            for m, sample in enumerate(bin_list):
                p = {'r': params['r'][b:b+1, m:m+1], 'p': params['p'][b:b+1, m:m+1] }
                names = tuple(p.keys())
                shapes = tuple([p[name].shape for name in names])
                tpr = [[processed_data[0][b][m]]], processed_data[1][b:b+1]
                f_rp = partial(param_slicer_wrapper, f=fun, names=names, shapes=shapes,
                               processed_data=tpr)
                jac_rp = jax.jit(jax.jacfwd(f_rp, argnums=0))
                slicer = partial(param_slicer, names=names, shapes=shapes)
                rp_funs.append((f_rp, jac_rp, slicer, get_bounds(names, shapes), names, (b, m)))
        
        t_names = tuple(filter(lambda x: x not in ('r', 'p'), params.keys()))
        t_shapes = tuple([params[name].shape for name in t_names])
        t_bounds = get_bounds(t_names, t_shapes)
        num_mults = params['r'].shape[-1]
        t_bounds.extend([(0.1, 10)] * num_mults)
        t_slicer = partial(param_slicer, names=t_names, shapes=t_shapes, mult=num_mults)
        t_fun = partial(param_slicer_wrapper, f=fun, names=t_names, shapes=t_shapes,
                          mult=num_mults, processed_data=processed_data)
        t_jac = jax.jit(jax.jacfwd(t_fun, argnums=0))
    
    prev_fun = float('inf')
    for n in range(num_iters):
        try:
            if verbose:
                print(f'\tCD iteration: {n+1}/{num_iters}')
                
            x0 = list()
            for name in t_names:
                x0.extend(params[name].flatten())
            x0 = jnp.array(x0)
            x0 = jnp.append(x0, jnp.ones(num_mults, dtype=float))
            fixed_params = {'r': params['r'], 'p': params['p']}
            if R is None:
                fun = partial(t_fun, **fixed_params)
                jac = partial(t_jac, **fixed_params)
            else:
                fun = partial(t_fun, R=R, **fixed_params)
                jac = partial(t_jac, R=R, **fixed_params)
            with warnings.catch_warnings(action="ignore"):
                res = minimize(fun, x0, jac=jac, method='SLSQP', 
                               options={'maxiter': max_iter, 'ftol': 1e-10},
                               bounds=t_bounds)
            if not res.success:
                print(res)
            # print(res)
            cur_fun = -res.fun
            p, kw = t_slicer(res.x, **fixed_params)
            for p, v in p.items():
                params[p] = v
            params['r'] = kw['r']
            for funr, jacr, slicer, bounds, names, (b, m) in rp_funs:
                x0 = list()    # r, p strategies
                for name in names:
                    x0.extend(params[name][b:b+1, m:m+1].flatten())
                x0 = jnp.asarray(x0)
                fixed_params = {'theta': params['theta']}
                if 'pi' in params:
                    fixed_params['pi'] = params['pi'][b:b+1]
                if 'w' in params:
                    fixed_params['w'] = params['w']
                if R is None:
                    fun = partial(funr, **fixed_params)
                    jac = partial(jacr, **fixed_params)
                else:
                    fun = partial(funr, R=R[b:b+1], **fixed_params)
                    jac = partial(jacr, R=R[b:b+1], **fixed_params)
                # print(fun(x0))
                with warnings.catch_warnings(action="ignore"):
                    res = minimize(fun, x0, jac=jac, method='SLSQP', bounds=bounds,
                                   options={'maxiter': max_iter, 'ftol': 1e-10})
                p, _ = slicer(res.x)
                for n in names:
                    params[n] = params[n].at[b:b+1, m:m+1].set(p[n])
            if verbose:
                print(f'Cur fun: {cur_fun:6f}\tPrev fun: {prev_fun:6f}')
            if prev_fun - res.fun < ftol:
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
        funs = rp_funs, (t_fun, t_jac, t_names, t_shapes, t_bounds, t_slicer)
        return params, res.fun, funs
    return params, res.fun
    
        

def fit_em(data: jnp.ndarray, num_mixtures: int, max_sz: int, num_poissons: int = 1,
           num_iters: int = 100, ftol: float = 1e-6, theta=None, r=None, p=None, w=None,
           pi=None, omega=None):
    from time import time
    theta, r, p, w, pi, omega = get_starting_values(data, num_mixtures, num_poissons, theta, r, p, w, pi, omega)
    processed_data = prepare_data(data)
    
    params = {'r': r, 'p': p, 'theta': theta}
        
    Q_fun = partial(Q_dist, max_sz=max_sz)
    
    calc_responsibilities = partial(compute_responsibilities, processed_data=processed_data,
                                    max_sz=max_sz)
    prev_loglik = -float('inf')
    funs = None
    for n_iter in range(num_iters):
        try:
            print(f'\tEM iteration: {n_iter+1}/{num_iters}')
            t0 = time()
            print('E-step...')
            T, C, O, R, loglik = calc_responsibilities(**params, pi=pi, w=w, omega=omega)
            if n_iter > 0:
                print(loglik, prev_loglik)
            assert loglik > prev_loglik - ftol, loglik
            if (loglik - prev_loglik) < ftol:
                print('EM converged.', loglik - prev_loglik)
                break
            prev_loglik = loglik
            print('M-step...')
            w, pi, omega = update_weights(T, C, O)
            
            omega = omega[..., :-1]
            w = w[:-1]
            print(params['theta'])
            print(w)
            print(omega)
            params, _, funs = coordinate_descent(Q_fun, data, processed_data, params=params, num_iters=1,
                                                 R=R, funs=funs, return_funs=True)
            if n_iter > 0:
                t = time() - t0
                print(f'Took {t:.3f}')
        except KeyboardInterrupt:
            print('EM algorithm stopped.')
            break
    
    T, C, O, R, loglik = calc_responsibilities(**params, pi=pi, w=w, omega=omega)
    w, pi, omega = update_weights(T, C, O)
    params['w'] = w
    params['pi'] = pi
    params['omega'] = omega
    return params, R, loglik


@partial(jax.jit, static_argnames=('max_sz', 'batch_size'))
def _calculate_posterior_tau_mean_single_component(x: int, r: float, p: float, theta: tuple,
                                                   max_sz: int, batch_size: int):
    g, g_forward, g_backward, _, g_mean = get_model_funs(theta)
    logR_res = compute_logR(r, p, g_mean, g, g_forward, g_backward, max_sz, compute_gamma_term=True)
    a, b, logR = logR_res
    a = jnp.maximum(a, 1)
    
    base = jnp.log1p(-p) * x - gammaln(x + 1)
    num_elements = (b - a + 1)
    num_steps = jnp.ceil(num_elements / batch_size).astype(int)

    def body(i, carry, mask=False):
        lse_unweighted, lse_weighted = carry
        left = a + i * batch_size
        taus = jnp.arange(0, batch_size) + left
        
        log_R_batch = logR[taus]
        gammas_batch = gammaln(x + r * taus)
        
        log_terms_batch = base + log_R_batch + gammas_batch
        log_terms_weighted_batch = jnp.log(taus) + log_terms_batch

        if mask:
            valid_mask = taus <= b
            log_terms_batch = jnp.where(valid_mask, log_terms_batch, -jnp.inf)
            log_terms_weighted_batch = jnp.where(valid_mask, log_terms_weighted_batch, -jnp.inf)
            
        lse_unweighted = jnp.logaddexp(lse_unweighted, logsumexp(log_terms_batch))
        lse_weighted = jnp.logaddexp(lse_weighted, logsumexp(log_terms_weighted_batch))
        return lse_unweighted, lse_weighted

    init_carry = (-jnp.inf, -jnp.inf)
    lse_unweighted, lse_weighted = jax.lax.fori_loop(0, num_steps - 1, body, init_carry)
    lse_unweighted, lse_weighted = body(num_steps - 1, (lse_unweighted, lse_weighted), mask=True)
    
    return jnp.where(num_elements > 0, jnp.exp(lse_weighted - lse_unweighted), 0.0)

def calc_tau_means(data, params, responsibilities, 
                   max_sz, hard_assignment=False, batch_size=32):
    theta = params['theta']
    vmap_tau_mean_f =  partial(_calculate_posterior_tau_mean_single_component, theta=theta,
                               max_sz=max_sz, batch_size=batch_size)
    vmap_tau_mean = jax.vmap(jax.vmap(jax.vmap(jax.vmap(vmap_tau_mean_f, in_axes=(None, 0, 0)), in_axes=(0, None, None)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    taus = vmap_tau_mean(data, params['r'], params['p'])
    return (responsibilities[:, np.newaxis] * taus).sum(axis=-1)
  

@dataclass(frozen=True)
class LatentCountsResult:
    params: dict
    responsibilities: np.ndarray
    counts: np.ndarray
    loglik: float

def infer_latent_counts(data: np.ndarray, num_mixture_components: int,
                        num_poisson_components: int = 1, max_iter=100, 
                        ftol=1e-7,) -> LatentCountsResult:
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platforms', 'cpu') 
    max_sz = 10000 + data.max() 
    params, _ = fit_nb(data, num_mixtures=num_mixture_components, 
                       verbose=False, max_iter=400)
    params, R, loglik = fit_em(data, num_mixtures=num_mixture_components,
                               num_poissons=num_mixture_components, max_sz=max_sz, 
                               num_iters=max_iter, ftol=ftol, **params)
    taus = calc_tau_means(data, params, R,
                          max_sz=max_sz)
    taus = np.asarray(taus)
    R = np.asarray(R)
    loglik = float(loglik)
    params = {n: np.asarray(v) for n, v in params.items()}
    return LatentCountsResult(params, R, taus, loglik)

import pandas as pd
from collections import defaultdict
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu') 
# jax.config.update("jax_log_compiles", 1)

filename = '/media/Data/Pr/MRPANorm/UTR5_sequence_counts_05_23_23.tsv'
df = pd.read_csv(filename, sep='\t',
                 index_col=0, header=[0, 1, 2])
df.index.name = 'seq'
df.columns.names = ['cell_type', 'replicate', 'bin']
df = df[['c1', 'c2']]

bins = defaultdict(list)
for i, (c, r, b) in enumerate(df.columns):
    bins[int(b) - 1].append(i)
bins = {b: np.array(inds, dtype=int) for b, inds in bins.items()}

X = list()

for _, subdf in df.groupby(level=['cell_type', 'replicate'], axis=1):
    X.append(subdf.values.T)
X = np.array(X).transpose((1, 0, 2))

max_sz = X.max() + 1000
num_mixtures = 1 + ('UTR5' in filename)


params, lls = fit_nb(X, num_mixtures=num_mixtures, verbose=False, max_iter=200)
params, R, loglik = fit_em(X, num_mixtures=num_mixtures, max_sz=max_sz, num_poissons=2, num_iters=1000,
                            ftol=1e-7, )#**params)
# params, R, loglik = fit_mle(X, num_mixtures=num_mixtures, max_sz=max_sz, num_iters=100, **params)
# 	EM iteration: 14/100
# E-step...
# -6.413714535678782 -6.413715740572989
# [725.90393307]
