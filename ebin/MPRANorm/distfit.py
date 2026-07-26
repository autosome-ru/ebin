import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.special
from .distributions import Distribution, Normal
from dataclasses import dataclass
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

def reconstruct_logp(quartiles: jnp.ndarray, dist: Distribution, params: jnp.ndarray):
    quartiles = jnp.asarray(quartiles)
    quartiles = quartiles.reshape(-1, 1)

    logcdf_values = dist.logcdf(quartiles, params)
    logp_first = logcdf_values[0]

    cdf_values = jnp.exp(logcdf_values)
    
    cdf_diffs = jnp.maximum(jnp.diff(cdf_values, axis=0), 0.0)
    logp_middle = jnp.log(cdf_diffs)

    logp_last = dist.logsf(quartiles[-1], params)

    log_probs = jnp.concatenate([
        jnp.atleast_1d(logp_first)[None, :],
        jnp.atleast_1d(logp_middle),
        jnp.atleast_1d(logp_last)[None, :]
    ], axis=0)

    return log_probs.squeeze()

def get_sequence_std_err(params_u_g, N_g, quartiles, dist, penalty):
    """
    Computes the standard error using the expected Fisher Information Matrix (FIM).
    """
    def get_log_probs(p_u):
        p_c = dist.transform_params(p_u)
        return reconstruct_logp(quartiles, dist, p_c)
        
    # Jacobian of log probabilities w.r.t unconstrained parameters: (num_bins, num_params)
    jac = jax.jacobian(get_log_probs)(params_u_g)
    
    log_probs = get_log_probs(params_u_g)
    probs = jnp.exp(log_probs)
    
    # Expected Fisher Information: N * sum_b ( p_b * grad(log p_b) * grad(log p_b)^T )
    fim_u = N_g * jnp.einsum('b,bi,bj->ij', probs, jac, jac)
    
    # Add prior/penalty information to the FIM if active
    if penalty:
        def prior_fn(p_u):
            p_c = dist.transform_params(p_u)
            return jnp.sum(dist.penalty(p_c))
        # Negative Hessian of the log prior adds directly to the FIM
        prior_info = -jax.hessian(prior_fn)(params_u_g)
        fim_u = fim_u + prior_info
        
    # Add a tiny ridge for inversion stability in extreme flat regions
    fim_u = fim_u + 1e-8 * jnp.eye(fim_u.shape[0])
    
    cov_u = jnp.linalg.inv(fim_u)
    
    # Cov_c = J * Cov_u * J^T
    J_trans = jax.jacfwd(dist.transform_params)(params_u_g)
    cov_c = J_trans @ cov_u @ J_trans.T
    
    var_c = jnp.diag(cov_c)
    return jnp.sqrt(jnp.maximum(var_c, 0.0))

@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 3, 4, 5))
def root_solver(fun, grad, extra_params, x0, xatol=1e-4, max_iter=100):
    def cond_fun(state):
        prev_x, i, x = state
        return (jnp.abs(prev_x - x) > xatol) & (i < max_iter)

    def body_fun(state):
        prev_x, i, x = state
        f = fun(x, extra_params)
        g = grad(x, extra_params)
        next_x = x - f / g
        return (x, i + 1, next_x)
    state = (x0 + 1, 0, x0)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    return state[2]

@root_solver.defjvp
def root_solver_jvp(fun, grad, x0, eps, max_iter, primals, tangents):
    
    extra_params, = primals
    t_extra_params, = tangents

    x_star = root_solver(fun, grad, extra_params, x0, eps, max_iter)
    f = lambda ep: fun(x_star, ep)
    
    J_x = grad(x_star, extra_params)
    
    _, J_params = jax.jvp(f, (extra_params,), (t_extra_params,))
    t_x_star = -J_params / J_x

    return x_star, t_x_star

def _fun(x, params_and_alpha, p: float, dist: Distribution, num_obs: jnp.ndarray = None):
    params, alpha = params_and_alpha
    if num_obs is None:
        return dist.cdf(x, params).mean() - p
    weights = num_obs + alpha
    weights = weights / jnp.sum(weights)
    cdf = dist.cdf(x, params) * weights 
    return jnp.sum(cdf) - p

def _grad(x, params_and_alpha, dist: Distribution, num_obs: jnp.ndarray = None):
    params, alpha = params_and_alpha
    if num_obs is None:
        return dist.pdf(x, params).mean()
    weights = num_obs + alpha
    weights = weights / jnp.sum(weights)
    pdf = dist.pdf(x, params)
    return jnp.sum(pdf * weights)

def estimate_quartiles(dist: Distribution, params: jnp.ndarray, n=4, x0=None,
                       num_obs: jnp.ndarray = None, alpha: float = 0.0,
                       max_iter=1000, xatol=1e-5, target_probs=None):
    if target_probs is None:
        target_probs = jnp.arange(1, n) / n
    else:
        target_probs = jnp.asarray(target_probs)
        
    n_q = len(target_probs)
    quartiles = jnp.zeros(n_q, dtype=float)
    grad = partial(_grad, dist=dist, num_obs=num_obs)
    for i in range(n_q):
        p = target_probs[i]
        if x0 is None:
            x = jax.scipy.stats.norm.ppf(p)
        else:
            x = x0[i]
        
        fun = partial(_fun, p=p, dist=dist, num_obs=num_obs)
        sol = root_solver(fun, grad, (params, alpha), x, xatol=xatol, max_iter=max_iter)
        quartiles = quartiles.at[i].set(sol)
    return quartiles

def loglik_bins(params, counts, dist: Distribution, quartiles=None, penalty=False,
                anchor=None, dirichlet=False, target_probs=None, taus=None,
                fix_middle_constraint=None):
    if dirichlet:
        alpha = params[-1]
        dist_params = params[:-1]
        if taus is not None:
            # counts is (M, B, K, T) or (M, B, T)
            expected_counts = jnp.sum(jnp.exp(counts) * taus, axis=-1)
            if expected_counts.ndim == 3:
                num_obs = expected_counts.sum(axis=(0, 1))
            else:
                num_obs = expected_counts.sum()
        else:
            # counts is (B, K) or (B,)
            if counts.ndim == 2:
                num_obs = counts.sum(axis=0)
            else:
                num_obs = counts.sum()
    else:
        dist_params = params
        num_obs = None
        alpha = 0.0
        
    m = len(dist.param_names)
    if len(dist_params) > m:
        dist_params = dist_params.reshape(m, -1)
    if anchor is not None:
        dist_params = jnp.insert(dist_params, anchor[0], anchor[1], axis=1)
        
    dist_params = dist.transform_params(dist_params)
    
    if quartiles is None:
        quartiles = estimate_quartiles(dist, dist_params, alpha=alpha, num_obs=num_obs, target_probs=target_probs)
        
    log_probs = reconstruct_logp(quartiles, dist, dist_params)
    
    if taus is not None:
        log_probs_expanded = log_probs[..., None]
        # SAFE MULTIPLICATION: prevent log_probs (-inf) * taus (0.0) -> NaN
        safe_taus = jnp.where(taus == 0.0, 1.0, taus)
        prob_term = jnp.where(taus == 0.0, 0.0, log_probs_expanded * safe_taus)
        term = prob_term + counts
        log_likelihood = jnp.sum(jax.scipy.special.logsumexp(term, axis=-1))
    else:
        log_likelihood = jnp.sum(log_probs * counts)
    
    if penalty:
        penalty_val = jnp.sum(dist.penalty(dist_params))
    else:
        penalty_val = 0.0

    res = -log_likelihood - penalty_val
    
    # ---------------------------------------------------------
    # IDENTIFIABILITY CONSTRAINT: Anchor the location translation
    # ---------------------------------------------------------
    if fix_middle_constraint is not None:
        mid_idx = len(target_probs) // 2
        # Apply a strong quadratic penalty. Since the translation gradient from the 
        # log-likelihood is 0, this penalty easily dominates and centers the mixture
        res += 1e6 * jnp.square(quartiles[mid_idx] - fix_middle_constraint)

    return res / log_probs.shape[0]

def run_adam(val_and_grad_fn, params, bounds=None, lr=0.01, num_steps=300):
    """A lightweight, fully JAX-jitted Adam optimizer that tracks the best state."""
    if bounds is not None:
        lows = jnp.array([b[0] if (b is not None and b[0] is not None) else -jnp.inf for b in bounds])
        highs = jnp.array([b[1] if (b is not None and b[1] is not None) else jnp.inf for b in bounds])
    else:
        lows, highs = None, None

    @jax.jit
    def optimize(init_p):
        init_val, _ = val_and_grad_fn(init_p)
        
        def step(state, _):
            p, m, v, i, best_p, best_val = state
            val, grad = val_and_grad_fn(p)
            
            # Standard Adam updates
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * jnp.square(grad)
            m_hat = m / (1.0 - jnp.power(0.9, i))
            v_hat = v / (1.0 - jnp.power(0.999, i))
            
            p_next = p - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            
            if lows is not None:
                p_next = jnp.clip(p_next, lows, highs)
            
            # Safety check: Update best parameters only if the loss is strictly better and valid
            is_valid = ~jnp.isnan(val)
            is_better = (val < best_val) & is_valid
            
            best_p_next = jnp.where(is_better, p, best_p)
            best_val_next = jnp.where(is_better, val, best_val)
            
            return (p_next, m, v, i + 1.0, best_p_next, best_val_next), None

        # State tuple: (current_p, m, v, step_idx, best_p, best_val)
        state = (init_p, jnp.zeros_like(init_p), jnp.zeros_like(init_p), 1.0, init_p, init_val)
        state, _ = jax.lax.scan(step, state, None, length=num_steps)
        
        # Return best parameters, best loss, and the initial loss for reporting
        return state[4], state[5], init_val
    
    return optimize(params)


@dataclass(frozen=True)
class DistfitResult:
    probs: np.ndarray
    params: np.ndarray
    alpha: float = None
    quartiles: np.ndarray = None
    stds: np.ndarray = None

def fit(observed_counts, dist: Distribution = Normal, correct_quartiles=True, penalty=False,
        quartile_anchor=False, dirichlet=False,
        ftol=None, gtol=None, use_cuda=False, quartile_mode=None,
        fix_middle_constraint: float = 0.0) -> DistfitResult:
    """
    Obtain estimates of latent distribution parameters. Alongside parameters,
    quartile probabilities are also reported for each object.

    Parameters
    ----------
    observed_counts : np.ndarray or list
        A tensor array representing counts. Expected sizes:
        - M x G x B: Raw count array aggregated across M replicates.
        - M x G x B x T: Log posterior counts distribution array across M replicates.
        - Single replicates are handled automatically and dimensions unified.
    dist : Distribution, optional
        Base latent distribution. The default is Normal.
    correct_quartiles : bool, optional
        Perform a joint global fit for all objects under a mixture model. Should produce
        more reliable results. The default is True.
    penalty : bool, optional
        Applies a MAP penalty to params of a distribution. Penalty is encoded by a respective
        Distribution class. The default is True.
    quartile_anchor : bool, optional
        Fix parameters of a single object prior to global fit to ensure that scale is location is identifiable. The default is False.
    dirichlet : bool, optional
        If False, the mixture model assumes that all objects have the same prior uniform probability, actual object
        frequencies are not taken into an account. If True, the model uses actual frequncies, but also applies
        an estimable pseudocount under the symmetric Dirichlet model. The default is False.
    ftol : bool, optional
        ftol to replace the default values in L-BFGS-B optimizer. The default is None.
    gtol : bool, optional
        gtol to replace the default values in L-BFGS-B optimizer. The default is None.
    use_cuda : bool, optional
        Perform computations on CUDA device if possible. The default is False.
    quartile_mode : str, tuple, or None, optional
        Mode for computing target quartiles/bin boundaries. Can be None or 'quartile' (default)
        for uniformly spaced probabilities, 'empirical' to compute probabilities empirically 
        from counts, or a custom tuple/array of target probabilities.

    Returns
    -------
    DistfitResult
    """
    
    if use_cuda:
        jax.config.update('jax_platforms', 'cuda')
    else:
        jax.config.update('jax_platforms', 'cpu')
    jax.config.update("jax_enable_x64", True)
    
    if isinstance(observed_counts, list):
        observed_counts = jnp.stack(observed_counts)
    else:
        observed_counts = jnp.asarray(observed_counts)
	

    is_log_prob = (observed_counts.ndim == 4) or (observed_counts.ndim == 3 and jnp.any(observed_counts < 0.0))

    if not is_log_prob:
        if observed_counts.ndim == 3:
            
            observed_counts = observed_counts.sum(axis=0)
            
        K, num_bins = observed_counts.shape
        counts = observed_counts + 0.1
        counts_for_target = counts
        counts_for_fit_all = counts.T
        counts_for_fit_k = counts
    else:
        if observed_counts.ndim == 3:
            observed_counts = observed_counts[None, ...] # Prepend replicate dim (1 x G x B x T)
            
        M, K, num_bins, T = observed_counts.shape
        taus_base = jnp.arange(T, dtype=jnp.float64)
        taus_plus_pc = taus_base + 0.1
        
        expected_counts = jnp.sum(jnp.exp(observed_counts) * taus_base, axis=-1)
        counts_for_target = expected_counts.sum(axis=0) + 0.1
        
     
        counts_for_fit_all = observed_counts.transpose((0, 2, 1, 3)) # (M, B, K, T)
        counts_for_fit_k = observed_counts.transpose((1, 0, 2, 3)) # (K, M, B, T)

    if quartile_mode is None or (isinstance(quartile_mode, str) and quartile_mode.lower() == 'quartile'):
        target_probs = jnp.arange(1, num_bins) / num_bins
    elif isinstance(quartile_mode, str) and quartile_mode.lower() == 'empirical':
        target_probs = jnp.cumsum(counts_for_target.sum(axis=0) / counts_for_target.sum())[:-1]
    elif isinstance(quartile_mode, (list, tuple, np.ndarray, jnp.ndarray)):
        target_probs = jnp.asarray(quartile_mode)
    else:
        raise ValueError(f"Invalid quartile_mode: {quartile_mode}")
    
    quartiles_marginal = dist.baseline_quartiles(num_bins)
    
    if is_log_prob:
        obj_fun = partial(loglik_bins, penalty=penalty, dist=dist, fix_middle_constraint=None,
                          target_probs=target_probs, taus=taus_plus_pc)
    else:
        obj_fun = partial(loglik_bins, penalty=penalty, dist=dist, fix_middle_constraint=None,
                          target_probs=target_probs)
        
    params = list()
    x0 = dist.staring_values()
    
    val_and_grad = jax.value_and_grad(partial(obj_fun, quartiles=quartiles_marginal), 0)
    val_and_grad = jax.jit(val_and_grad)
    fun = partial(val_and_grad, counts=counts_for_fit_all)
    x0s = np.repeat(x0.reshape(-1,1), K, axis=1).flatten()
    best_val = fun(x0s)[0]
    res = minimize(fun,
                   x0s, jac=True,
                   options={'maxiter': 10000}, method='L-BFGS-B')
    if res.fun < best_val:
        tres_x0 = res.x
        best_val = res.fun
    else:
        tres_x0 = x0s
    tres = minimize(fun, tres_x0, jac=True,
                    options={'maxiter': 10000}, method='TNC')
    if tres.fun < best_val:
        best_val = tres.fun
        res = tres
    
    x0s = res.x.reshape(len(x0), -1).T
    if not res.success:
        print(res)
    funs = list()
    print("Starting local pre-training...")
    for k in tqdm(range(K)):
        fun = partial(val_and_grad, counts=counts_for_fit_k[k])
        res = minimize(
            fun=fun, x0=x0s[k], jac=True,
            method='SLSQP', options={'maxiter': 10000}
        )
        if not res.success:
            print(res)
            res_alt = minimize(fun=fun, x0=x0s[k], jac=True, 
                              method='TNC', options={'maxiter': 1000})
            res = min((res, res_alt), key=lambda x: x.fun)
        x = res.x
        params.append(x)
        funs.append(res.fun)
    
    params = jnp.array(params).T
    
    if correct_quartiles:
        if quartile_anchor:
            i = np.argmin(funs)
            anchor = (int(i), params[:, i])
            params = np.delete(params, i, axis=1)
        else:
            anchor = None
            
        params_1d = np.array(params).flatten()
        if dirichlet:
            if is_log_prob:
                counts_for_fit_cq = counts_for_fit_all
                taus_cq = taus_base
            else:
                counts_for_fit_cq = observed_counts.T
            params_1d = jnp.append(params_1d, 100.0) 
        else:
            if is_log_prob:
                counts_for_fit_cq = counts_for_fit_all
                taus_cq = taus_plus_pc
            else:
                counts_for_fit_cq = counts.T
            
        if is_log_prob:
            fun_base = partial(loglik_bins, counts=counts_for_fit_cq, penalty=penalty, dist=dist, anchor=anchor, 
                          dirichlet=dirichlet, target_probs=target_probs, taus=taus_cq,
                          fix_middle_constraint=fix_middle_constraint)
        else:
            fun_base = partial(loglik_bins, counts=counts_for_fit_cq, penalty=penalty, dist=dist, anchor=anchor, 
                          dirichlet=dirichlet, target_probs=target_probs,
                          fix_middle_constraint=fix_middle_constraint)
                          
        fun = jax.jit(jax.value_and_grad(fun_base, argnums=0))
        
        options = {'maxiter': 10000}
        if ftol is not None:
            options['ftol'] = ftol
        if gtol is not None:
            options['gtol'] = gtol
        bounds = None
        if dirichlet:
            bounds = [(None, None)] * len(params_1d)
            bounds[-1] = (1e-6, None)

        print("Starting Global Optimization...")
        
        adam_lr = 0.01
        adam_steps = 300
        best_val = fun(params_1d)[0]
        for try_idx in range(4):
            if try_idx > 0:
                print(f"\n--- Retrying Global Optimization (Try {try_idx + 1}/4) ---")
                
            print("Running L-BFGS-B...")
            res = minimize(fun, params_1d, jac=True, bounds=bounds,
                           options=options, method='L-BFGS-B')
            print(res)
            
            print("Running TNC...")
            tres_x0 = params_1d if res.fun > best_val else res.x
            tres = minimize(fun, tres_x0, jac=True, bounds=bounds,
                           options=options, method='TNC')
            print(tres)
            
            if tres.fun < res.fun:
                res = tres
                improved = True
                best_val = res.fun
            else:
                improved = False
                
            # Check for convergence
            if res.success:
                print(f"Global optimization converged successfully on Try {try_idx + 1}.")
                break
            else:
                if try_idx < 3:
                    print(f"Optimization did not converge completely. Preparing for Try {try_idx + 2}...")
                    if res.success or improved:
                        params_1d = res.x
                    print(f"Running Adam Optimizer Pre-training (lr={adam_lr}, steps={adam_steps})...")
                    best_adam_p, best_adam_val, start_adam_val = run_adam(fun, params_1d, bounds=bounds, 
                                                                          lr=adam_lr, num_steps=adam_steps)
                    
                    if best_adam_val < start_adam_val:
                        print(f"Adam accepted: improved loss from {start_adam_val:.4f} to {best_adam_val:.4f}")
                        params_1d = np.array(best_adam_p)
                        best_val = best_adam_val
                    else:
                        print(f"Adam rejected: could not improve upon starting loss of {start_adam_val:.4f}")
                        best_val = start_adam_val
                    adam_lr /= 2.0
                    adam_steps *= 2
                else:
                    print("Warning: Final joint fit didn't converge perfectly after 4 tries:", res.message)

        if dirichlet:
            params_u = res.x[:-1].reshape(len(x0), -1)
            alpha_est = res.x[-1]
        else:
            params_u = res.x.reshape(len(x0), -1)
            
        if quartile_anchor:
            params_u = np.insert(params_u, anchor[0], anchor[1], axis=1)
    else:
        # If we didn't run correct_quartiles, `params` are already unconstrained
        params_u = np.array(params)
 
    # Propagate unconstrained params to constrained params for returning
    params_c = dist.transform_params(params_u)
    
    if correct_quartiles and dirichlet:
        final_alpha = alpha_est
        if is_log_prob:
            final_num_obs = expected_counts.sum(axis=(0, 2))
        else:
            final_num_obs = observed_counts.sum(axis=1)
    else:
        final_alpha = 0.0
        final_num_obs = None
        
    quartiles_est = estimate_quartiles(dist, params_c, x0=quartiles_marginal,
                                       num_obs=final_num_obs, alpha=final_alpha,
                                       target_probs=target_probs)
                                       
    if correct_quartiles:
        quartiles_marginal = quartiles_est

    # =====================================================================
    # STANDARD ERROR COMPUTATION VIA EXPECTED FIM & DELTA METHOD
    # =====================================================================
    N_all = jnp.sum(counts_for_target, axis=1) # shape (K,) containing total counts per sequence

    @jax.jit
    def compute_all_stds(p_u_all, N_all_g):
        return jax.vmap(
            lambda p_u, N: get_sequence_std_err(p_u, N, quartiles_marginal, dist, penalty)
        )(p_u_all, N_all_g)
        
    try:
        stds = compute_all_stds(params_u.T, N_all)
        stds = np.array(stds)
    except Exception as e:
        print("Warning: Standard error computation failed:", e)
        stds = None
    # =====================================================================
        
    p = np.exp(reconstruct_logp(quartiles_marginal, dist, params_c))
    params_c = np.array(params_c)
    
    return DistfitResult(p.T, params_c.T, alpha=final_alpha, quartiles=quartiles_marginal, stds=stds)