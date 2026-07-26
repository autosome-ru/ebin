import numpy as np
from sklearn.cluster import KMeans
import jax
import jax.numpy as jnp
from jax.scipy.stats import nbinom
from jax.scipy.special import logsumexp
from functools import partial



@partial(jax.jit, static_argnames=['num_mixtures', 'bin_shape']) 
def _e_step(params, data_2d, num_mixtures, bin_shape):
    """Performs the E-step of the EM algorithm (generalized)."""
    B, M = bin_shape
    n_sequences, n_conditions = data_2d.shape
    
    # log(w_k)
    log_w = jnp.log(params['w'])
    
    # Calculate log likelihood of data under each NB component
    # Shape: (n_sequences, n_conditions, num_mixtures)
    log_nb_pmfs = jnp.zeros((n_sequences, n_conditions, num_mixtures))
    for k in range(num_mixtures):
        # nbinom.logpmf is broadcast-safe
        log_nb_pmfs = log_nb_pmfs.at[:, :, k].set(
            nbinom.logpmf(data_2d, n=params['r'][k, :], p=params['p'][k, :])
        )

    # log P(y|NB_k) + log(w_k)
    log_weighted_nb = log_nb_pmfs + log_w[jnp.newaxis, jnp.newaxis, :]
    
    # log( sum_k(w_k * P(y|NB_k)) )
    log_sum_nb = logsumexp(log_weighted_nb, axis=2)
    
    # --- Zero-inflation part with new `pi` shape ---
    # `pi` has shape (B,). We need to map it to each condition.
    # Condition `j` belongs to bin `j // M`.
    bin_map = jnp.arange(n_conditions) // M
    pi_mapped = params['pi'][bin_map] # Shape: (n_conditions,)
    
    log_pi = jnp.log(pi_mapped)
    log_one_minus_pi = jnp.log(1 - pi_mapped)
    
    is_zero = (data_2d == 0)
    
    # Broadcast log_pi to (n_sequences, n_conditions) for logsumexp
    log_pi_broadcast = jnp.broadcast_to(log_pi, (n_sequences, n_conditions))
    
    log_prob_for_zeros = logsumexp(jnp.stack([
        log_pi_broadcast,
        log_one_minus_pi + log_sum_nb
    ]), axis=0)
    
    log_prob_nonzero = log_one_minus_pi + log_sum_nb
    
    log_likelihood_data = jnp.where(is_zero, log_prob_for_zeros, log_prob_nonzero)
    
    # --- Calculate Responsibilities ---
    log_resp_zero = log_pi - log_likelihood_data
    
    log_resp_nb = (log_one_minus_pi[jnp.newaxis, :, jnp.newaxis] + 
                   log_weighted_nb - 
                   log_likelihood_data[:, :, jnp.newaxis])
    
    responsibilities = {
        'zero': jnp.where(is_zero, jnp.exp(log_resp_zero), 0),
        'nb': jnp.exp(log_resp_nb) # Shape: (n_seq, n_cond, n_mix)
    }
    
    return responsibilities, jnp.sum(log_likelihood_data)


@partial(jax.jit, static_argnames=['num_mixtures', 'bin_shape']) # ADDED 'bin_shape'
def _m_step(responsibilities, data_2d, num_mixtures, bin_shape):
    """Performs the M-step of the EM algorithm (generalized)."""
    B, M = bin_shape
    n_sequences, n_conditions = data_2d.shape

    # --- Update pi (zero-inflation per bin) ---
    resp_zero_reshaped = responsibilities['zero'].reshape(n_sequences, B, M)
    # Sum responsibilities over all sequences and replicates in a bin
    sum_resp_zero_per_bin = jnp.sum(resp_zero_reshaped, axis=(0, 2))
    # Divide by total number of data points in that bin (N * M)
    pi_new = sum_resp_zero_per_bin / (n_sequences * M)

    # --- Update w (shared mixture weights) ---
    total_resp_nb_k = jnp.sum(responsibilities['nb'], axis=(0, 1))
    w_new = total_resp_nb_k / jnp.sum(total_resp_nb_k)
    
    # --- Update r and p for each NB component and condition ---
    r_new = jnp.zeros((num_mixtures, n_conditions))
    p_new = jnp.zeros((num_mixtures, n_conditions))

    for k in range(num_mixtures):
        resp_k = responsibilities['nb'][:, :, k]  # (n_sequences, n_conditions)
        sum_resp_k = jnp.sum(resp_k, axis=0)
        
        mu_k = jnp.sum(resp_k * data_2d, axis=0) / (sum_resp_k + 1e-10)
        var_k = jnp.sum(resp_k * (data_2d - mu_k[jnp.newaxis, :])**2, axis=0) / (sum_resp_k + 1e-10)
        
        var_k = jnp.maximum(var_k, mu_k + 1e-6) # Ensure overdispersion
        
        p_k = mu_k / var_k
        r_k = mu_k * p_k / (1 - p_k)

        p_new = p_new.at[k, :].set(jnp.clip(p_k, 1e-6, 1 - 1e-6))
        r_new = r_new.at[k, :].set(jnp.maximum(1e-6, r_k))

    return {'w': w_new, 'pi': pi_new, 'r': r_new, 'p': p_new}


def _initialize_params(data_2d, num_mixtures, bin_shape):
    """Initializes model parameters using k-means."""
    B, M = bin_shape
    n_sequences, n_conditions = data_2d.shape
    
    # Global mixture weights
    w = jnp.ones(num_mixtures) / num_mixtures
    
    # Bin-specific zero inflation
    data_reshaped = data_2d.reshape(n_sequences, B, M)
    pi = jnp.mean(data_reshaped == 0, axis=(0, 2)) * 0.5 # Start conservatively
    
    # Condition-specific NB params
    r = jnp.zeros((num_mixtures, n_conditions))
    p = jnp.zeros((num_mixtures, n_conditions))

    for j in range(n_conditions):
        col_data = data_2d[:, j]
        non_zero_data = col_data[col_data > 0].reshape(-1, 1)
        
        if len(non_zero_data) < num_mixtures:
            mean_nz = jnp.mean(non_zero_data) if len(non_zero_data) > 0 else 1.0
            var_nz = jnp.var(non_zero_data) if len(non_zero_data) > 1 else 2.0
            if var_nz <= mean_nz: var_nz = mean_nz + 1.0
            
            p_j = mean_nz / var_nz
            r_j = mean_nz * p_j / (1 - p_j)
            
            r = r.at[:, j].set(jnp.linspace(r_j * 0.8, r_j * 1.2, num_mixtures))
            p = p.at[:, j].set(jnp.repeat(p_j, num_mixtures))
            continue

        kmeans = KMeans(n_clusters=num_mixtures, random_state=42, n_init='auto').fit(non_zero_data)
        sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
        
        for k_idx in sorted_indices:
            cluster_data = non_zero_data[kmeans.labels_ == k_idx]
            mean_k, var_k = jnp.mean(cluster_data), jnp.var(cluster_data)
            
            if var_k <= mean_k: var_k = mean_k + 1e-4
            
            p_k = mean_k / var_k
            r_k = mean_k * p_k / (1 - p_k)
            
            comp_idx = list(sorted_indices).index(k_idx)
            r = r.at[comp_idx, j].set(jnp.maximum(1e-6, r_k))
            p = p.at[comp_idx, j].set(jnp.clip(p_k, 1e-6, 1 - 1e-6))

    return {'w': w, 'pi': pi, 'r': r, 'p': p}


def estimate_params_mom(X: np.ndarray, K: int, random_state: int = 42, params: dict = None):
    """
    Method of Moments initialization for the Hierarchical Gamma-Poisson-NB model.
    """
    B, S, G = X.shape
    
    X_flat = X.transpose(2, 0, 1).reshape(G, B * S)
    X_log = np.log1p(X_flat)
    
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    z_g = kmeans.fit_predict(X_log)
    
    w = np.zeros(K)
    for k in range(K):
        w[k] = np.mean(z_g == k)
        
    mu = np.zeros((B, S, K))
    var = np.zeros((B, S, K))
    
    sum_mu_prods = 0.0
    sum_covs = 0.0
    
    for k in range(K):
        mask = (z_g == k)
        N_k = np.sum(mask)
        
        if N_k < 2:
            mu[:, :, k] = 1e-6
            var[:, :, k] = 1e-6
            continue
            
        X_k = X[:, :, mask]
        
        mu[:, :, k] = np.mean(X_k, axis=2)
        var[:, :, k] = np.var(X_k, axis=2, ddof=1)
        
        if S > 1:
            for b in range(B):
                X_bk = X_k[b] # Shape: (S, N_k)
                cov_mat = np.cov(X_bk)
                
                sum_off_diag_cov = (np.sum(cov_mat) - np.sum(np.diag(cov_mat))) / 2.0
                sum_covs += sum_off_diag_cov
                
                mu_bk = mu[b, :, k]
                sum_off_diag_mu = (np.sum(mu_bk)**2 - np.sum(mu_bk**2)) / 2.0
                sum_mu_prods += sum_off_diag_mu
                
    if S > 1 and sum_covs > 1e-8:
        a = sum_mu_prods / sum_covs
        a = np.clip(a, 1e-2, 1e4) # Bound to sane physical values
    else:
        a = 10.0 # Default fallback if S=1 or data is highly noisy
        
    b_param = a # Enforce beta = alpha for scale identifiability
    
    mu_safe = np.maximum(mu, 1e-6)
    D = var - (mu_safe**2) * (1.0 + 1.0 / a)
    valid_mask = D > (mu_safe * 1e-4)
    
    p = np.where(valid_mask, mu_safe / np.maximum(D, 1e-11), 0.99999)
    p = np.clip(p, 1e-6, 0.999999) 
    
    r = mu_safe * p / (1.0 - p)
    r = np.maximum(r, 1e-12) 
    
    new_params = params.copy()
    new_params['r'] = r
    new_params['p'] = p
    new_params['w'] = w
    
    # We only return `a` since `beta=1.0` dynamically inside the EM loop.
    new_params['theta'] = np.array([a])
    return new_params

def fit_nb(data: jnp.ndarray, num_mixtures: int, max_iter: int = 100, tol: float = 1e-7, verbose: bool = True):
    """
    Fits a Zero-Inflated Negative Binomial (ZINB) mixture model to count data.
    """
    assert data.ndim == 3, "Input data must be a 3D array of shape (B, M, N)."
    B, M, N = data.shape
    bin_shape = (B, M)
    
    data_2d = data.transpose((2, 0, 1)).reshape((N, B * M))
    
    params = _initialize_params(data_2d, num_mixtures, bin_shape)
    log_likelihoods = []

    for i in range(max_iter):
        responsibilities, ll = _e_step(params, data_2d, num_mixtures, bin_shape)
        params = _m_step(responsibilities, data_2d, num_mixtures, bin_shape)
        
        log_likelihoods.append(ll.item())
        
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            if verbose:
                print(f"Converged after {i+1} iterations.")
            break
    else:
        if verbose:
            print("Reached max iterations.")
    

    final_params = {
        'w': params['w'], 
        'pi': params['pi'],
        'r': params['r'].reshape(num_mixtures, B, M).transpose((1, 2, 0)),
        'p': params['p'].reshape(num_mixtures, B, M).transpose((1, 2, 0)),
    }
    
    # Retrieve the estimated alpha rate and divide (so that Expected Count matches the plain NB fit mapping)
    alpha = estimate_params_mom(data, num_mixtures, params=final_params)['theta'][0]
    final_params['r'] = final_params['r'] / alpha
    final_params['theta'] = np.array([alpha,])
    
    return final_params, log_likelihoods