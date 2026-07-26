from functools import partial
try:
    from .ftzkiller import FTZKiller
except ImportError:  # allow running as a loose module
    from ftzkiller import FTZKiller
from scipy.linalg import eigh_tridiagonal as scipy_eigh_tridiagonal
from jax import pure_callback
from scipy.special import roots_genlaguerre
from jax.scipy.special import gammaln
from jax.lax import scan
from jax import vmap
import jax.numpy as jnp
import numpy as np
import jax

from enum import Enum 

class Rules(Enum):
    GenLaguerre = 'GenLaguerre'
    Charlier = 'Charlier'
    ShiftedJacobi = 'ShiftedJacobi'



def _calculate_for_single_lambda(lam, diagonal, off_diagonal):
    """
    Helper function to compute log(|[v]_1|) for a particualr eigenvalue.
    """

    # Pre-calculate the first two components (indices 0 and 1)
    u0 = 1.0
    u1 = -(diagonal[0] - lam) * u0 / off_diagonal[0]

    initial_carry = (u1, u0)
    
    # The scan will compute components u[2] through u[n-1], which is n-2 steps.
    # All input arrays for the scan must therefore have length n-2.
    scan_diagonals = diagonal[1:-1]             # Needs a_2, ..., a_{n-1} (0-indexed: diag[1]...diag[n-2])
    scan_off_diagonals_prev = off_diagonal[:-1] # Needs b_1, ..., b_{n-2} (0-indexed: off[0]...off[n-3])
    scan_off_diagonals_curr = off_diagonal[1:]  # Needs b_2, ..., b_{n-1} (0-indexed: off[1]...off[n-2])

    def _recurrence_step(carry, scan_inputs):
        u_curr, u_prev = carry
        a_k, b_k_minus_1, b_k = scan_inputs
        
        u_next = -((a_k - lam) * u_curr + b_k_minus_1 * u_prev) / b_k
        
        new_carry = (u_next, u_curr)
        component_to_collect = u_next
        return new_carry, component_to_collect

    # Run the scan to get components u_2, ..., u_{n-1}
    _, u_rest = scan(
        _recurrence_step,
        initial_carry,
        (scan_diagonals, scan_off_diagonals_prev, scan_off_diagonals_curr)
    )

    # Combine all components into a single vector u
    u = jnp.concatenate([jnp.array([u0, u1]), u_rest])

    # Stably compute the log of the first component's magnitude 
    u_max = jnp.max(jnp.abs(u))
    w = u / u_max
    
    sum_w_sq = jnp.dot(w, w)
    log_s_squared = 2 * jnp.log(u_max) + jnp.log(sum_w_sq)
    
    result = jnp.where(u_max > 0, -0.5 * log_s_squared, -jnp.inf)
    
    return result



@jax.jit
def calc_first_components(diagonal, off_diagonal, eigenvalues):
    """
    Calculates the logarithm of the absolute value of the first components of all
    eigenvectors of a symmetric tridiagonal matrix.

    Args:
        diagonal (jnp.ndarray): The main diagonal of the matrix (a_1, ..., a_n).
        off_diagonal (jnp.ndarray): The first off-diagonal (b_1, ..., b_{n-1}).
        eigenvalues (jnp.ndarray): The pre-computed eigenvalues of the matrix.

    Returns:
        jnp.ndarray: An array containing log(|[v_j]_1|) for each eigenvector v_j.
    """
    n = diagonal.shape[0]

    # Handle edge cases that don't fit the scan logic
    if n == 0:
        return jnp.array([])
    if n == 1:
        return jnp.array([0.0])
    if n == 2:
        def _calc_n2(lam, diag, offdiag):
            u0 = 1.0
            u1 = -(diag[0] - lam) * u0 / offdiag[0]
            u = jnp.array([u0, u1])
            u_max = jnp.max(jnp.abs(u))
            w = u / u_max
            sum_w_sq = jnp.dot(w, w)
            log_s_squared = 2 * jnp.log(u_max) + jnp.log(sum_w_sq)
            return -0.5 * log_s_squared
        
        return vmap(_calc_n2, in_axes=(0, None, None))(eigenvalues, diagonal, off_diagonal)

    vmapped_calculator = vmap(
        _calculate_for_single_lambda, in_axes=(0, None, None)
    )
    
    return vmapped_calculator(eigenvalues, diagonal, off_diagonal)

@jax.custom_jvp
def eigh_tridiagonal(diag, offdiag):
  def scipy_eigh_vals_only(d, o):
      return scipy_eigh_tridiagonal(d, o, eigvals_only=True)
  
  n = diag.shape[0]
  with FTZKiller():
      w = pure_callback(
          scipy_eigh_vals_only,
          jax.ShapeDtypeStruct((n,), diag.dtype), # Shape of the output
          diag, offdiag,
          vmap_method='sequential',
      )
      w = jax.block_until_ready(w)
  return w


@eigh_tridiagonal.defjvp
def _eigh_tridiagonal_jvp(primals, tangents):
  diag, offdiag = primals
  diag_dot, offdiag_dot = tangents

  def scipy_eigh_full(d, o):
    with FTZKiller():
        return scipy_eigh_tridiagonal(d, o, eigvals_only=False)
  
  n = diag.shape[0]
  
  w, v = pure_callback(
       scipy_eigh_full,
        (
            jax.ShapeDtypeStruct((n,), diag.dtype),
            jax.ShapeDtypeStruct((n, n), diag.dtype)
        ),
        diag, offdiag,
        vmap_method='sequential',
    )
  
  delta_T = jnp.diag(diag_dot) + jnp.diag(offdiag_dot, k=1) + jnp.diag(offdiag_dot, k=-1)
  
  w_dot = jnp.einsum('ik,ij,jk->k', v, delta_T, v)

  return w, w_dot

def compute_roots_genlaguerre_scipy(n: int, alpha: float):
    n = int(n)
    alpha = float(alpha)
    
    with FTZKiller():
        roots, weights = roots_genlaguerre(n, alpha)
    return roots.astype(np.float64), weights.astype(np.float64)

def approximate_charliet_smallest_root(n: int, mu: float) -> float:
    """
    This function uses the formula:
        x_1 ≈ μ^n / Σ_{k=1 to n} [ C(n,k) * (k-1)! * μ^(n-k) ]
    """

    log_mu = jnp.log(mu)
    log_numerator = n * log_mu

    k_values = jnp.arange(1, n + 1)

    log_binom_coeff = gammaln(n + 1) - gammaln(k_values + 1) - gammaln(n - k_values + 1)
    
    log_factorial = gammaln(k_values)

    log_mu_term = (n - k_values) * log_mu
    log_terms_denominator = log_binom_coeff + log_factorial + log_mu_term

    log_denominator = jax.scipy.special.logsumexp(log_terms_denominator)
    log_root = log_numerator - log_denominator

    return jnp.exp(log_root)

def construct_jacobi(alpha, beta, n):
    """
    Constructs the tridiagonal Jacobi matrix for w(x) = x^(alpha-1)(1-x)^(beta-1).
    
    Args:
        alpha (float): Parameter alpha > 0.
        beta (float): Parameter beta > 0.
        n (int): The size of the matrix (n x n).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       - diag (main diagonal, size n)
                                       - offdiag (sub/super-diagonal, size n-1)
    """
    if n <= 0:
        return jnp.array([]), jnp.array([])

    k = jnp.arange(n, dtype=np.float64)
    ab = alpha + beta

    diag = jnp.where(
        k == 0,
        alpha / ab,
        0.5 * (1 + ((alpha - 1)**2 - (beta - 1)**2) / ((2*k + ab - 2) * (2*k + ab)))
    )

    if n == 1:
        return diag, jnp.array([])

    k_off = k[:-1]
    c = 2*k_off + ab
    offdiag = jnp.where(
        k_off == 0,
        jnp.sqrt((alpha * beta) / (ab**2 * (ab + 1))),
        (1/c) * jnp.sqrt(((k_off + 1)*(k_off + alpha)*(k_off + beta)*(k_off + ab - 1)) / (c**2 - 1))
    )

    return diag, offdiag

@partial(jax.jit, static_argnames=('num_nodes', 'rule', 'norm', 'eigh_solver',
                                   ))
def compute_nodes_and_logweights(num_nodes: int, param: float, rule: Rules,
                                 norm: bool = True, eigh_solver: bool = None):
    if eigh_solver is None:
        if rule == Rules.Charlier:
            eigh_solver = True
        else:
            eigh_solver = False
    k = jnp.arange(num_nodes)
    lognorm = 0.0
    if rule == Rules.GenLaguerre:
        if not norm:
            lognorm = gammaln(param + 1)
        diag = 2 * k + param + 1
        k_off = jnp.arange(1, num_nodes, dtype=float)
        offdiag = -jnp.sqrt(k_off * (k_off + param))
    elif rule == Rules.Charlier:
        diag = k + param
        k = jnp.arange(1, num_nodes)
        offdiag = jnp.sqrt(k * param)
        if not norm:
            lognorm = param 
    elif rule == Rules.ShiftedJacobi:
        diag, offdiag = construct_jacobi(*param, num_nodes)
        if not norm:
            lognorm = jax.scipy.special.betaln(*param)
    if eigh_solver:
        d = jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
        with FTZKiller():
            nodes, logweights = jnp.linalg.eigh(d)
            logweights = jnp.log(jnp.abs(logweights.at[0].get()))
    else:
        nodes = eigh_tridiagonal(diag, offdiag)
        logweights = calc_first_components(diag, offdiag, nodes)
    logweights = lognorm + 2 * logweights
    return jnp.clip(nodes, 0.0, None), logweights

# import numdifftools as nd 

# jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_platforms', 'cpu')


# @partial(jax.jit, static_argnames=('n', 'mode'))
# def fun(alpha, mode: bool, n=20):
#     n, l = compute_nodes_and_logweights(n, alpha, rule=Rules.Charlier,
#                                         eigh_solver=True)
#     return ((l  ) ).mean()

# f_old = partial(fun, mode=True)
# grad_old = jax.jit(jax.jacfwd(f_old, argnums=0))
# grad_old_nd = nd.Gradient(f_old)

# # f_new= partial(fun, mode=False)
# # grad_new = jax.jit(jax.jacrev(f_new, argnums=0))
# # grad_new_nd = nd.Gradient(f_new)


# alpha = 1e-3

# print('funs:', f_old(alpha), )
# print('grad:', grad_old(alpha), )
# print('grad_nd:', grad_old_nd(alpha), )
# print('graddiff', np.abs(grad_old(alpha) - grad_old_nd(alpha)))




