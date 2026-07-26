import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax.scipy.stats as stats


class Distribution(ABC):
    
    param_names = ()
    support = (-float('inf'), float('inf'))
    
    @staticmethod
    @abstractmethod
    def logcdf(x: jnp.ndarray, params: jnp.ndarray):
        pass
    
    @staticmethod
    @abstractmethod
    def logsf(x: jnp.ndarray, params: jnp.ndarray):
        pass
    
    @staticmethod
    @abstractmethod
    def logpdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def pdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def cdf(x: jnp.ndarray, params: jnp.ndarray):
        pass
    
    @staticmethod
    @abstractmethod
    def sf(x: jnp.ndarray, params: jnp.ndarray):
        pass
    
    @staticmethod
    def penalty(params: jnp.ndarray):
        return 0.0
    
    @staticmethod
    @abstractmethod
    def baseline_quartiles(num_bins: int) -> jnp.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def _starting_values() -> jnp.ndarray:
        pass

    # Kept for backward compatibility if other scripts reference it
    @staticmethod
    def _staring_values() -> jnp.ndarray:
        return Distribution._starting_values()
    
    @classmethod
    def starting_values(cls) -> jnp.ndarray:
        return cls.inverse_transform_params(cls._starting_values())

    @classmethod
    def staring_values(cls) -> jnp.ndarray:
        return cls.starting_values()
    
    @staticmethod
    @abstractmethod
    def transform_params(params: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def inverse_transform_params(params: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def normalize_param_estimates(params: jnp.ndarray, stds: jnp.ndarray = None, quartiles: jnp.ndarray = None):
        pass

    @classmethod
    @abstractmethod
    def get_background_starting_values(cls, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        pass
    

class Normal(Distribution):
    param_names = ('mu', 'sigma')
    support = (-float('inf'), float('inf'))
    
    @staticmethod
    def logcdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.logcdf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def cdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.cdf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def logsf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.logsf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def sf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.sf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def logpdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.logpdf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def pdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        return stats.norm.pdf(x, loc=mu, scale=sigma)
    
    @staticmethod
    def baseline_quartiles(num_bins: int) -> jnp.ndarray:
        p = jnp.arange(num_bins)[1:] / num_bins
        return stats.norm.ppf(p)
    
    @staticmethod
    def penalty(params: jnp.ndarray, mu_sigma=2.0, sigma_alpha=3.0, sigma_beta=1.0):
        mu, sigma = params
        sigma = -(sigma_alpha + 1) * jnp.log(sigma) - sigma_beta / sigma
        mu = - mu ** 2 / (2 * mu_sigma ** 2)
        return sigma + mu
    
    @staticmethod
    def transform_params(params: jnp.ndarray, quartile_constraint: bool = False) -> jnp.ndarray:
        params = jnp.asarray(params)
        params = params.at[1].set(jnp.exp(params[1]))
        return params
    
    @staticmethod
    def inverse_transform_params(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.asarray(params)
        params = params.at[1].set(jnp.log(params[1]))
        return params
    
    @staticmethod
    def _starting_values() -> jnp.ndarray:
        return jnp.array([0.0, 0.5])
    
    @staticmethod
    def normalize_param_estimates(params: jnp.ndarray, stds: jnp.ndarray = None, quartiles: jnp.ndarray = None):
        mu_mean = params.at[0].get().mean()
        params = params.at[0].subtract(mu_mean)
        
        var = (params ** 2).sum(axis=0).mean()
        k = (1 / var) ** 0.5
        params = params * k
        
        res = {'params': params}
        
        if stds is not None:
            res['stds'] = stds * k
            
        if quartiles is not None:
            res['quartiles'] = (quartiles - mu_mean) * k
        if stds is None and quartiles is None:
            return params
        elif stds is not None and quartiles is None:
            return params, res['stds']
        elif stds is None and quartiles is not None:
            return params, res['quartiles']
        else:
            return params, res['stds'], res['quartiles']

    @classmethod
    def get_background_starting_values(cls, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu = jnp.linspace(-2.0, 2.0, n)
        sigma = jnp.ones(n) * 1.0
        params = jnp.vstack([mu, sigma])
        omega_logits = jnp.zeros(n)
        return cls.inverse_transform_params(params), omega_logits


class LeftTruncatedNormal(Distribution):
    param_names = ('mu', 'sigma')
    left = -1.5
    support = (left, float('inf'))
    
    @staticmethod
    def logcdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        cdf_val = LeftTruncatedNormal.cdf(x, params)
        return jnp.where(x >= a, jnp.log(jnp.maximum(cdf_val, 1e-300)), -jnp.inf)
    
    @staticmethod
    def cdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        num = stats.norm.cdf(x, loc=mu, scale=sigma) - stats.norm.cdf(a, loc=mu, scale=sigma)
        den = stats.norm.sf(a, loc=mu, scale=sigma)
        return jnp.where(x >= a, jnp.maximum(num / den, 0.0), 0.0)
    
    @staticmethod
    def logsf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        log_sf_x = stats.norm.logsf(x, loc=mu, scale=sigma)
        log_sf_a = stats.norm.logsf(a, loc=mu, scale=sigma)
        return jnp.where(x >= a, log_sf_x - log_sf_a, 0.0)
    
    @staticmethod
    def sf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        sf_x = stats.norm.sf(x, loc=mu, scale=sigma)
        sf_a = stats.norm.sf(a, loc=mu, scale=sigma)
        return jnp.where(x >= a, jnp.minimum(sf_x / sf_a, 1.0), 1.0)
    
    @staticmethod
    def logpdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        log_pdf_x = stats.norm.logpdf(x, loc=mu, scale=sigma)
        log_sf_a = stats.norm.logsf(a, loc=mu, scale=sigma)
        return jnp.where(x >= a, log_pdf_x - log_sf_a, -jnp.inf)
    
    @staticmethod
    def pdf(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = params
        x = jnp.asarray(x)
        a = LeftTruncatedNormal.left
        pdf_x = stats.norm.pdf(x, loc=mu, scale=sigma)
        sf_a = stats.norm.sf(a, loc=mu, scale=sigma)
        return jnp.where(x >= a, pdf_x / sf_a, 0.0)
    
    @staticmethod
    def baseline_quartiles(num_bins: int) -> jnp.ndarray:
        p = jnp.arange(num_bins)[1:] / num_bins
        a = LeftTruncatedNormal.left
        p_a = stats.norm.cdf(a)
        p_prime = p * stats.norm.sf(a) + p_a
        return stats.norm.ppf(p_prime)
    
    @staticmethod
    def penalty(params: jnp.ndarray, mu_sigma=2.0, sigma_alpha=3.0, sigma_beta=1.0):
        mu, sigma = params
        sigma = -(sigma_alpha + 1) * jnp.log(sigma) - sigma_beta / sigma
        mu = - mu ** 2 / (2 * mu_sigma ** 2)
        return sigma + mu
    
    @staticmethod
    def transform_params(params: jnp.ndarray, quartile_constraint: bool = False) -> jnp.ndarray:
        params = jnp.asarray(params)
        params = params.at[1].set(jnp.exp(params[1]))
        return params
    
    @staticmethod
    def inverse_transform_params(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.asarray(params)
        params = params.at[1].set(jnp.log(params[1]))
        return params
    
    @staticmethod
    def _starting_values() -> jnp.ndarray:
        return jnp.array([0.0, 0.5])
    
    @staticmethod
    def normalize_param_estimates(params: jnp.ndarray, stds: jnp.ndarray = None,
                                  quartiles = None):
        res = [params]
        if stds is not None:
            res.append(stds)
        if quartiles is not None:
            res.append(quartiles)
        if len(res) == 1:
            return res[0]
        return tuple(res)

    @classmethod
    def get_background_starting_values(cls, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu = jnp.linspace(-1.0, 2.0, n)
        sigma = jnp.ones(n) * 1.0
        params = jnp.vstack([mu, sigma])
        omega_logits = jnp.zeros(n)
        return cls.inverse_transform_params(params), omega_logits