import numpy as np
import scipy
from scipy.stats import gamma as gamma_dist

### Python implementation of https://arxiv.org/pdf/hep-ex/0001020

atomic_number = {
    'lead': 82,
    'iron': 26,
    'copper': 29,
    'aluminum': 13
}

### From https://pdg.lbl.gov/2025/AtomicNuclearProperties/expert.html
critical_energy = {
    82: 7.43,
    26: 21.0,
    29: 19.0,
    13: 28.0
}

### Eq (4)
def get_beta(alpha, T):
    return (alpha - 1) / T

### Eq (7) -- default yields mean value (A.1.1)
def get_T(y, 
          t_1=0.858):

    return np.log(y) - t_1

### Eq (8) -- default yields mean value (A.1.1)
def get_alpha(y, Z, 
              a_1=0.21, 
              a_2=0.492, 
              a_3=2.38):
    return a_1 + (a_2 + a_3/Z) * np.log(y)

### Eq (11b)
def get_cholesky_covariance_matrix(sigma1, sigma2, rho12):

    sigma_matrix = np.array([
        [sigma1, 0],
        [0, sigma2]
    ])

    rho_matrix = np.array([
            [np.sqrt(1+rho12),  np.sqrt(1-rho12)],
            [np.sqrt(1+rho12), -np.sqrt(1-rho12)]
        ])/np.sqrt(2)

    return sigma_matrix @ rho_matrix


def get_mean_params(y, Z):
    T = get_T(y)
    alpha = get_alpha(y, Z)
    beta = get_beta(alpha, T)

    return {
        'T': T,
        'alpha': alpha,
        'beta': beta
    }

def get_fluc_params(y, Z):

    ### Eq (9)
    def get_sigma(y, s1, s2):
        return 1 / (s1 + s2*np.log(y))

    ### Eq (10)
    def get_rho(y, r1, r2):
        return r1 + r2*np.log(y)

    ### A.1.2
    mean_ln_T         = np.log(get_T(y, t_1=0.812))
    sigma_ln_T        = get_sigma(y, s1=-1.4, s2=1.26)
    mean_ln_alpha     = np.log(get_alpha(y, Z, a_1=0.81, a_2=0.458, a_3=2.26))
    sigma_ln_alpha    = get_sigma(y, s1=-0.58, s2=0.86)
    rho_ln_T_ln_alpha = get_rho(y, r1=0.705, r2=-0.023)

    cov_matrix = get_cholesky_covariance_matrix(sigma_ln_T, sigma_ln_alpha, rho_ln_T_ln_alpha)

    ### Two random variables
    z = np.random.randn(2)

    ### Eq (11a)
    ln_T_i, ln_alpha_i = np.array([mean_ln_T, mean_ln_alpha]) + cov_matrix @ z

    ### Final parameters
    T_i = np.exp(ln_T_i)
    alpha_i = np.exp(ln_alpha_i)
    beta_i = get_beta(alpha_i, T_i)

    return {
        'T': T_i,
        'alpha': alpha_i,
        'beta': beta_i,
        'mean_ln_T': mean_ln_T,
        'sigma_ln_T': sigma_ln_T,
        'mean_ln_alpha': mean_ln_alpha,
        'sigma_ln_alpha': sigma_ln_alpha,
        'rho_ln_T_ln_alpha': rho_ln_T_ln_alpha
    }


def get_longitudinal_profile(alpha, beta):

    def pdf(t):
        ### Eq (2)
        return gamma_dist.pdf(t, alpha, scale=1/beta)

    return pdf



### Eq (24)
def get_R_core(tau, z_1, z_2):
    return z_1 + z_2 * tau

### Eq (25)
def get_R_tail(tau, k_1, k_2, k_3, k_4):
    term1 = np.exp(k_3 * (tau - k_2))
    term2 = np.exp(k_4 * (tau - k_2))
    return k_1 * (term1 + term2)

### Eq (26)
def get_p(tau, p_1, p_2, p_3):
    tau_prime = (p_2 - tau) / p_3
    p = p_1 * np.exp(tau_prime - np.exp(tau_prime))
    if not 0 <= p <= 1:
        raise ValueError(f"p should be in [0,1], got {p} for tau={tau}")
    return p  

### A.1.3
def get_radial_profile_params(tau, E, Z):

    lnE = np.log(E)
    z_1 = 0.0251 + 0.00319*lnE
    z_2 = 0.1162 - 0.000381*Z
    k_1 = 0.659  - 0.00309*Z
    k_2 = 0.645
    k_3 = -2.59
    k_4 = 0.3585 + 0.0421*lnE
    p_1 = 2.632  - 0.00094*Z
    p_2 = 0.401  + 0.00187*Z
    p_3 = 1.313  - 0.0686*lnE

    R_C = get_R_core(tau=tau, z_1=z_1, z_2=z_2)
    R_T = get_R_tail(tau=tau, k_1=k_1, k_2=k_2, k_3=k_3, k_4=k_4)
    p = get_p(tau=tau, p_1=p_1, p_2=p_2, p_3=p_3)

    return R_C, R_T, p


def get_radial_profile(T, E, Z,
                   fluctuate=True, 
                   alpha=None,
                   mean_ln_alpha=None):

    
    def radial_component(r, R):
        num = 2*r*R**2
        den = (r**2 + R**2)**2
        return num/den

    def radial_profile(r, t):

        if fluctuate:
            assert alpha is not None and mean_ln_alpha is not None, \
                "alpha and mean_ln_alpha needed to fluctuate"
            beta = get_beta(alpha, T)
            mean_t = alpha / beta
            ### Eq (34)
            tau = (t / mean_t) * np.exp(mean_ln_alpha) / (np.exp(mean_ln_alpha) - 1)
        else:
            tau = t / T

        ### Get radial profile parameters
        R_core, R_tail, p = get_radial_profile_params(tau, E, Z)

        ### Eq (23)
        return p*radial_component(r, R_core) + (1-p)*radial_component(r, R_tail)

    return radial_profile


class Shower:

    def __init__(self, E, material):
        self.E = E
        self.Z = atomic_number[material] if isinstance(material, str) else material
        self.Ec = critical_energy[self.Z]

        ### compute mean parameters
        self.y = E/self.Ec
        self.mean_params = get_mean_params(self.y, self.Z)

        ### compute mean profile
        self.mean_longitudinal_profile = get_longitudinal_profile(
                                            alpha=self.mean_params['alpha'], 
                                            beta=self.mean_params['beta']
                                        )
        self.mean_radial_profile = get_radial_profile(
                                            T=self.mean_params['T'],
                                            E=self.E, Z=self.Z,
                                            fluctuate=False)

        ### fluctuated params and profiles
        self.fluc_params = None
        self.fluc_longitudinal_profile = None
        self.fluc_radial_profile = None

    def fluctuate_profile(self):

        self.fluc_params = get_fluc_params(self.y, self.Z)
        self.fluc_longitudinal_profile = get_longitudinal_profile(
                                            alpha=self.fluc_params['alpha'], 
                                            beta=self.fluc_params['beta'])

        self.fluc_radial_profile = get_radial_profile(
                                        T=self.fluc_params['T'],
                                        E=self.E, Z=self.Z,
                                        fluctuate=True,
                                        alpha=self.fluc_params['alpha'],
                                        mean_ln_alpha=self.fluc_params['mean_ln_alpha'])