from abc import ABC, abstractmethod 
import numpyro
from scipy.stats import lomax
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt



class Trigger(ABC):
    def __init__(self,prior):
        """
        Abstract Trigger class to be extented for Hawkes models. The trigger is assumed to be a pdf and the reproduction rate is coded separately. The required methods to implement are:
        
        - `compute_trigger`: compute the trigger function (pdf)
        
        - `compute_integral`: compute the integral of the trigger function given limits (cdf)
        
        - `get_par_names`: returns a list of the parameter names used in the trigger function
        
        `simulate_trigger` is used only if a user wishes to simulate from the trigger function.

        UNIT CONTRACT: SPATIAL triggers operate entirely in REAL coordinate
        units (the units of the input X/Y columns): compute_trigger receives
        real-unit displacements, compute_integral real-unit rectangle limits,
        simulate_trigger returns a real-unit displacement, and any sampled
        parameters (and their priors) are real-unit quantities. TEMPORAL
        triggers remain in internal time units (data time rescaled to
        [0, 50]); this asymmetry is deliberate and documented -- the temporal
        conversion is a pure relabel, deferred to the conversion layer.
        Custom spatial triggers written against the pre-contract API (internal
        units) will still run but their parameters change meaning.

        Parameters
        ----------
        prior: dict of numpyro distributions
            Used to sample parameters for trigger
        """
        self.prior = prior

    def sample_parameters(self):
        """
        Sample parameters using numpyro
        e.g. return {'beta': numpyro.sample('beta', self.prior['beta'])}
        
        Returns
        -------
        dict of a single sample of parameters
        """
        names = self.get_par_names()
        return {n:numpyro.sample(n,self.prior[n]) for n in names}
    
    @abstractmethod
    def simulate_trigger(self,pars):
        """
        Simulate a point from the trigger function (assuming the trigger is a pdf). Optional. Only necessay for data simulation.
        Parameters
        ----------
        pars: dict
            parameters for the trigger to generate point.
        Returns
        -------
            spatial triggers - np.array [2]
            temporal triggers - float
        """
        pass
    
    @abstractmethod
    def compute_trigger(self, pars, pairs_and_values):
        """
        Compute the trigger function. Computes the trigger function for the [n,n] difference matrix of points.
        Parameters
        ----------
        pars: dict
            results from sample_parameters
        pairs_and_values: tuple
            tuple containing coords and values
        Returns
        -------
        jax numpy matrix [n,n]. Trigger function computed for each entry in the matrix
        """
        coords, values = pairs_and_values
        trigger_vals = jnp.exp(-values / pars['beta']) / pars['beta']
        return coords, trigger_vals
    
    @abstractmethod
    def compute_integral(self,pars,limits):
        """
        Compute the integral of the trigger function from the given limits. For temporal triggers, the integral is computed from 0 to the upper bound. For spatial triggers, the integral is over the rectangle defined by [[x_max,x_min],[y_max,y_min]]
        Parameters
        -----------
        pars: dict
            results from sample_parameters
        limits: jax numpy matrix
            limits of integration with shape
                temporal - [n] compute integal from 0 to limit
                spatial - [2, 2, n] compute integral over rectangle defined by [[x_max,x_min],[y_max,y_min]]
                spatiotemporal - ([n], [2, 2, n]) combination of temporal limits and spatial limits
        Returns
        -------
        jax numpy [n]
        """
        pass

    @abstractmethod
    def get_par_names(self):
        """
        Get list of parameter names. Parameter names may not overlap with any other parameter in the model.
        Excluded names include ['alpha','a_0','b_0','f_xy','v_xy','f_t','v_t','w','mu_xyt','rate_t','z_spatial','z_temporal','rate_xy']. Each parameter named here must have a prior with the same name specified in the model.
        Returns
        -------
        list of names of parameters
        """
        pass

class Temporal_Power_Law(Trigger):

    def __init__(self,prior):
        r"""
        Power Law Temporal trigger. Lomax distribution given by,
    
        $$f(t;\beta,\gamma) = \beta \gamma^\beta (\gamma + t)^{-\beta - 1}$$

        """
        super().__init__(prior)
    
    def simulate_trigger(self,pars):
        return lomax.rvs(pars['beta'])*pars['gamma']

    def compute_trigger(self, pars, pairs_and_values):
        coords, values = pairs_and_values
        trigger_vals = pars['beta']/pars['gamma'] * (1 + values/pars['gamma']) ** (-pars['beta'] - 1)
        return coords, trigger_vals

    def compute_integral(self,pars,dif):
        return 1-(1+dif/pars['gamma'])**(-pars['beta'])

    def get_par_names(self):
        return ['beta','gamma']

#------------------------------------------------    
# Method 2: Event-based Simulation (Hawkes-like process)
def simulate_hawkes_like_events(tpl, params, total_time=1500, baseline_rate=0.1):
    """
    Simulate a sequence of events where each event can trigger more events
    """
    # Convert parameters to scalars if they're arrays
    params_scalar = {
        'beta': float(params['beta']) if hasattr(params['beta'], 'item') else params['beta'],
        'gamma': float(params['gamma']) if hasattr(params['gamma'], 'item') else params['gamma']
    }
    
    events = []
    current_time = 0
    
    # Start with some initial events
    np.random.seed(42)  # For reproducibility
    
    while current_time < total_time:
        # Add baseline inter-arrival time (exponential)
        baseline_wait = np.random.exponential(1/baseline_rate)
        current_time += baseline_wait
        
        if current_time >= total_time:
            break
            
        events.append(current_time)
        
        # Each event can trigger additional events with decreasing probability
        n_triggered = np.random.poisson(0.5)  # Average 0.5 triggered events
        
        for _ in range(n_triggered):
            # Use your temporal power law for triggered event timing
            trigger_delay = tpl.simulate_trigger(params_scalar)
            
            # Cap the delay to reasonable values
            trigger_delay = min(trigger_delay, 100)  # Max 100 days
            
            triggered_time = current_time + trigger_delay
            if triggered_time < total_time:
                events.append(triggered_time)
    
    # Sort events and calculate inter-arrival times
    events = sorted(events)
    inter_arrival_times = np.diff(events)
    
    return inter_arrival_times

# Visualization function
def plot_inter_arrival_distribution(inter_arrival_times, title="Frequency Distribution of Time Difference Between Events"):
    """
    Create a histogram similar to your reference plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create histogram with appropriate bins
    # Most values should be small (clustering) with long tail
    max_time = min(1500, np.percentile(inter_arrival_times, 99))  # Limit to 99th percentile
    bins = np.linspace(0, max_time, 50)
    
    counts, bins, patches = plt.hist(inter_arrival_times, 
                                bins=bins, 
                                alpha=0.7, 
                                color='gray', 
                                edgecolor='black')
    
    plt.xlabel('Time Difference (days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    plt.text(0.7, 0.8, f'Total events: {len(inter_arrival_times)}\n'
                    f'Mean interval: {np.mean(inter_arrival_times):.2f} days\n'
                    f'Median interval: {np.median(inter_arrival_times):.2f} days', 
            transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return counts, bins

#------------------------------------------------    

class Temporal_Exponential(Trigger):
    r"""
    Temporal exponential trigger function given by,

    $$f(t;\beta) = \frac{1}{\beta} e^{-t/\beta}$$
    
    """
    
    def simulate_trigger(self, pars):
        return np.random.exponential(pars['beta'])
    
    def compute_trigger(self, pars, pairs_and_values):
        coords, values = pairs_and_values
        trigger_vals = jnp.exp(-values / pars['beta']) / pars['beta']
        return coords, trigger_vals
    
    def compute_integral(self,pars,dif):
        return 1-jnp.exp(-dif/pars['beta'])
    
    def get_par_names(self):
        return ['beta']


class Spatial_Symmetric_Gaussian(Trigger):
    r"""
    Single parameter symmetric spatial gaussian trigger given by,

    $$\varphi(\mathbf{x};\sigma_x^2) = \frac{1}{2 \pi \sigma_x} exp(-\frac{1}{2\sigma_x^2} \mathbf{x} \cdot \mathbf{x})$$

    UNIT CONTRACT (real-unit spatial trigger): sigmax_2 is the kernel variance
    in SQUARED REAL units -- the units of the input X/Y columns -- and its
    prior must be specified accordingly. The density is per REAL area;
    compute_trigger receives real-unit displacements, compute_integral
    real-unit rectangle limits, and simulate_trigger returns a real-unit
    displacement. The internal <-> real conversion happens at the likelihood
    boundary (likelihood.py), never here. Offspring displacements are
    therefore N(0, sigmax_2 * I) in real coordinates, isotropic regardless of
    the bounding rectangle's shape.
    """

    def simulate_trigger(self, pars):
        return np.random.normal(scale=pars['sigmax_2']**0.5,size=2)
    
    def compute_trigger(self, pars, pairs_and_dxdy):
        coords, dx_vals, dy_vals = pairs_and_dxdy
        S_diff_sq = (dx_vals**2 + dy_vals**2) / pars['sigmax_2']
        trigger_vals = jnp.exp(-0.5 * S_diff_sq) / (2 * jnp.pi * pars['sigmax_2'])
        return coords, trigger_vals
    
    def compute_integral(self,pars,dif):
        gaussianpart1 = 0.5*jax.scipy.special.erf(dif[0,0]/jnp.sqrt(2*pars['sigmax_2']))+\
                    0.5*jax.scipy.special.erf(dif[0,1]/jnp.sqrt(2*pars['sigmax_2']))
        
        gaussianpart2 = 0.5*jax.scipy.special.erf(dif[1,0]/jnp.sqrt(2*pars['sigmax_2']))+\
                    0.5*jax.scipy.special.erf(dif[1,1]/jnp.sqrt(2*pars['sigmax_2']))
        return gaussianpart2*gaussianpart1

    def get_par_names(self):
        return ['sigmax_2']