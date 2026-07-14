# general libraries
from io import BytesIO
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import warnings
import pickle
import pkgutil
import math
import matplotlib.dates as mdates
import calendar


# JAX
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax.scipy.special import logsumexp
from numpyro.infer import log_likelihood

import jax
import numpyro

from .utils import aligned_difference_pairs, exp_sq_kernel
from .inference_functions import (spatiotemporal_hawkes_model, spatiotemporal_LGCP_model,
                                  run_mcmc, run_SVI, get_samples)
from .trigger import Temporal_Exponential, Spatial_Symmetric_Gaussian
from .decode_fields import (decode_temporal_field, decode_seasonal_field,
                        decode_spatial_field)
from .likelihood import (seasonal_time_integral, spatial_refinement_masses,
                         background_masses)


def _load_decoder(name):
    # pkgutil.get_data returns None for a missing resource in a zipped/installed
    # package (which would then blow up as a TypeError inside pickle.loads), but
    # raises FileNotFoundError/OSError when running from a source tree. Handle both
    # so a missing artifact always surfaces as the actionable message below.
    try:
        raw = pkgutil.get_data(__name__, f"decoders/{name}")
    except (FileNotFoundError, OSError):
        raw = None
    if raw is None:
        raise FileNotFoundError(
            f"Decoder artifact 'bstpp/decoders/{name}' is missing from the package. "
            "Train it with VAE_Train.py and commit the pickle together with a sidecar "
            f"'{name}.meta.txt' recording kernel, length_scale, var_loc/var_scale, "
            "z_dim/hidden dims, and whether training draws were standardized.")
    return pickle.loads(raw)


def load_Boko_Haram():
    """
    Load Boko Haram dataset
    Returns
    -------
    dict
        events: event dataset from https://ucdp.uu.se/downloads/
        covariates: covariates from PRIO-GRID (https://grid.prio.org/#/)
    """
    events = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/BH_conflicts.csv")))
    cov = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/BH_cov.csv")))
    boundaries = np.array([[3,15.5],[4,16.5]])
    return {"events":events, "covariates":cov,'boundaries':boundaries}


def load_Chicago_Shootings():
    """
    Load Chicago Shootings dataset
    Returns
    -------
    dict
        Shooting report data from:
            https://data.cityofchicago.org/Public-Safety/Chicago-Shootings/fsku-dr7m
        Community Area boundaries from:
            https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6
        Community Area Covariates from:
            https://datahub.cmap.illinois.gov/maps/2a0b0316dc2c4ecfa40a171c635503f8/about
    """
    events_2022 = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/Chicago_2022_xyt.csv")))
    events_2023 = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/Chicago_2023_xyt.csv")))
    cov = gpd.read_file(BytesIO(pkgutil.get_data(__name__, "data/Chicago_cov.zip")))
    boundaries = gpd.read_file(BytesIO(pkgutil.get_data(__name__, "data/Boundaries - Community Areas (current).zip")))
    return {"events_2022":events_2022, "events_2023":events_2023,
            "covariates":cov, "boundaries":boundaries}


def add_month_column(df, t_col='T', origin='2020-01-01'):
    """
    Adds a 'month' column (1-12) to the dataframe based on the time column.
    Assumes 'T' is in days since origin.
    """
    df = df.copy()
    df['month'] = pd.to_datetime(df[t_col], unit='D', origin=origin).dt.month
    return df


def add_month_grid_and_labels(ax, start_date, num_days,label_every_n_months=3):
    # Ensure we are working with pandas Timestamp
    start_date = pd.Timestamp(start_date)
    end_date = start_date + pd.Timedelta(days=int(num_days))


    # Generate ticks at the start of each month
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    xticks = (month_starts - start_date).days  # x values are in "days since start"

    # Show label every n months AND always label the last tick
    xlabels = []
    for i, date in enumerate(month_starts):
        if i % label_every_n_months == 0 or i == len(month_starts) - 1:
            xlabels.append(date.strftime('%b \n %Y'))
        else:
            xlabels.append('')

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xlabels,
        fontsize=8
    )

    #ax.set_xticklabels(xlabels) #, rotation=45
    #ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)


class Point_Process_Model:
    def __init__(self,model,data,A,T,offset_seasonal=0,spatial_cov=None,cov_names=None,
                 cov_grid_size=None,standardize_cov=True,sp_var_mu=2.0,**kwargs):
        """
        Spatiotemporal Point Process Model.
        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        model: str
            one of ['cox_hawkes','lgcp','hawkes'].
        data: str or pd.DataFrame
            either file path or DataFrame containing spatiotemporal data. Required
            columns are 'X', 'Y', 'T'. The seasonal coordinate 'A' (day-of-year) is
            NOT required: it is derived as (T + offset_seasonal) mod S. If an 'A'
            column is supplied it is validated for consistency with 'T' (to within
            1 day) and a ValueError is raised on mismatch, rather than trusted.
        A: np.array [2x2], GeoDataFram
            Spatial region of interest. If np.array first row is the x-range, second row is y-range.
        T: float
            Maximum time in region of interest. Time is assumed to spart at 0.
        offset_seasonal: float
            Day-of-year corresponding to T=0, used to derive the seasonal coordinate
            A = (T_days + offset_seasonal) mod S. Defaults to 0.
        spatial_cov: str,pd.DataFrame,gpd.GeoDataFrame
            Either file path (.csv or .shp), DataFrame, or GeoDataFrame containing spatial covariates.
            Spatial covariates must cover all the points in data.
            If spatial_cov is a csv or pd.DataFrame, the first 2 columns must be 'X', 'Y' and cov_grid_size must be specified.
        cov_names: list
            List of covariate names. Must all be columns in spatial_cov.
        cov_grid_size: list-like
            Spatial covariate grid (width, height).
        standardize_cov: bool
            Standardize covariates
        sp_var_mu: float
            Fixed log-amplitude multiplier applied to the spatial VAE decoder output.
            Calibrate as sp_var_mu = 0.5*log(sigma2_target / E[f^2]_decoder), where
            E[f^2]_decoder is the empirical second moment of decoder outputs under z~N(0,I)
            and sigma2_target is the intended marginal variance of the spatial log-intensity
            field. Note this is an amplitude multiplier, NOT the var_loc used in VAE
            training. A sampled amplitude (and a matching knob for the seasonal field, which
            currently has none) is planned follow-up work.
        priors: dict
            priors for parameters (a_0,w,alpha,beta,sigmax_2). Must be a numpyro distribution.
        """
        if type(data) is str:
            data = pd.read_csv(data)
        self.data = data

        args={}
        args['T']=50
        args['S']=24 #24
        # Spatial grid is 1x1
        args['t_min']=0
        args['x_min']=0
        args['x_max']=1
        args['y_min']=0
        args['y_max']=1
        args['model']=model

        args['offset_seasonal'] = offset_seasonal

        if type(A) is gpd.GeoDataFrame:
            A_ = np.stack((A.bounds.min(axis=0)[['minx','miny']],
                      A.bounds.max(axis=0)[['maxx','maxy']])).T
            #proportion of area of rectangle A_ covered by A. Used for Hawkes integral.
            args['A_area'] = A.area.sum()/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
        else:# A is rectangle specified by np.array
            args['A_area'] = 1
            A_ = A
        args['A_'] = A_

        # create computational grids

        # time grid
        n_t=50
        x_t = jnp.arange(0, args['T'], args['T']/n_t)
        args["n_t"]=n_t
        args["x_t"]=x_t

        # seasonal grid
        n_s=24 #24
        x_a = jnp.arange(0, args['S'], args['S']/n_s)
        args["n_s"]=n_s
        args["x_a"]=x_a
        self.S = 365

        # spatial grid
        n_xy = 25
        args["n_xy"]= n_xy
        cols = np.arange(0, 1, 1/n_xy)
        polygons = []
        for y in cols:
            for x in cols:
                polygons.append(Polygon([(x,y), (x+1/n_xy, y), (x+1/n_xy, y+1/n_xy), (x, y+1/n_xy)]))
        comp_grid = gpd.GeoDataFrame({'geometry':polygons,'comp_grid_id':np.arange(n_xy**2)})
        comp_grid.geometry = comp_grid.geometry.scale(xfact=A_[0,1]-A_[0,0],yfact=A_[1,1]-A_[1,0],
                                                      origin=(0,0)).translate(A_[0,0],A_[1,0])

        if type(A) is gpd.GeoDataFrame:
            # find grid cells overlapping with A
            comp_grid.crs = A.crs
            args['spatial_grid_cells'] = np.unique(comp_grid.sjoin(A, how='inner')['comp_grid_id'])
            self.A = A
        else:
            self.A = comp_grid
            args['spatial_grid_cells'] = np.arange(25**2)

        self.comp_grid = comp_grid
        self.T = T

        # Seasonal index of each temporal grid cell midpoint (diagonal a = sigma(t)):
        # internal cell i covers real days [i, i+1) * (T / n_t); map its midpoint through
        # day-of-year (mod self.S) onto the internal seasonal grid [0, args['S']).
        # NOTE: the likelihood no longer integrates via this midpoint index (it uses the
        # exact overlap matrix season_overlap below); season_idx_of_t is retained for
        # diagnostics (rate_time) and scripts/recover_test.py's intercept combination.
        t_mid_days = (np.arange(n_t) + 0.5) * (T / n_t)
        a_mid = ((t_mid_days + offset_seasonal) % self.S) / self.S * args['S']
        args['season_idx_of_t'] = np.searchsorted(np.asarray(x_a), a_mid, side='right') - 1

        # Exact seasonal overlap matrix W (n_t x n_s), in INTERNAL time units:
        #   W[i, k] = measure{ t in temporal cell i : sigma(t) in seasonal cell k }.
        # f_t (n_t cells) and f_a (n_s cells) are piecewise constant but the cell widths
        # differ (T/n_t days vs self.S/n_s days), so temporal cells straddle seasonal
        # boundaries and the time integral of exp(a_0 + f_t + f_a[sigma(t)]) has a closed
        # form: contract exp(f_a) against W. Temporal cell i covers real days
        # [i, i+1) * (T / n_t); sigma(d) = (d + offset_seasonal) mod self.S; seasonal cell
        # k = floor(sigma / self.S * n_s). In the shifted coordinate s = d + offset_seasonal
        # every split point we need -- seasonal-cell edges AND the year wrap -- is an integer
        # multiple of h_day = self.S / n_s (since self.S = n_s * h_day), and on the interval
        # [p*h_day, (p+1)*h_day) the seasonal cell is p mod n_s. Day-lengths convert to
        # internal units via * (args['T'] / T).
        h_day = self.S / n_s
        dt_day = T / n_t
        W = np.zeros((n_t, n_s))
        for i in range(n_t):
            s = i * dt_day + offset_seasonal
            s_end = (i + 1) * dt_day + offset_seasonal
            while s < s_end - 1e-9:
                p = int(np.floor(s / h_day + 1e-9))
                seg_end = min((p + 1) * h_day, s_end)
                W[i, p % n_s] += seg_end - s
                s = seg_end
        W *= args['T'] / T
        assert np.allclose(W.sum(axis=1), args['T'] / n_t, rtol=1e-6), \
            "season_overlap rows must sum to the internal temporal cell width args['T']/n_t"
        args['season_overlap'] = jnp.asarray(W)

        args,points = self._scale_xyt(data,args,comp_grid)
        self.points = points

        if args['model'] in ['lgcp','cox_hawkes']:
            args["gp_kernel"]=exp_sq_kernel

            # temporal VAE training arguments
            args["hidden_dim_temporal"]= 35
            args["z_dim_temporal"]= 11

            # seasonal VAE training arguments
            args["hidden_dim1_seasonal"]= 24 #75 #35
            args["hidden_dim2_seasonal"]= 12 #50 #30
            args["z_dim_seasonal"]= 8 #20 #50 #10 11

            # spatial VAE training arguments
            args["hidden_dim1_spatial"]= 75
            args["hidden_dim2_spatial"]= 50
            args["z_dim_spatial"]=20

            decoder_params = _load_decoder("decoder_1d_T50_fixed_ls")
            args["decoder_params_temporal"] = decoder_params

            # load decoder for seasonal
            decoder_params = _load_decoder("decoder_1d_T24_circ_small_l8")
            args["decoder_params_seasonal"] = decoder_params

            #Load 2d spatial trained decoder
            decoder_params = _load_decoder("2d_decoder_15_5_large.pkl")
            args["decoder_params_spatial"] = decoder_params

        if spatial_cov is not None:
            #convert input into geopandas dataframe.
            if type(spatial_cov) is str:
                if spatial_cov[-4:] == '.zip' or spatial_cov[-4:] == '.shp':
                    spatial_cov = gpd.read_file(spatial_cov)
                else:
                    spatial_cov = pd.read_csv(spatial_cov)
            # exact-type check is intentional: GeoDataFrame subclasses DataFrame and must NOT
            # take this branch (it already carries geometry); isinstance would misroute it.
            if type(spatial_cov) is pd.DataFrame:
                polygons = []
                for i in spatial_cov.index:
                    polygons.append(Polygon([(spatial_cov.loc[i,'X']-cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']-cov_grid_size[1]/2),
                                             (spatial_cov.loc[i,'X']+cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']-cov_grid_size[1]/2),
                                             (spatial_cov.loc[i,'X']+cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']+cov_grid_size[1]/2),
                                             (spatial_cov.loc[i,'X']-cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']+cov_grid_size[1]/2)]))
                spatial_cov = gpd.GeoDataFrame(data=spatial_cov,geometry=polygons)
                spatial_cov.crs = self.A.crs

            spatial_cov['cov_ind'] = np.arange(len(spatial_cov))
            #find covariate cell index for each point
            self.points.crs = spatial_cov.crs
            args['cov_ind'] = self.points.sjoin(spatial_cov).sort_values(by='point_id')['cov_ind'].values
            if len(args['cov_ind']) != len(self.points):
                raise Exception("Spatial covariates are not defined for all data points!")

            args['num_cov'] = len(cov_names)
            self.cov_names = cov_names
            self.spatial_cov = spatial_cov

            X_s = spatial_cov[self.cov_names].values
            if X_s.ndim == 1:
                X_s = X_s[:, None]
            # standardize covariates
            if standardize_cov:
                args['spatial_cov'] = (X_s-X_s.mean(axis=0))/(X_s.var(axis=0)**0.5)
            else:
                args['spatial_cov'] = X_s

            #Create Computational Grid GeoDataFrame
            if args['model'] in ['lgcp','cox_hawkes']:
                self.comp_grid.crs = spatial_cov.crs
                #find covariate cell intersection with computational grid cells area
                intersect = gpd.overlay(self.comp_grid, spatial_cov, how='intersection', keep_geom_type=True)
                intersect['area'] = intersect.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
                intersect = intersect[intersect['area']>1e-10]
                args['int_df'] = intersect
                #find cells on the computational grid that are in the domain
                args['spatial_grid_cells'] = np.unique(self.comp_grid.sjoin(spatial_cov, how='inner')['comp_grid_id'])
            else:
                args['cov_area'] = (spatial_cov.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))).values

        # Integration arrays for the pure spatial_refinement_integral atom
        # (eq. 24): plain NumPy arrays only, derived once here, so no pandas /
        # GeoPandas object is ever read inside traced likelihood code. The
        # no-covariate grid is the special case of the common refinement with
        # uniform cell areas 1/n_xy^2.
        if args['model'] in ['lgcp', 'cox_hawkes']:
            if 'int_df' in args:
                args['integration_field_indices'] = args['int_df']['comp_grid_id'].values
                args['integration_cov_indices'] = args['int_df']['cov_ind'].values
                args['integration_areas'] = args['int_df']['area'].values
            else:
                args['integration_field_indices'] = args['spatial_grid_cells']
                args['integration_cov_indices'] = None
                args['integration_areas'] = np.full(
                    len(args['spatial_grid_cells']), 1.0 / args['n_xy'] ** 2,
                    dtype=np.float32)

        #Set up parameter priors
        default_priors = {}
        if 'num_cov' in args:
            default_priors["w"] = dist.Normal(jnp.zeros(args['num_cov']),jnp.ones(args["num_cov"]))
        args['sp_var_mu'] = float(sp_var_mu)
        for par, prior in kwargs.items():
            if isinstance(prior,dist.Distribution):
                default_priors[par] = prior
            else:
                raise Exception(f"Unknown argument {par}. Prior distributions must be instances of numpyro Distribution.")
        args['priors'] = default_priors
        self.args = args

    def __str__(self):
        return "Point Process Model"

    def load_rslts(self,file_name):
        """
        Load previously computed results
        Parameters
        ----------
        file_name: string
            File where pickled results are held
        """
        with open(file_name, 'rb') as f:
            output = pickle.load(f)
        if 'svi_results' in output:
            self.svi_results = output['svi_results']
        if 'mcmc' in output:
            self.mcmc = output['mcmc']
        self.samples = output['samples']

    def save_rslts(self,file_name):
        """
        Save previously computed results
        Parameters
        ----------
        file_name: string
            File where to save results
        """
        output = dict()
        if 'svi_results' in dir(self):
            output['svi_results'] = self.svi_results
        if 'mcmc' in dir(self):
            output['mcmc'] = self.mcmc
        output['samples'] = self.samples
        with open(file_name, 'wb') as f:
            output = pickle.dump(output,f)


    def run_svi(self,num_steps,lr,num_samples=1000,resume=False,plot_loss=True,**kwargs):
        """
        Perform Stochastic Variational Inference on the model.
        Parameters
        ----------
        num_samples: int, default=1000
            Number of samples to generate after SVI.
        resume: bool, default=False
            Pick up where last SVI run was left off. Can only be true if model has previous run_svi call.
        lr: float, default=0.001
            learning rate for SVI
        num_steps: int, default=10000
            Number of interations for SVI to run.
        plot_loss: bool

        auto_guide: numpyro AutoGuide, default=AutoMultivariateNormal
            See numpyro AutoGuides for details.
        init_strategy: function, default=init_to_median
            See numpyro init strategy documentation
        """
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        self.args["num_samples"] = num_samples
        # Model-aware site list: LGCP has no 'Itot_excite' site. Predictive drops
        # missing names silently, which would also mask a typo'd site, so only
        # request 'Itot_excite' for the models that actually define it.
        sites = list(self.get_params().keys()) + ['loglik', 'Itot_txy']
        if self.args['model'] in ('hawkes', 'cox_hawkes'):
            sites += ['Itot_excite']
        if resume:
            kwargs['num_steps'] = num_steps
            kwargs['lr'] = lr
            optimizer = numpyro.optim.Adam(
                jax.example_libraries.optimizers.inverse_time_decay(kwargs['lr'],kwargs['num_steps'],4)
            )
            self.svi.optim = optimizer
            self.svi_results = self.svi.run(rng_key, kwargs['num_steps'], self.args,
                                            init_state=self.svi_results.state)
            self.samples = get_samples(rng_key,self.model,self.svi.guide,self.svi_results,self.args,sites)
        else:
            self.svi,self.svi_results,self.samples=run_SVI(rng_key, self.model, self.args, num_steps, lr, sites, **kwargs)
        loss = np.asarray(self.svi_results.losses)
        plt.plot(np.arange(int(.01*len(loss)),len(loss)),loss[int(.01*len(loss)):])
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()


    def run_mcmc(self,batch_size=1,num_warmup=500,num_samples=1000,
                 num_chains=1,thinning=1):
        """
        Run MCMC posterior sampling on model.

        Parameters
        ----------
        batch_size: int
            See numpyro documentation for description
        num_warmup: int
        num_samples: int
        num_chains: int
        thinning: int
        """
        self.args["batch_size"]= batch_size
        self.args["num_warmup"]= num_warmup
        self.args["num_samples"] = num_samples
        self.args["num_chains"] = num_chains
        self.args["thinning"] = thinning
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)

        self.mcmc = run_mcmc(rng_key_post, self.model, self.args)
        self.samples=self.mcmc.get_samples()


    def _scale_xyt(self,data,args,comp_grid):
        #scale temporal events
        t_events_total=data['T'].values/self.T*args["n_t"]
        args["t_events"]=t_events_total
        args['indices_t']=np.searchsorted(args['x_t'], t_events_total, side='right')-1

        a_days = (data['T'].values + args['offset_seasonal']) % self.S
        if 'A' in data.columns:
            supplied = np.asarray(data['A'].values, dtype=float) % self.S
            if not np.allclose(supplied, a_days, atol=1.0):
                raise ValueError(
                    "'A' column is inconsistent with 'T' + offset_seasonal "
                    "(tolerance 1 day). Drop 'A' or fix offset_seasonal.")
        a_events_total = a_days/self.S*args["n_s"]
        args["a_events"]=a_events_total
        args['indices_a']=np.searchsorted(args['x_a'], a_events_total, side='right')-1

        #scale spatial events
        x_range = args['A_'][0]
        x_events_total=(data['X']-x_range[0]).to_numpy(copy=True)
        x_events_total/=(x_range[1]-x_range[0])
        y_range = args['A_'][1]
        y_events_total=(data['Y']-y_range[0]).to_numpy(copy=True)
        y_events_total/=(y_range[1]-y_range[0])

        xy_events_total=np.array((x_events_total,y_events_total)).transpose()

        geometry = gpd.points_from_xy(data.X, data.Y,crs=comp_grid.crs)
        points = gpd.GeoDataFrame(data=data,geometry=geometry)
        points['point_id'] = np.arange(len(data))

        #find grid cells where points are located
        args['indices_xy'] = points.sjoin(comp_grid).sort_values(by='point_id')['comp_grid_id'].values

        if len(args['indices_xy']) != len(points):
            raise Exception("Computational grid does not encompass all data points!")

        args["xy_events"]=xy_events_total.transpose()
        return args,points

    def log_expected_likelihood(self, data):
        if type(data) is str:
            data = pd.read_csv(data)
        if 'day' in data.columns:
            data = data.drop(columns=['day'])
        for col in ['X', 'Y', 'T']:
            if not np.issubdtype(data[col].dtype, np.number):
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=['X', 'Y', 'T'])
        if len(data) == 0:
            raise ValueError("No valid data points after cleaning")

        # Only pass the minimal required arguments for likelihood
        test_args, points = self._scale_xyt(data, self.args.copy(), self.comp_grid)

        # Held-out events must lie within the training time horizon [0, T]: the
        # excitation compensator jnp.minimum(T - t, window) is only defined there
        # (times > T make it negative and silently corrupt the integral).
        if np.any(test_args['t_events'] < 0) or np.any(test_args['t_events'] > self.args['T']):
            raise ValueError(
                "Held-out events must lie within the training time horizon "
                f"[0, {self.args['T']}] (in internal rescaled time units); the "
                "excitation compensator is only defined on that interval.")

        # Rebuild the excitation difference-pairs on the TEST events. test_args
        # was copied from self.args, whose coords/t_vals/x_vals/y_vals are the
        # TRAINING pairs; reusing them pairs training-event differences with
        # test-event indices (segment_sum silently drops out-of-range indices).
        # LGCP uses no pairs, so it needs no rebuild.
        if self.args['model'] in ('hawkes', 'cox_hawkes'):
            coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
                test_args['t_events'],
                test_args['xy_events'][0],
                test_args['xy_events'][1],
                self.args['window'],
                spatial_window=self.args.get('spatial_window'),
            )
            test_args['coords'] = coords
            test_args['t_vals'] = t_vals
            test_args['x_vals'] = x_vals
            test_args['y_vals'] = y_vals

        if 'cov_ind' in self.args:
            test_args['cov_ind'] = points.sjoin(self.spatial_cov).sort_values(by='point_id')['cov_ind'].values

        # Remove training-specific keys if present
        for k in ['batch_size', 'num_samples', 'num_warmup', 'num_chains', 'thinning']:
            test_args.pop(k, None)

        post_loglik = log_likelihood(self.model, self.samples, test_args)["loglik_factor"]
        exp_log_density = logsumexp(post_loglik, axis=0) - jnp.log(jnp.shape(post_loglik)[0])
        return exp_log_density.sum().item()

    def expected_AIC(self):
        r"""
        Calculate the expected AIC over the posterior distribution.
        For $k = $ number of model parameters, expected AIC is defined as,
        $$E_{\theta|X}[AIC] = \frac{-2}{S}\sum_{s=1}{S}{log(p(X|\theta^s))} + 2k$$
        """
        k = sum(self.get_params().values())
        return -2*self.samples['loglik'].mean().item() + 2*k


    def cov_weight_post_summary(self,trace=False):
        """
        Plot and summarize posteriors of weights and bias.
        Returns
        -------
        pd.DataFrame
            summary of weights and bias
        trace: bool
            plot trace or histogram of posteriors
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if 'spatial_cov' not in self.args:
            raise Exception("Spatial covariates were not included in the model.")

        n = self.samples['w'].shape[1] + 1  # number of covariates + intercept
        c = 2                       # always 2 columns
        r = math.ceil(n / c)        # as many rows as needed
        fig, ax = plt.subplots(r, c, figsize=(12, 3 * r), sharex=False)
        fig.suptitle('Covariate Weights', fontsize=16)
        w_samples = self.samples['w']
        if w_samples.ndim == 1:
            w_samples = w_samples[:, None]

        # Flatten ax for easy indexing
        ax = ax.flatten()

        for i in range(w_samples.shape[1]):
            if trace:
                ax[i].plot(w_samples[:, i])
                ax[i].set_ylabel(self.cov_names[i])
            else:
                ax[i].hist(w_samples[:, i])
                ax[i].set_xlabel(self.cov_names[i])
        # Plot the intercept
        if trace:
            ax[w_samples.shape[1]].plot(self.samples['a_0'])
            ax[w_samples.shape[1]].set_ylabel("$a_0$")
        else:
            ax[w_samples.shape[1]].hist(self.samples['a_0'])
            ax[w_samples.shape[1]].set_xlabel("$a_0$")
        # Hide unused axes
        for j in range(n, len(ax)):
            ax[j].axis('off')
   

        w_samps = np.concatenate((w_samples,self.samples['a_0'].reshape(-1,1)),axis=1)
        mean = w_samps.mean(axis=0)
        std = w_samps.var(axis=0)**0.5
        p = (w_samps>0).mean(axis=0)
        quantiles = np.quantile(w_samps,[0.025,0.975],axis=0)
        w_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'P(w>0)':p,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=self.cov_names+['a_0'])

        ##### Plot mean and 90% CI of weights #####
        # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        # x = range(len(w_summary))

        # # Extract values
        # means = w_summary['Post Mean']
        # lower = w_summary['[0.025']
        # upper = w_summary['0.975]']
        # errors = [means - lower, upper - means]

        # ax.errorbar(
        #     x,
        #     means,
        #     yerr=errors,
        #     fmt='o',
        #     capsize=3,
        #     color='#990000',
        #     ecolor='#011F5B',
        #     label='90% CI'
        # )

        # # Horizontal zero line
        # ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # # Labeling
        # ax.set_xticks(x)

        # wrapped_labels = [label.replace('_', '_\n') for label in w_summary.index]
        # ax.set_xticklabels(wrapped_labels)

        # ax.set_xlabel('Covariate')
        # ax.set_ylabel('Weight Value')
        # ax.yaxis.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=0.7)

        # ax.legend()
        # plt.tight_layout()
        # plt.show()

        ##### Plot mean and 90% CI of weights #####
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        x = range(len(w_summary))

        # Extract values
        means = w_summary['Post Mean']
        lower = w_summary['[0.025']
        upper = w_summary['0.975]']
        errors = [means - lower, upper - means]

        ax.errorbar(
            x,
            means,
            yerr=errors,
            fmt='o',
            capsize=3,
            color='#990000',
            ecolor='#011F5B',
            label='90% CI'
        )

        # Horizontal zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # --- NEW: vertical separators (dashed blue) ---
        # Pairs to separate (left, right)
        pairs = [
            ('edu_hd_avg', 'ndvi_mean_4yr'),  # 'edu_hd_avg' handled below (typo -> 'edu_hs_avg')
            ('ndvi_mean_4yr', 'RLD'),
            ('GW', 'vac_area'),
            ('landcare_area', 'alloc_avg_d_cnt'),
            ('unique_device_ratio_aw', 'reporting_rate'),
            ('reporting_rate','betweenness_avg_w')
        ]

        # Map names to indices in your plotted order
        name_to_idx = {name: i for i, name in enumerate(w_summary.index)}

        # Normalize the possible typo
        def _fix(name):
            return 'edu_hs_avg' if name == 'edu_hd_avg' else name

        # Compute x-positions (midpoints) and draw lines
        for a, b in pairs:
            a, b = _fix(a), _fix(b)
            if a in name_to_idx and b in name_to_idx:
                mid = 0.5 * (name_to_idx[a] + name_to_idx[b])
                ax.axvline(mid, color='blue', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
            else:
                # If a name isn't present, skip silently; optionally print/log a warning
                pass

        # Labeling
        ax.set_xticks(list(x))
        wrapped_labels = [label.replace('_', '_\n') for label in w_summary.index]
        ax.set_xticklabels(wrapped_labels,fontsize = 7)

        ax.set_xlabel('Covariate')
        ax.set_ylabel('Weight Value')
        ax.yaxis.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=0.7)

        # Make sure end separators (if any) are visible
        ax.set_xlim(-0.5, len(w_summary) - 0.5)

        ax.legend()
        plt.tight_layout()
        plt.show()


        return w_summary


       

    def plot_temporal(self, rescale=True, start_date=None):
        """
        Plot mean posterior temporal gaussian process.

        Parameters
        ----------
        rescale: bool
            Scale posteriors to original dimensions of the data.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: temporal background in constant.")

        b_scale = 1.
        if rescale:
            b_scale = 50/self.T
        # x_t = jnp.arange(0, self.args['T'], self.args['T']/self.args["n_t"])/b_scale
        x_t = jnp.linspace(0, self.args["T"], self.args["n_t"] + 1)[:-1] / b_scale
        f_t_post=self.samples["f_t"]
        f_t_hpdi = hpdi(self.samples["f_t"])
        f_t_post_mean=jnp.mean(f_t_post, axis=0)

        fig,ax=plt.subplots(1,1,figsize=(15,5)) #(8,5)


        # Plot the temporal intensity
        event_time_height = np.ones(len(self.args['t_events']))*f_t_hpdi.min()
        ax.plot(self.args['t_events']/b_scale, event_time_height,'+',color="red",
                alpha=.15, label="observed times")
        ax.set_ylabel('$f_t$')
        ax.set_xlabel('t')

        ## adjust grid
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.6, alpha=0.3)

        ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
        ax.fill_between(x_t, f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")

        # Set labels
        ax.set_ylabel('$f_t$')
        if start_date is not None:
            ax.set_xlabel("Date")
            total_days = x_t.max()
            #add_month_grid_and_labels(ax, start_date, total_days)
            add_month_grid_and_labels(ax, start_date, total_days, label_every_n_months=3)
            ##########################################################
        else:
            ax.set_xlabel('t')
            ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)

        # ax.set_xmargin(0)          # same idea as margins(x=0), but explicit for x
        ax.set_xlim(float(x_t.min()), float(x_t.max()))

        ax.legend()
        plt.tight_layout()

    #------------------------------------------------ 
    def plot_temporal_components(
    self,
    rescale=True,
    start_date=None,
    show_gp=True,
    show_hist=False,
    ax=None
    ):
        """
        Plot temporal components:
        - GP mean + CI (left y-axis)
        - Optional event-time histogram (right y-axis)

        You can draw GP only, histogram only, or both stacked together.

        Parameters
        ----------
        rescale : bool
            Scale posteriors to original dimensions of the data.
        start_date : datetime/date/str or None
            If provided, x-axis will be labeled as dates.
        show_gp : bool
            Whether to draw GP mean and CI.
        show_hist : bool
            Whether to draw event-time histogram.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes', 'lgcp']:
            raise Exception("Nothing to plot: temporal background is constant.")

        if not show_gp and not show_hist:
            raise ValueError("At least one of show_gp or show_hist must be True.")

        b_scale = 1.
        if rescale:
            b_scale = 50 / self.T

        x_t = jnp.linspace(
            0, self.args["T"], self.args["n_t"] + 1
        )[:-1] / b_scale

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax2 = None

        # =========================
        # GP posterior (left axis)
        # =========================
        if show_gp:
            f_t_post = self.samples["f_t"]
            f_t_hpdi = hpdi(f_t_post)
            f_t_post_mean = jnp.mean(f_t_post, axis=0)

            event_time_height = np.ones(len(self.args['t_events'])) * f_t_hpdi.min()
            ax.plot(
                self.args['t_events'] / b_scale,
                event_time_height,
                '+',
                color="red",
                alpha=.15,
                label="observed times"
            )

            ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
            ax.fill_between(
                x_t,
                f_t_hpdi[0],
                f_t_hpdi[1],
                alpha=0.4,
                color="palegoldenrod",
                label="90%CI rate"
            )

            ax.set_ylabel('$f_t$')
            ax.grid(True, which='both', axis='both',
                    linestyle=':', linewidth=0.6, alpha=0.3)

        # =========================
        # Histogram (right axis)
        # =========================
        if show_hist:
            ax2 = ax.twinx()

            t_events = np.asarray(self.args["t_events"]) / b_scale
            bins = np.linspace(
                float(x_t.min()),
                float(x_t.max()),
                int(self.args["n_t"]) + 1
            )

            ax2.hist(
                t_events,
                bins=bins,
                alpha=0.25,
                edgecolor="none",
                label="event frequency"
            )

            ax2.set_ylabel("Frequency")
            ax2.grid(False)

        # =========================
        # X-axis formatting
        # =========================
        if start_date is not None:
            ax.set_xlabel("Date")
            total_days = x_t.max()
            add_month_grid_and_labels(
                ax, start_date, total_days, label_every_n_months=3
            )
        else:
            ax.set_xlabel('t')

        ax.set_xlim(float(x_t.min()), float(x_t.max()))
        if ax2 is not None:
            ax2.set_xlim(float(x_t.min()), float(x_t.max()))

        # =========================
        # Legend handling
        # =========================
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2

        if handles:
            ax.legend(handles, labels, loc="best")

        plt.tight_layout()
#------------------------------------------------    
        

    def plot_seasonal(self,rescale=True,ref_year=2021):
        """
        Plot mean posterior seasonal gaussian process.

        Parameters
        ----------
        rescale: bool
            Scale posteriors to original dimensions of the data.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: seasonal background in constant.")

        offset = self.args['offset_seasonal']

        # Create month start positions (in days since Jan 1)
        month_days = np.cumsum([0] + [calendar.monthrange(ref_year, m)[1] for m in range(1, 12)])
        month_names = list(calendar.month_abbr)[1:]  # ['January', ..., 'December']

        # Apply offset and wrap around using modulo
        month_days_offset = (month_days + offset) % self.S

        # Sort for plotting (modulo wrap-around may disorder them)
        sorted_idx = np.argsort(month_days_offset)
        xticks = month_days_offset[sorted_idx]
        xlabels = [month_names[i] for i in sorted_idx]

        b_scale = 1.
        if rescale:
            b_scale = self.args["n_s"]/self.S
        x_a = jnp.arange(0, self.args["S"], self.args["S"]/self.args["n_s"]) / b_scale
        f_a_post=self.samples["f_a"]
        f_a_hpdi = hpdi(self.samples["f_a"])
        f_a_post_mean=jnp.mean(f_a_post, axis=0)

        fig,ax=plt.subplots(1,1,figsize=(9,5))
        event_time_height = np.ones(len(self.args['a_events']))*f_a_hpdi.min()
        ax.plot(self.args['a_events']/b_scale, event_time_height,'+',color="red",
                alpha=.15, label="observed times")
        ax.set_ylabel('$f_a$')
        ax.set_xlabel('Date')
        ax.set_xticks(xticks)
        #ax.set_xticklabels(xlabels, rotation=45)
        ax.set_xticklabels(
        xlabels,
        fontsize=8
        )
        #ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.6, alpha=0.3)

        ax.plot(x_a, f_a_post_mean, label="mean estimated $f_a$")
        ax.fill_between(x_a, f_a_hpdi[0], f_a_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
        ax.legend(loc='upper right')

#------------------------------------------------    
    def plot_seasonal_components(
        self,
        rescale=True,
        ref_year=2021,
        show_gp=True,
        show_hist=False,
        ax=None
    ):
        """
        Plot seasonal components:
        - GP mean + CI (left y-axis)
        - Optional event-time histogram (right y-axis)

        You can draw GP only, histogram only, or both stacked together.

        Parameters
        ----------
        rescale : bool
            Scale posteriors to original dimensions of the data.
        ref_year : int
            Reference year for month calculations.
        show_gp : bool
            Whether to draw GP mean and CI.
        show_hist : bool
            Whether to draw event-time histogram.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: seasonal background is constant.")

        if not show_gp and not show_hist:
            raise ValueError("At least one of show_gp or show_hist must be True.")

        offset = self.args['offset_seasonal']

        # Create month start positions (in days since Jan 1)
        month_days = np.cumsum([0] + [calendar.monthrange(ref_year, m)[1] for m in range(1, 12)])
        month_names = list(calendar.month_abbr)[1:]  # ['Jan', ..., 'Dec']

        # Apply offset and wrap around using modulo
        month_days_offset = (month_days + offset) % self.S

        # Sort for plotting (modulo wrap-around may disorder them)
        sorted_idx = np.argsort(month_days_offset)
        xticks = month_days_offset[sorted_idx]
        xlabels = [month_names[i] for i in sorted_idx]

        b_scale = 1.
        if rescale:
            b_scale = self.args["n_s"]/self.S
        x_a = jnp.arange(0, self.args["S"], self.args["S"]/self.args["n_s"]) / b_scale
        f_a_post = self.samples["f_a"]
        f_a_hpdi = hpdi(self.samples["f_a"])
        f_a_post_mean = jnp.mean(f_a_post, axis=0)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax2 = None

        # =========================
        # GP posterior (left axis)
        # =========================
        if show_gp:
            event_time_height = np.ones(len(self.args['a_events'])) * f_a_hpdi.min()
            ax.plot(
                self.args['a_events']/b_scale,
                event_time_height,
                '+',
                color="red",
                alpha=.15,
                label="observed times"
            )

            ax.plot(x_a, f_a_post_mean, label="mean estimated $f_a$")
            ax.fill_between(
                x_a,
                f_a_hpdi[0],
                f_a_hpdi[1],
                alpha=0.4,
                color="palegoldenrod",
                label="90%CI rate"
            )

            ax.set_ylabel('$f_a$')
            ax.grid(True, which='both', axis='both',
                    linestyle=':', linewidth=0.6, alpha=0.3)

        # =========================
        # Histogram (right axis)
        # =========================
        if show_hist:
            ax2 = ax.twinx()

            a_events = np.asarray(self.args["a_events"]) / b_scale
            bins = np.linspace(
                float(x_a.min()),
                float(x_a.max()),
                int(self.args["n_s"]) + 1
            )

            ax2.hist(
                a_events,
                bins=bins,
                alpha=0.25,
                edgecolor="none",
                color="#E87A90",
                label="event frequency"
            )

            ax2.set_ylabel("Frequency")
            ax2.grid(False)

        # =========================
        # X-axis formatting
        # =========================
        ax.set_xlabel('Date')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)

        # Remove gaps on left and right edges
        ax.set_xlim(float(x_a.min()), float(x_a.max()))
        if ax2 is not None:
            ax2.set_xlim(float(x_a.min()), float(x_a.max()))

        # =========================
        # Legend handling
        # =========================
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2

        if handles:
            ax.legend(handles, labels, loc="best")

        plt.tight_layout()
#------------------------------------------------    
    def plot_temp_date(self, start_date, rescale=True):
        """
        Plot mean posterior temporal gaussian process with dates on x-axis.

        Parameters
        ----------
        start_date: str
            The starting date in 'YYYY-MM-DD' format. This should match the start date
            of your original dataset.
        rescale: bool
            Scale posteriors to original dimensions of the data.
        """
        if start_date == '2023-01-01':
            warnings.warn("Using default start date. Please specify the correct start date for your dataset.")

        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: temporal background in constant.")

        # Convert start_date to datetime
        start_date = pd.to_datetime(start_date)

        # Time scaling
        b_scale = 1.
        if rescale:
            b_scale = 50/self.T

        # Create time grid and convert to dates
        x_t = jnp.arange(0, self.args['T'], self.args['T']/self.args["n_t"])/b_scale
        dates_t = [start_date + pd.Timedelta(days=float(t)) for t in x_t]

        # Get posterior samples and calculate statistics
        f_t_post = self.samples["f_t"]
        f_t_hpdi = hpdi(self.samples["f_t"])
        f_t_post_mean = jnp.mean(f_t_post, axis=0)

        # Convert event times to dates
        event_dates = [start_date + pd.Timedelta(days=float(t)) for t in self.args['t_events']/b_scale]

        # Create plot
        fig, ax = plt.subplots(1,1,figsize=(8,5))

        # Plot event times
        event_time_height = np.ones(len(event_dates))*f_t_hpdi.min()
        ax.plot(event_dates, event_time_height, '+', color="red",
                alpha=.15, label="observed times")

        # Plot posterior mean and confidence interval
        ax.plot(dates_t, f_t_post_mean, label="mean estimated $f_t$")
        ax.fill_between(dates_t, f_t_hpdi[0], f_t_hpdi[1], alpha=0.4,
                        color="palegoldenrod", label="90%CI rate")

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add labels and legend
        ax.set_ylabel('$f_t$')
        ax.set_xlabel('Date')
        ax.legend()

        # Adjust layout to prevent label cutoff
        #plt.tight_layout()

        #return fig

    def plot_spatial(self,include_cov=False, **kwargs):
        """
        Plot mean posterior spatial intensity (ignoring self-excitation) with/without covariates

        Parameters
        ----------
        include_cov: bool
            Include effects of spatial covariates.
        kwargs: dict
            Plotting parameters for geopandas plot.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp'] and not include_cov:
            raise Exception("Nothing to plot: spatial background is constant")
        if include_cov and 'spatial_cov' not in self.args:
            raise Exception("No spatial covariates are in the model and include_cov was set to True")

        if 'alpha' not in kwargs:
            kwargs['alpha'] = .1

        if self.args['model'] in ['cox_hawkes','lgcp'] and include_cov:
            self._plot_cov_comp_grid(**kwargs)
            #self._plot_cov_comp_grid(ax_list=ax_list, **kwargs)
        elif include_cov:
            self._plot_cov(**kwargs)
        else:
            self._plot_grid(**kwargs)
            #self._plot_grid(ax_list=ax_list, **kwargs)

    def _plot_grid(self,**kwargs):
        """
        Plot spatial for computational grid only
        """

        f_xy_post = self.samples["f_xy"]
        f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
        self.comp_grid['post_mean'] = f_xy_post_mean
        intersect = gpd.overlay(self.comp_grid, self.A[['geometry']], how='intersection',keep_geom_type=True)
        fig, ax = plt.subplots(1,3, figsize=(10, 5),gridspec_kw={'width_ratios': [10,10,1]})
        intersect.plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $f_s$')
        ax[2].set_axis_off()
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        intersect.plot(column='post_mean',ax=ax[1],legend=True,cax=cbar_ax)
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s$ With Events')
        return fig

    def _plot_cov_comp_grid(self,**kwargs):
        """
        Plot spatial for computational grid and spatial covariates.
        """
        post_samples = (self.samples['b_0'][:,self.args['int_df']['cov_ind'].values] +
                        self.samples["f_xy"][:,self.args['int_df']['comp_grid_id'].values])
        self.args['int_df']['post_mean'] = post_samples.mean(axis=0)

        fig, ax = plt.subplots(1,3, figsize=(10, 5),gridspec_kw={'width_ratios': [10,10,1]})
        self.args['int_df'].plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $f_s + X(s)w$')
        ax[2].set_axis_off()
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        self.args['int_df'].plot(column='post_mean',ax=ax[1],legend=True,cax=cbar_ax)
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s + X(s)w$ With Events')

    def _plot_cov(self,**kwargs):
        """
        Plot spatial for covariates only.
        """
        self.spatial_cov['post_mean'] = self.samples['b_0'].mean(axis=0)
        fig, ax = plt.subplots(1,3, figsize=(10, 5),gridspec_kw={'width_ratios': [10,10,1]})
        self.spatial_cov.plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $X(s)w$')
        ax[2].set_axis_off()
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        self.spatial_cov.plot(column='post_mean',ax=ax[1],legend=True,cax=cbar_ax)
        ax[1].set_title('Mean Posterior $X(s)w$ With Events')
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s + X(s)w$ With Events')

    def _sim_cox(self, parameters, rng=None):
        """Exact sampler for the factorized Cox background, in internal units.

        mu(t,s) = g(t) h(s); N ~ Poisson(Ig*Ih); times ~ g/Ig via inverse CDF on the EXACT
        breakpoint partition of the piecewise-constant field (temporal-cell edges union
        seasonal crossings); locations ~ h*area/Ih via cell multinomial + uniform-in-cell.
        Returns np.array [N, 3] of (X_real, Y_real, T_internal).

        The sampler and the likelihood now both compute the EXACT integral of the
        piecewise-constant time field exp(a_0 + f_t + f_a[sigma(t)]) -- the sampler by
        summing g_j*len_j over breakpoint segments, the likelihood via the season_overlap
        matrix -- so Ig == Itot_time remains a float-precision identity, no longer a shared
        quadrature approximation. There is no longer any grid-coupling caveat to honor.

        rng: numpy.random.Generator, optional
            Used for the spatial GeoSeries.sample_points draw, which ignores numpy's legacy
            global seed; the Poisson / inverse-CDF / multinomial draws use np.random, so seed
            those with np.random.seed(...). Provide both for a fully reproducible draw.

        Known approximations (all vanish for a rectangle domain -- use a rectangle A for
        simulate-and-recover / SBC):
          - spatial_window truncation is intentionally NOT mirrored in offspring thinning;
            the spatial excitation integral is left untruncated by design, so keep
            spatial_window=None for calibration work.
          - For a GeoDataFrame domain, boundary cells are sampled over the FULL cell and then
            clipped by simulate()'s A-filter, and the excitation compensator integrates over
            the bounding rectangle A_ rather than the polygon.
        """
        n_t, T_int = self.args['n_t'], self.args['T']
        n_s, offset = self.args['n_s'], self.args['offset_seasonal']
        # --- EXACT breakpoint partition of [0, T_int]: split at every temporal-cell edge
        # AND every seasonal-cell crossing, so exp(a_0 + f_t + f_a[sigma(t)]) is constant on
        # each segment and the segment midpoint is a safe evaluation point. Ig computed here
        # equals the likelihood's Itot_time to float precision (both are the exact integral
        # of the same piecewise-constant field via the season_overlap matrix).
        edges = np.arange(n_t + 1) * (T_int / n_t)
        h_day = self.S / n_s
        # seasonal crossings: real days d in [0, self.T] with (d + offset) a multiple of h_day
        m_lo = int(np.ceil((offset) / h_day - 1e-9))
        m_hi = int(np.floor((self.T + offset) / h_day + 1e-9))
        cross_int = (np.arange(m_lo, m_hi + 1) * h_day - offset) * (T_int / self.T)
        bp = np.unique(np.clip(np.concatenate([edges, cross_int]), 0.0, T_int))
        seg_lo = bp[:-1]
        seg_len = np.diff(bp)
        mid = seg_lo + 0.5 * seg_len
        t_cell = np.clip((mid / (T_int / n_t)).astype(int), 0, n_t - 1)
        s_cell = np.clip(((mid * (self.T / T_int) + offset) % self.S / self.S * n_s).astype(int),
                         0, n_s - 1)
        g = np.exp(float(parameters['a_0']) + np.asarray(parameters['f_t'])[t_cell]
                   + np.asarray(parameters['f_a'])[s_cell])
        # COUNT rate from the likelihood's own atom: the Poisson mean below is
        # literally Itot_time * (sum of the eq. 24 masses). The breakpoint
        # masses w = g*seg_len still drive the CONDITIONAL time draw (self-
        # normalized; its normalization equals Ig to float precision, tested
        # by test_sim_likelihood_integral_identity).
        _, Ig = seasonal_time_integral(parameters['a_0'], parameters['f_t'],
                                       parameters['f_a'], self.args['season_overlap'])
        Ig = float(Ig)
        # --- spatial profile on the model's own grid (copy: no shared-state mutation)
        if 'spatial_cov' in self.args:
            geo_df = self.args['int_df']   # rows 1:1 with the integration arrays
        else:
            # Support geometries via INDEXED lookup by comp_grid_id -- never a
            # geometric self-join (see the array-domain fix commit and
            # test_sim_cox_array_domain_support_regression).
            fi = np.asarray(self.args['integration_field_indices'])
            geo_df = self.comp_grid.set_index('comp_grid_id').loc[fi]
        # One mass vector for BOTH the count rate and the conditional cell
        # draw: the eq. 24 masses atom (spatial_refinement_integral is its
        # sum, so Ih is the likelihood's own integral by construction).
        b_0 = parameters['b_0'] if 'spatial_cov' in self.args else None
        h_mass = np.asarray(spatial_refinement_masses(
            parameters['f_xy'], self.args['integration_field_indices'],
            self.args['integration_areas'], b_0,
            self.args['integration_cov_indices']))
        if len(geo_df) != len(h_mass):
            raise RuntimeError(f"geometry rows ({len(geo_df)}) misaligned with "
                               f"integration arrays ({len(h_mass)})")
        Ih = h_mass.sum()
        # --- exact two-step draw
        N = np.random.poisson(Ig * Ih)
        if N == 0:
            return np.empty((0, 3))
        # inverse-CDF on segment MASS (g_j * len_j), then uniform within the chosen segment
        w = g * seg_len
        cdf = np.cumsum(w) / w.sum()
        bins = np.searchsorted(cdf, np.random.uniform(size=N), side='right')
        times = seg_lo[bins] + np.random.uniform(size=N) * seg_len[bins]
        cells = np.random.choice(len(h_mass), size=N, p=h_mass / h_mass.sum())
        counts = np.bincount(cells, minlength=len(h_mass))
        xy = self._sample_cells(geo_df, counts, rng=rng)
        # times and locations are independent given the factorization: pairing is arbitrary
        return np.column_stack((xy, times))


    def set_window(self, window, spatial_window=None):
        window = float(window)
        if spatial_window is not None:
            spatial_window = float(spatial_window)
        self.args['window'] = window
        self.args['spatial_window'] = spatial_window

        # Recompute pairs with both windows
        coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
            self.args['t_events'],
            self.args['xy_events'][0],
            self.args['xy_events'][1],
            window=window,
            spatial_window=spatial_window
        )

        self.args['coords'] = coords
        self.args['t_vals'] = t_vals
        self.args['x_vals'] = x_vals
        self.args['y_vals'] = y_vals





    def get_params(self):
        pars = {}
        pars['z_spatial'] = self.args['z_dim_spatial']
        pars['z_temporal'] = self.args['z_dim_temporal']
        pars['z_seasonal'] = self.args['z_dim_seasonal']
        pars['f_xy'] = 0
        pars['f_t'] = 0
        pars['f_a'] = 0
        pars['a_0'] = 1
        if 'spatial_cov' in self.args:
            spatial_cov = self.args['spatial_cov']
            if spatial_cov.ndim == 1:
                spatial_cov = spatial_cov[:, None]
                self.args['spatial_cov'] = spatial_cov
            pars['w'] = spatial_cov.shape[1]
        # Convention: VAE latent dims (z_spatial/z_temporal/z_seasonal) are counted in k,
        # so absolute AIC is penalized for latent variables; only relative AIC comparisons
        # under the same convention are meaningful.
        return pars


class Hawkes_Model(Point_Process_Model):
    def __init__(self,data, A, T, cox_background='cox',temporal_trig=Temporal_Exponential,
                 spatial_trig=Spatial_Symmetric_Gaussian,window=None,spatial_window=None,**kwargs):
        r"""
        Spatiotemporal Point Process Model given by,

        $$\lambda(t,s) = \mu(s,t) + \sum_{i:t_i<t}{\alpha f(t-t_i;\beta) \varphi(s-s_i;\sigma^2)}$$

        where $f$ is defined by spatial_trig, $\\varphi$ is defined by spatial_trig. If cox_background is true, $\mu$ is given by

        $$\mu(s,t) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes.
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(15,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$

        Otherwise, the $\mu$ is given by

        $$\mu(s,t) = exp(a_0 + X(s)w)$$

        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        data: str or pd.DataFrame
            either file path or DataFrame containing spatiotemporal data. Columns must include 'X', 'Y', 'T'. The file must be sorted by 'T'.
        A: np.array [2x2], GeoDataFram
            Spatial region of interest. If np.array first row is the x-range, second row is y-range.
        T: float
            Maximum time in region of interest. Time is assumed to spart at 0.
        cox_background: bool
            use gaussian processes in background
        temporal_trig: class Trigger
            an implementation of Trigger to parameterize the temporal triggering mechanism.
        spatial_trig: class Trigger
            an implementation of Trigger to parameterize the spatial triggering mechanism.
        window: float, optional
            Temporal truncation window for the self-excitation kernel, in the internal
            rescaled time units (data time is rescaled to [0, 50]). Excitation pairs with
            dt > window are dropped from the sum, and the excitation integral is truncated
            to match -- this is the exact likelihood of the truncated-kernel model. window
            must comfortably exceed the posterior temporal scale beta (rule of thumb:
            window >= 5*beta) or be left at the default. Defaults to T, i.e. the full
            window / no truncation, which exactly reproduces the previous likelihood.
        spatial_window: float, optional
            Spatial truncation radius for excitation pairs, in the rescaled spatial units
            (the [0, 1] square). KNOWN APPROXIMATION: this truncates pairs spatially WITHOUT
            a matching truncation of the spatial integral, so it does not yield an exact
            likelihood; leave it None for calibration/SBC work. Defaults to None.
        sp_var_mu: float
            Fixed log-amplitude multiplier applied to the spatial VAE decoder output; see
            Point_Process_Model for calibration guidance. Default 2.0.
        kwargs: dict
            parameters from Point_Process_Model
        """
        self.model = spatiotemporal_hawkes_model
        if cox_background:
            name = 'cox_hawkes'
        else:
            name = 'hawkes'
        super().__init__(name, data, A, T, **kwargs)

        self.args['t_trig'] = temporal_trig(self.args['priors'])
        self.args['sp_trig'] = spatial_trig(self.args['priors'])
        if window is None:
            window = float(self.args['T'])   # exact likelihood: no truncation
        self.set_window(float(window),
                        float(spatial_window) if spatial_window is not None else None)

    def __str__(self):
        model = "Hawkes" if self.args['model'] == "hawkes" else "Cox Hawkes"
        return f"{model} Model with Covariates" if 'num_cov' in self.args else f"{model} Model without Covariates"

    def get_params(self):
        """
        Returns
        -------
            dict of parameter names as keys and lengths as values
        """
        pars = {}
        pars['alpha'] = 1
        for n in self.args['t_trig'].get_par_names():
            pars[n] = 1
        for n in self.args['sp_trig'].get_par_names():
            pars[n] = 1

        if self.args['model'] == 'cox_hawkes':
            pars['z_spatial'] = self.args['z_dim_spatial']
            pars['z_temporal'] = self.args['z_dim_temporal']
            pars['z_seasonal'] = self.args['z_dim_seasonal']
            pars['f_xy'] = 0
            pars['f_t'] = 0
            pars['f_a'] = 0
        pars['a_0'] = 1
        if 'spatial_cov' in self.args:
            spatial_cov = self.args['spatial_cov']
            if spatial_cov.ndim == 1:
                spatial_cov = spatial_cov[:, None]
                self.args['spatial_cov'] = spatial_cov
            pars['w'] = spatial_cov.shape[1]
            # b_0 = X(s)w is a deterministic site; request it (value 0 so it does
            # not count toward AIC k) so plot_spatial(include_cov=True) can read it.
            # Was dropped from this method in the fork; LGCP_Model.get_params keeps it.
            pars['b_0'] = 0
        # Convention: VAE latent dims (z_spatial/z_temporal/z_seasonal) are counted in k,
        # so absolute AIC is penalized for latent variables; only relative AIC comparisons
        # under the same convention are meaningful.
        return pars

    def plot_prop_excitation(self):
        """
        Plots a histogram of the posterior distribution of the proportion of the intensity due to self-excitation.

        Returns
        -------
            float: posterior mean of proportion of intensity due to self-excitation
        """
        p = self.samples['Itot_excite']/self.samples['Itot_txy']
        plt.hist(p,density=True)
        plt.title("Proportion of Intensity Due to Self-Excitation")
        plt.xlabel("Proportion of Intensity Due to Self-Excitation")
        return p.mean().item()

    def plot_trigger_posterior(self,trace=False):
        """
        Plot histograms of posterior trigger parameters.
        Returns
        -------
        pd.DataFrame
            Summary of trigger parameters.
        trace: bool
            plot trace or histogram of parameters
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        par_names = self.args['t_trig'].get_par_names()+self.args['sp_trig'].get_par_names()
        if trace:
            fig, ax = plt.subplots(1+len(par_names),1,figsize=(5,8), sharex=True)
            plt.suptitle("Trace Plots for Trigger Parameter Posteriors")
            ax[0].plot(self.samples['alpha'])
            ax[0].set_ylabel(r"${\alpha} $")
            for i in range(len(par_names)):
                ax[i+1].plot(self.samples[par_names[i]])
                ax[i+1].set_ylabel(par_names[i])
        else:
            fig, ax = plt.subplots(1, 1+len(par_names),figsize=(8,4), sharex=False)
            plt.suptitle("Trigger Parameter Posteriors")
            ax[0].hist(self.samples['alpha'])
            ax[0].set_xlabel(r"${\alpha} $")
            for i in range(len(par_names)):
                ax[i+1].hist(self.samples[par_names[i]])
                ax[i+1].set_xlabel(par_names[i])

        trig_pos = np.stack([self.samples[name] for name in ['alpha']+par_names]).T
        mean = trig_pos.mean(axis=0)
        std = trig_pos.var(axis=0)**0.5
        p_val = [(self.samples[name]>0).mean() for name in ['alpha']+par_names]
        quantiles = np.quantile(trig_pos,[0.025,0.975],axis=0)
        trig_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,r'P(w>0)':p_val,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=['alpha']+par_names)
        return trig_summary

    def plot_trigger_time_decay(self, t_units='days'):
        """
        Plot temporal trigger kernel sample posterior for a range of time lags.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")

        par_names = self.args['t_trig'].get_par_names()
        scale = 50 / self.T


        # Estimate a good maximum for t
        post_mean = {name: self.samples[name].mean().item() for name in par_names}
        t_max = self.args['T']
        t_grid = np.linspace(0, t_max, 250)  # 250 time lags from 0 to T

        fig, ax = plt.subplots(figsize=(7, 7))

        # Plot 100 posterior samples
        for i in np.random.choice(np.arange(len(self.samples['alpha'])), 100):
            pars = {name: self.samples[name][i].item() for name in par_names}
            # For each t in t_grid, create a fake (coords, t_vals) tuple
            # coords is not used for plotting, so just use dummy indices
            coords = np.zeros((len(t_grid), 2), dtype=int)
            t_vals = jnp.array(t_grid)
            _, trigger_vals = self.args['t_trig'].compute_trigger(pars, (coords, t_vals))
            ax.plot(t_grid / scale, trigger_vals, color='black', alpha=0.1)

        # Plot posterior mean
        coords = np.zeros((len(t_grid), 2), dtype=int)
        t_vals = jnp.array(t_grid)
        _, mean_trigger_vals = self.args['t_trig'].compute_trigger(post_mean, (coords, t_vals))
        ax.plot(t_grid / scale, mean_trigger_vals, color='blue', label='Posterior Mean')

        fig.suptitle('Time Decay of Trigger Function')
        ax.set_ylabel('Trigger Intensity')
        ax.set_xlabel(f'{t_units.capitalize()} After First Event')
        ax.axhline(0, color='black', linestyle='--')
        ax.axvline(0, color='black', linestyle='--')
        ax.legend()
        plt.show()

#------------------------------------------------    
    def analyze_trigger_decay_distribution(self, n_bootstrap=100, t_units='days'):
        """
        Bootstrap analysis of trigger decay characteristics
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        
        par_names = self.args['t_trig'].get_par_names()
        scale = 50 / self.T
        t_max = self.args['T']
        t_grid = np.linspace(0, t_max, 250)
        
        # Storage for characteristics
        half_lives = []
        decay_constants = []
        peak_intensities = []
        time_to_1percent = []
        
        # Bootstrap sampling
        n_samples = len(self.samples[par_names[0]])
        bootstrap_indices = np.random.choice(n_samples, n_bootstrap, replace=True)
        
        for i in bootstrap_indices:
            # Extract parameters for this sample
            pars = {name: self.samples[name][i].item() for name in par_names}
            
            # Compute trigger values
            coords = np.zeros((len(t_grid), 2), dtype=int)
            t_vals = jnp.array(t_grid)
            _, trigger_vals = self.args['t_trig'].compute_trigger(pars, (coords, t_vals))
            
            # Extract characteristics
            trigger_vals = np.array(trigger_vals)
            peak_intensity = np.max(trigger_vals)
            peak_intensities.append(peak_intensity)
            
            # Find half-life (time to 50% of peak)
            half_peak = peak_intensity / 2
            if np.any(trigger_vals <= half_peak):
                half_life_idx = np.where(trigger_vals <= half_peak)[0][0]
                half_life = t_grid[half_life_idx] / scale
                half_lives.append(half_life)
            
            # Find time to 1% of peak
            one_percent = peak_intensity * 0.01
            if np.any(trigger_vals <= one_percent):
                one_percent_idx = np.where(trigger_vals <= one_percent)[0][0]
                time_1pct = t_grid[one_percent_idx] / scale
                time_to_1percent.append(time_1pct)
            
            # Estimate decay constant (fit exponential decay)
            # Assuming exponential: f(t) = A * exp(-t/tau)
            # We can estimate tau from the slope in log space
            if len(trigger_vals) > 10:
                valid_vals = trigger_vals[trigger_vals > peak_intensity * 0.01]
                if len(valid_vals) > 5:
                    log_vals = np.log(valid_vals)
                    t_subset = t_grid[:len(valid_vals)] / scale
                    if len(t_subset) > 1:
                        # Simple linear fit to log values
                        slope = (log_vals[-1] - log_vals[0]) / (t_subset[-1] - t_subset[0])
                        tau = -1 / slope  # decay constant
                        decay_constants.append(tau)
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Distribution of Trigger Decay Characteristics (n={n_bootstrap} bootstrap samples)')
        
        # Half-life distribution
        if half_lives:
            axes[0,0].hist(half_lives, bins=20, alpha=0.7, edgecolor='black')
            axes[0,0].axvline(np.mean(half_lives), color='red', linestyle='--', 
                            label=f'Mean: {np.mean(half_lives):.1f} {t_units}')
            axes[0,0].set_xlabel(f'Half-life ({t_units})')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Half-life Distribution')
            axes[0,0].legend()
        
        # Peak intensity distribution
        if peak_intensities:
            axes[0,1].hist(peak_intensities, bins=20, alpha=0.7, edgecolor='black')
            axes[0,1].axvline(np.mean(peak_intensities), color='red', linestyle='--',
                            label=f'Mean: {np.mean(peak_intensities):.3f}')
            axes[0,1].set_xlabel('Peak Trigger Intensity')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Peak Intensity Distribution')
            axes[0,1].legend()
        
        # Decay constant distribution
        if decay_constants:
            axes[1,0].hist(decay_constants, bins=20, alpha=0.7, edgecolor='black')
            axes[1,0].axvline(np.mean(decay_constants), color='red', linestyle='--',
                            label=f'Mean: {np.mean(decay_constants):.1f} {t_units}')
            axes[1,0].set_xlabel(f'Decay Constant ({t_units})')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Decay Constant Distribution')
            axes[1,0].legend()
        
        # Time to 1% distribution
        if time_to_1percent:
            axes[1,1].hist(time_to_1percent, bins=20, alpha=0.7, edgecolor='black')
            axes[1,1].axvline(np.mean(time_to_1percent), color='red', linestyle='--',
                            label=f'Mean: {np.mean(time_to_1percent):.1f} {t_units}')
            axes[1,1].set_xlabel(f'Time to 1% of Peak ({t_units})')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Time to Effective Zero')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("TRIGGER DECAY CHARACTERISTICS SUMMARY:")
        print("="*50)
        if half_lives:
            print(f"Half-life: {np.mean(half_lives):.1f} ± {np.std(half_lives):.1f} {t_units}")
        if peak_intensities:
            print(f"Peak Intensity: {np.mean(peak_intensities):.3f} ± {np.std(peak_intensities):.3f}")
        if decay_constants:
            print(f"Decay Constant: {np.mean(decay_constants):.1f} ± {np.std(decay_constants):.1f} {t_units}")
        if time_to_1percent:
            print(f"Time to 1% of Peak: {np.mean(time_to_1percent):.1f} ± {np.std(time_to_1percent):.1f} {t_units}")
        
        return {
            'half_lives': half_lives,
            'peak_intensities': peak_intensities, 
            'decay_constants': decay_constants,
            'time_to_1percent': time_to_1percent
        }
#------------------------------------------------


    def _sample_cells(self, geo_df, counts, rng=None):
        """Uniform points within cells: counts[k] points in geo_df row k.

        Shared by the Cox and pure-Hawkes background samplers so both paths
        draw locations from one tested primitive. rng=None means geopandas'
        own nondeterministic default -- the pure-Hawkes caller deliberately
        passes None to preserve its documented legacy behavior.
        """
        counts = np.asarray(counts)
        nz = counts > 0
        pts = geo_df[nz].sample_points(size=counts[nz], rng=rng).explode(index_parts=False)
        if len(pts) != counts.sum():
            raise RuntimeError(f"sample_points returned {len(pts)} points for "
                               f"{counts.sum()} requested; refusing to truncate")
        return np.stack((pts.x.values, pts.y.values), axis=1)

    def _sim_hawkes_bg(self, parameters):
        """Constant / covariate background via per-cell Poisson superposition.

        Cell rates are background_masses(...) -- the integrand of the
        background compensator atoms, whose sum the unit tests pin against
        constant/covariate_background_integral -- so the simulator and
        likelihood share one background-mass expression (no runtime assert:
        one shared computation plus tests, per review). Superposing per-cell
        Poisson draws is distributionally identical to Poisson(total) +
        multinomial. Legacy quirk preserved and documented: the sample_points
        draw here is UNSEEDED (ignores np.random and the rng argument);
        unifying all draws under one Generator is a recorded follow-up, and
        until then simulate() is not fully seed-reproducible on this path.
        """
        a_0 = float(parameters['a_0'])
        T_int = self.args['T']
        if 'spatial_cov' in self.args:
            geo_df = self.spatial_cov
            mu = np.exp(a_0 + np.asarray(parameters['b_0']))
            areas = np.asarray(self.args['cov_area'])
        else:
            geo_df = self.A
            A_ = self.args['A_']
            mu = np.exp(a_0)
            areas = (geo_df.area / ((A_[0, 1]-A_[0, 0]) * (A_[1, 1]-A_[1, 0]))).values
        cell_rates = np.asarray(background_masses(mu, areas, T_int))
        num = np.random.poisson(cell_rates)
        xy = self._sample_cells(geo_df, num, rng=None)
        return np.column_stack((xy, np.random.uniform(0, T_int, size=len(xy))))

    def _sim_offspring(self,bg,par):
        i = 0
        while i < len(bg):
            for j in range(np.random.poisson(lam=par['alpha'])):
                #simulate trigger and rescale
                sp_dif = (self.args['A_'][:,1]-self.args['A_'][:,0])*\
                            self.args['sp_trig'].simulate_trigger(par)
                t_dif = [self.args['t_trig'].simulate_trigger(par)]
                # window-consistent thinning: match the truncated-kernel likelihood
                # (Poisson(alpha) parents thinned by F(w) => expected offspring alpha*F(w))
                if t_dif[0] > self.args['window']:
                    continue
                cand = bg[i] + np.append(sp_dif, t_dif)
                # Prop 1.1(ii): offspring outside the bounding rectangle X --
                # the region the compensator charges (eq. 27) -- are discarded
                # BEFORE they can parent. Pre-fix they stayed in the cascade
                # until simulate()'s final sjoin, so hidden out-of-domain
                # events excited observed ones (second-order count bias; see
                # test_offspring_cascade_discards_outside_rectangle_before_parenting).
                A_ = self.args['A_']
                if not (A_[0, 0] <= cand[0] <= A_[0, 1]
                        and A_[1, 0] <= cand[1] <= A_[1, 1]):
                    continue
                bg = np.concatenate((bg,[cand]))
            i += 1
        return bg

    def simulate(self,parameters=None,rng=None):
        """
        Simulate data from mean posterior parameters.
        Parameters
        ----------
        parameters: dict
            Parameters to simulate from. If parameters is None, use mean of posterior samples. keys are string parameter names. values are np.array or float. Names must be same as those that appear in the sample from the model.
        rng: numpy.random.Generator, optional
            Reproducible spatial sampling; passed through to _sim_cox.sample_points. Seed the
            Poisson/CDF draws separately with np.random.seed(...). See _sim_cox.
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns (real units)
                simulated data
        """
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        # Same decode functions as the model layer (decoders.py): before
        # extraction this was the THIRD independent copy of the decoder logic.
        if 'f_t' not in parameters and 'z_temporal' in parameters:
            v_t = decode_temporal_field(parameters['z_temporal'],
                                        self.args["decoder_params_temporal"],
                                        self.args["hidden_dim_temporal"], self.args["n_t"])
            parameters['f_t'] = v_t[0:self.args["n_t"]]
        if 'f_a' not in parameters and 'z_seasonal' in parameters:
            v_a = decode_seasonal_field(parameters['z_seasonal'],
                                        self.args["decoder_params_seasonal"],
                                        self.args["hidden_dim1_seasonal"],
                                        self.args["hidden_dim2_seasonal"], self.args["n_s"])
            parameters['f_a'] = v_a[0:self.args["n_s"]]
        if 'f_xy' not in parameters and 'z_spatial' in parameters:
            # exp(sp_var_mu) calibration applied INSIDE decode_spatial_field
            parameters['f_xy'] = decode_spatial_field(
                parameters['z_spatial'], self.args["decoder_params_spatial"],
                self.args["hidden_dim1_spatial"], self.args["hidden_dim2_spatial"],
                self.args["n_xy"], self.args['sp_var_mu'])
        if 'w' in parameters and 'b_0' not in parameters:
            parameters['b_0'] = self.args['spatial_cov'] @ parameters['w']

        if self.args['model'] == 'cox_hawkes':
            bg = self._sim_cox(parameters, rng=rng)
        else:
            bg = self._sim_hawkes_bg(parameters)
        sample = self._sim_offspring(bg,parameters)
        #filter out offspring after cutoff
        sample = sample[sample.T[2]<self.args['T']]
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        #filter to time window
        points['T'] = (points['T']*self.T/self.args['T'])
        #filter to spatial window
        return points.sjoin(self.A[['geometry']])[['X','Y','T','geometry']]


class LGCP_Model(Point_Process_Model):
    def __init__(self,*args,**kwargs):
        r"""
        Spatiotemporal LGCP Model given by,

        $$\lambda(t,s) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes.
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(15,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$

        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        args: list
            Parameters from Point_Process_Model
        sp_var_mu: float
            Fixed log-amplitude multiplier applied to the spatial VAE decoder output; see
            Point_Process_Model for calibration guidance. Default 2.0.
        kwargs: dict
            Parameters from Point_Process_Model
        """
        name = 'lgcp'
        self.model = spatiotemporal_LGCP_model
        super().__init__(name,*args,**kwargs)

    def __str__(self):
        return "Log Gaussian Cox Model with Covariates" if 'num_cov' in self.args else "Log Gaussian Cox Model without Covariates"

    def get_params(self):
        """
        Returns
        -------
            dict of parameter names as keys and lengths as values
        """
        pars = {}
        pars['z_spatial'] = self.args['z_dim_spatial']
        pars['z_temporal'] = self.args['z_dim_temporal']
        pars['z_seasonal'] = self.args['z_dim_seasonal']
        pars['f_xy'] = 0
        pars['f_t'] = 0
        pars['f_a'] = 0
        pars['a_0'] = 1
        if 'spatial_cov' in self.args:
            spatial_cov = self.args['spatial_cov']
            if spatial_cov.ndim == 1:
                spatial_cov = spatial_cov[:, None]
                self.args['spatial_cov'] = spatial_cov
            pars['w'] = spatial_cov.shape[1]
            pars['b_0'] = 0
        # Convention: VAE latent dims (z_spatial/z_temporal/z_seasonal) are counted in k,
        # so absolute AIC is penalized for latent variables; only relative AIC comparisons
        # under the same convention are meaningful.
        return pars

    def simulate(self,parameters=None,rng=None):
        """
        Simulate data from mean posterior parameters. Requires model inference.
        Parameters
        ----------
        parameters: dict
            Parameters to simulate from. If parameters is None, use mean of posterior samples. keys are string parameter names. values are np.array or float. Names must be same as those that appear in the sample from the model.
        rng: numpy.random.Generator, optional
            Reproducible spatial sampling; passed through to _sim_cox.sample_points. Seed the
            Poisson/CDF draws separately with np.random.seed(...). See _sim_cox.
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns (real units)
                simulated data
        """
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        sample = self._sim_cox(parameters, rng=rng)
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        points['T'] = (points['T']*self.T/self.args['T'])

        return points
