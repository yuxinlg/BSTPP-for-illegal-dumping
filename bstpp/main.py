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

from .utils import (aligned_difference_pairs, exp_sq_kernel,
                    within_real_box_window, accepts_rng_kwarg)
from .inference_functions import (spatiotemporal_hawkes_model, spatiotemporal_LGCP_model,
                                  run_mcmc, run_SVI, get_samples)
from .trigger import Temporal_Exponential, Spatial_Symmetric_Gaussian
from .decode_fields import (decode_temporal_field, decode_seasonal_field,
                        decode_spatial_field)
from .likelihood import (seasonal_time_integral, spatial_refinement_masses,
                         background_masses)
from .data_contracts import (validate_events, validate_covariates,
                             validate_covariate_coverage, enforce,
                             DataContractError)
from .preparation import (ModelData, prepare_domain, prepare_partitions,
                          attach_covariate_partitions,
                          finalize_integration_arrays, T_INTERNAL)
from .config import NumericalConfig
from .excitation_support import (
    build_excitation_support,
    resolve_excitation_support_mode,
    resolve_sigma_bounds,
    truncate_sigmax_2_prior,
)
from .polygon_mass import assert_polygon_mass_table_budget, warn_if_sigma_near_bound
from .cutoffs import (
    CutoffProvenance,
    SpatialCutoffRecord,
    TemporalCutoffRecord,
    internal_to_days,
    resolve_computational_cutoffs,
    scale_temporal_prior_to_internal,
    spatial_omitted_mass,
    temporal_omitted_mass,
    days_to_internal,
)

# Sentinel for set_window: omitted arguments leave the current value unchanged.
# Explicit None clears / restores the untruncated default (distinct from omit).
_UNSET = object()


def _is_exact_temporal_exponential(temporal_trig) -> bool:
    """Exact-type capability gate for exponential tolerance / mean-lag APIs."""
    return temporal_trig is Temporal_Exponential


def _is_exact_spatial_gaussian(spatial_trig) -> bool:
    """Exact-type capability gate for Gaussian sigma / spatial-tolerance APIs."""
    return spatial_trig is Spatial_Symmetric_Gaussian


def _reject_unsupported_kernel_kwargs(
    *,
    temporal_trig,
    spatial_trig,
    mean_lag_days_prior,
    temporal_cutoff_tol,
    spatial_cutoff_tol,
    cutoff_tol,
    design_mean_lag_days,
    design_sigma,
    min_sigma,
    max_sigma,
) -> None:
    """Reject exponential/Gaussian-only cutoff kwargs for unsupported kernels.

    Exact-type gates (not isinstance, not parameter-name inference). Shared
    ``cutoff_tol`` requires both axes to own their tolerance capability.
    """
    temporal_ok = _is_exact_temporal_exponential(temporal_trig)
    spatial_ok = _is_exact_spatial_gaussian(spatial_trig)
    t_name = getattr(temporal_trig, "__name__", repr(temporal_trig))
    s_name = getattr(spatial_trig, "__name__", repr(spatial_trig))

    if cutoff_tol is not None and not (temporal_ok and spatial_ok):
        raise TypeError(
            "Shared cutoff_tol requires Temporal_Exponential and "
            "Spatial_Symmetric_Gaussian on both axes (got "
            f"temporal_trig={t_name}, spatial_trig={s_name}). "
            "Use axis-specific temporal_cutoff_tol / spatial_cutoff_tol "
            "when only one axis has the supported exact kernel.")

    if not temporal_ok:
        if mean_lag_days_prior is not None:
            raise TypeError(
                "mean_lag_days is valid only for Temporal_Exponential "
                f"(got temporal_trig={t_name}). Temporal_Power_Law.beta is a "
                "shape parameter, not a mean lag; pass an explicit window / "
                "temporal_cutoff_days and a legacy beta prior instead.")
        if temporal_cutoff_tol is not None:
            raise TypeError(
                "temporal_cutoff_tol is valid only for Temporal_Exponential "
                f"(got temporal_trig={t_name}). Pass an explicit window / "
                "temporal_cutoff_days for unsupported temporal triggers.")
        if design_mean_lag_days is not None:
            raise TypeError(
                "design_mean_lag_days is valid only for Temporal_Exponential "
                f"(got temporal_trig={t_name}). Pass an explicit physical "
                "temporal cutoff instead.")

    if not spatial_ok:
        if spatial_cutoff_tol is not None:
            raise TypeError(
                "spatial_cutoff_tol is valid only for "
                "Spatial_Symmetric_Gaussian "
                f"(got spatial_trig={s_name}). Pass an explicit "
                "spatial_window for custom rectangle spatial triggers.")
        if design_sigma is not None:
            raise TypeError(
                "design_sigma is valid only for Spatial_Symmetric_Gaussian "
                f"(got spatial_trig={s_name}).")
        if min_sigma is not None or max_sigma is not None:
            raise TypeError(
                "min_sigma / max_sigma apply only to "
                "Spatial_Symmetric_Gaussian "
                f"(got spatial_trig={s_name}).")


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
                 cov_grid_size=None,spatial_cov_crs=None,standardize_cov=None,sp_var_mu=2.0,
                 data_contracts='reject',**kwargs):
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
        spatial_cov_crs: CRS-like, optional
            Declared CRS of CSV/plain-DataFrame covariate coordinates (parsed by
            ``CRS.from_user_input``). Required when those tabular covariates are
            paired with a CRS-bearing domain; must match the domain CRS.
            GeoDataFrame covariates are self-describing and do not use this
            argument. Array domains (no CRS) do not require it.
        standardize_cov: None or str
            None (default): covariate values preserved exactly as supplied.
            ``'domain_area'``: area-weighted standardization with the exact
            clipped |C_c ∩ A| areas as weights (D-11); cells with no domain
            mass do not influence mean/scale. Legacy booleans are rejected
            explicitly (OP-3/OP-4 settled pre-3f). Whichever method runs is
            reported on self.standardization (D-10):
            {'method', 'columns', 'mean', 'scale'}, with
            X = standardized * scale + mean.
        sp_var_mu: float
            Fixed log-amplitude multiplier applied to the spatial VAE decoder output.
            Calibrate as sp_var_mu = 0.5*log(sigma2_target / E[f^2]_decoder), where
            E[f^2]_decoder is the empirical second moment of decoder outputs under z~N(0,I)
            and sigma2_target is the intended marginal variance of the spatial log-intensity
            field. Note this is an amplitude multiplier, NOT the var_loc used in VAE
            training. A sampled amplitude (and a matching knob for the seasonal field, which
            currently has none) is planned follow-up work.
        data_contracts: str
            'reject' (default) or 'report'. Phase 3a boundary validation:
            nonfinite coordinates/covariates, out-of-horizon times,
            out-of-domain events (polygon boundary is inside, D-4), invalid
            geometry, and CRS mismatches are collected into
            self.data_contract_report. 'reject' raises DataContractError
            listing every offending row; 'report' warns and leaves legacy
            behavior bit-unchanged (the section-14 dry-run instrument).
            Default flipped report -> reject on reviewer sign-off of the
            committed dry run (2026-07-20); see
            refactor-patches/phase3a/rebaseline_record.md.
        priors: dict
            priors for parameters (a_0,w,alpha,beta,sigmax_2). Must be a numpyro distribution.
        """
        if type(data) is str:
            data = pd.read_csv(data)
        # Ownership: retain owned copies of caller-supplied mutable inputs.
        data = data.copy()
        self.data = data

        # Phase 3a data contracts: validate the raw events and domain BEFORE
        # any grid construction, so reject mode fails with the actual defect
        # rather than a downstream sjoin symptom. Report mode only warns:
        # every numerical code path below is bit-unchanged.
        self._data_contracts_mode = data_contracts
        self.data_contract_report = enforce(
            validate_events(data, A, T), len(data), data_contracts)

        # D-10: always report whether/how covariates were standardized;
        # overwritten by the covariate leg when covariates are supplied.
        self.standardization = {'method': 'none', 'columns': [],
                                'mean': None, 'scale': None}

        # Phase 3b seam: owned copies of events/domain (prepare_domain also
        # copies); covariate_source stays the raw handle for load-error
        # ordering until the covariate leg normalizes and copies.
        if type(A) is gpd.GeoDataFrame:
            domain_owned = A.copy(deep=True)
        else:
            domain_owned = np.array(A, dtype=float, copy=True)
        self.model_data = ModelData(
            events=data.copy(), domain=domain_owned, horizon_days=T,
            offset_seasonal=offset_seasonal, covariate_source=spatial_cov,
            cov_names=cov_names, cov_grid_size=cov_grid_size)

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

        # Phase 3b seam: domain geometry, bounding rectangle, unit scales and
        # the geographic-CRS contract warning live in prepare_domain; args
        # entries below are the legacy adapter view of the same objects.
        self.prepared_domain = prepare_domain(A)
        A_ = self.prepared_domain.bounds
        args['A_area'] = self.prepared_domain.area_ratio
        args['A_'] = A_
        args['axis_scales'] = self.prepared_domain.axis_scales

        # Phase 3b seam: temporal/seasonal/spatial partitions, comp_grid,
        # in-domain cells, season_idx_of_t and the exact overlap matrix W
        # live in prepare_partitions (verbatim extraction); args entries are
        # the legacy adapter view of the same objects.
        parts = prepare_partitions(self.prepared_domain, T, offset_seasonal)
        self.prepared_partitions = parts
        self.S = 365
        args["n_t"] = parts.n_t
        args["x_t"] = parts.x_t
        args["n_s"] = parts.n_s
        args["x_a"] = parts.x_a
        args["n_xy"] = parts.n_xy
        args['spatial_grid_cells'] = parts.spatial_grid_cells
        args['season_idx_of_t'] = parts.season_idx_of_t
        args['season_overlap'] = parts.season_overlap
        comp_grid = parts.comp_grid
        self.comp_grid = comp_grid
        # Polygon plotting domain is the owned PreparedDomain.domain copy.
        self.A = (self.prepared_domain.domain
                  if self.prepared_domain.is_polygon else comp_grid)
        self.T = T

        args,points = self._scale_xyt(data,args,parts.support_cells)
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
                # Tabular X/Y have no self-describing CRS. When the domain
                # declares a CRS, require an explicit spatial_cov_crs and
                # assign THAT value before validation -- never infer by
                # copying the domain CRS.
                domain_crs = self.prepared_domain.crs
                if domain_crs is not None:
                    if spatial_cov_crs is None:
                        raise ValueError(
                            "spatial_cov_crs is required when CSV/plain-"
                            "DataFrame covariates are used with a CRS-"
                            "bearing domain; declare the CRS of the "
                            "tabular X/Y coordinates (e.g. "
                            "spatial_cov_crs='EPSG:26918').")
                    from pyproj.crs import CRS
                    spatial_cov = spatial_cov.set_crs(
                        CRS.from_user_input(spatial_cov_crs))
                elif spatial_cov_crs is not None:
                    from pyproj.crs import CRS
                    spatial_cov = spatial_cov.set_crs(
                        CRS.from_user_input(spatial_cov_crs))

            # Phase 3a data contracts, covariate leg: validated on the
            # normalized GeoDataFrame, before the legacy membership sjoin, so
            # reject mode names the defect instead of the sjoin symptom.
            # Phase 3c coverage contract (IV; D-7): the layer must cover A
            # exactly once -- gaps and positive-area overlaps are violations
            # with the actual offending geometries exported on the report.
            _pts_xy = np.column_stack((
                pd.to_numeric(data['X'], errors='coerce').to_numpy(dtype=float),
                pd.to_numeric(data['Y'], errors='coerce').to_numpy(dtype=float)))
            self.data_contract_report.checks.extend(enforce(
                validate_covariates(spatial_cov, cov_names, A, points_xy=_pts_xy)
                + validate_covariate_coverage(spatial_cov, A),
                len(data), data_contracts).checks)

            # Ownership: copy before attaching cov_ind / membership so the
            # caller's frame is never mutated or retained.
            spatial_cov = spatial_cov.copy(deep=True)
            spatial_cov['cov_ind'] = np.arange(len(spatial_cov))
            # Align points CRS for the covariate sjoin without the deprecated
            # attribute override (GeoDataFrame.crs = ...). Points already carry
            # the domain CRS from _scale_xyt when the domain declares one;
            # validate_covariates has already required CRS equality, so a
            # matching CRS needs no write. Only set when points are CRS-less.
            if self.points.crs is None and spatial_cov.crs is not None:
                self.points = self.points.set_crs(spatial_cov.crs)
            # D-22 unique membership, covariate leg: an event exactly on a
            # shared covariate-polygon edge joins every incident polygon;
            # ties resolve deterministically to the LARGEST cov_ind,
            # mirroring the field grid's max-comp_grid_id rule (identical to
            # left-closed on row-major raster-like layers; for arbitrary
            # polygon layers it is a documented deterministic convention,
            # not a geometric statement). Unique joins are bit-unchanged.
            args['cov_ind'] = (self.points.sjoin(spatial_cov)
                               .groupby('point_id')['cov_ind'].max()
                               .sort_index().values)
            if len(args['cov_ind']) != len(self.points):
                raise Exception("Spatial covariates are not defined for all data points!")

            args['num_cov'] = len(cov_names)
            self.cov_names = cov_names

            # Phase 3b seam: covariate partition products (design matrix,
            # common refinement with exact intersection areas, in-domain
            # cell override / covariate areas) live on PreparedPartitions;
            # args entries below are the legacy adapter view. Event
            # membership (cov_ind above) stays event-side by design.
            # attach_covariate_partitions copies spatial_cov into
            # parts.cov_gdf; self.spatial_cov uses that owned frame.
            attach_covariate_partitions(parts, self.prepared_domain,
                                        spatial_cov, cov_names,
                                        standardize_cov, args['model'])
            self.spatial_cov = parts.cov_gdf
            # D-10 (3c API): the model always reports whether/how it
            # standardized -- method 'none' | 'domain_area' with
            # invertible mean/scale (X = standardized * scale + mean).
            # Legacy method name 'count' is not implemented.
            self.standardization = parts.standardization
            args['spatial_cov'] = parts.cov_values
            if args['model'] in ['lgcp','cox_hawkes']:
                args['int_df'] = parts.int_df
                args['spatial_grid_cells'] = parts.spatial_grid_cells
            else:
                args['cov_area'] = parts.cov_area

        # Integration arrays for the pure spatial_refinement_integral atom
        # (eq. 24): plain NumPy arrays only, derived once on the seam object,
        # so no pandas / GeoPandas object is ever read inside traced
        # likelihood code.
        finalize_integration_arrays(parts, args['model'])
        if args['model'] in ['lgcp', 'cox_hawkes']:
            args['integration_field_indices'] = parts.integration_field_indices
            args['integration_cov_indices'] = parts.integration_cov_indices
            args['integration_areas'] = parts.integration_areas

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


    def run_svi(self,num_steps,lr,num_samples=1000,resume=False,plot_loss=True,
                rng_key=None,**kwargs):
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
        rng_key: jax PRNGKey, optional
            Key used for SVI optimization and posterior sampling. When
            omitted, preserve the package's historical fixed-key behavior
            (``PRNGKey(10)`` split). Same convention as ``run_mcmc``.

        auto_guide: numpyro AutoGuide, default=AutoMultivariateNormal
            See numpyro AutoGuides for details.
        init_strategy: function, default=init_to_median
            See numpyro init strategy documentation
        """
        # Same None → PRNGKey(10) convention as run_mcmc (deterministic default).
        if rng_key is None:
            rng_key, rng_key_predict = random.split(random.PRNGKey(10))
            rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        else:
            rng_key_post = rng_key
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
            self.svi_results = self.svi.run(rng_key_post, kwargs['num_steps'], self.args,
                                            init_state=self.svi_results.state)
            self.samples = get_samples(rng_key_post,self.model,self.svi.guide,self.svi_results,self.args,sites)
        else:
            self.svi,self.svi_results,self.samples=run_SVI(rng_key_post, self.model, self.args, num_steps, lr, sites, **kwargs)
        self._warn_sigma_near_bound()
        if plot_loss:
            loss = np.asarray(self.svi_results.losses)
            plt.plot(np.arange(int(.01*len(loss)),len(loss)),loss[int(.01*len(loss)):])
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()


    def run_mcmc(self,batch_size=1,num_warmup=500,num_samples=1000,
                 num_chains=1,thinning=1,rng_key=None):
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
        rng_key: jax PRNGKey, optional
            Key used for NUTS initialization and transitions. When omitted,
            preserve the package's historical fixed-key behavior. Supplying a
            key lets replicated workflows such as SBC use an independent,
            reproducible MCMC stream for each fit.
        """
        self.args["batch_size"]= batch_size
        self.args["num_warmup"]= num_warmup
        self.args["num_samples"] = num_samples
        self.args["num_chains"] = num_chains
        self.args["thinning"] = thinning
        if rng_key is None:
            rng_key, rng_key_predict = random.split(random.PRNGKey(10))
            rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        else:
            rng_key_post = rng_key

        self.mcmc = run_mcmc(rng_key_post, self.model, self.args)
        self.samples=self.mcmc.get_samples()
        self._warn_sigma_near_bound()


    def _warn_sigma_near_bound(self):
        """Warn when posterior sigma approaches a configured sigma bound."""
        support = self.args.get('excitation_support')
        if support is None or support.max_sigma_real is None:
            return
        samples = getattr(self, 'samples', None)
        if samples is None or 'sigmax_2' not in samples:
            return
        warn_if_sigma_near_bound(
            samples['sigmax_2'],
            support.max_sigma_real,
            min_sigma_real=support.min_sigma,
        )


    def _scale_xyt(self,data,args,field_support):
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
        # Exact float64 real-unit event coordinates (row order matches
        # xy_events). Mass-table identity must use these, not a roundtrip
        # through float32 axis_scales which can alias distinct events.
        args["xy_events_real"] = np.asarray(
            (np.asarray(data["X"], dtype=np.float64),
             np.asarray(data["Y"], dtype=np.float64)),
            dtype=np.float64,
        )

        geometry = gpd.points_from_xy(data.X, data.Y,crs=field_support.crs)
        points = gpd.GeoDataFrame(data=data,geometry=geometry)
        points['point_id'] = np.arange(len(data))

        #find grid cells where points are located
        # D-22 deterministic unique membership (3a micro-rebaseline): cells are
        # left-closed/right-open per axis, [e_k, e_{k+1}), with the domain's
        # max-x/max-y edges closed. A point exactly on an internal grid line
        # intersects both adjacent cells in the sjoin; the left-closed cell is
        # the one with the LARGER comp_grid_id (ids are row-major from the
        # bottom-left: +1 across an x-edge, +25 across a y-edge, both at a
        # corner), so ties resolve to the per-point max id. Points with a
        # unique join -- all strictly-interior events -- are bit-unchanged.
        # Points exactly on the outermost right/top edge join only the last
        # cell, which IS the closed-edge assignment.
        # 3c-2 (10.c clipped-geometry reuse): the join source is the SUPPORT
        # object (C_c ∩ A clipped cells; the full grid on rectangle domains,
        # where this is the legacy join bit-identically), so every event maps
        # to a cell that carries domain mass. Where ∂A runs along a grid line
        # the left-closed cell can have zero support; the max-id rule then
        # resolves within the support (the event's mass-carrying cell), the
        # one refinement of D-22 this introduces. Events outside A join
        # nothing and fail the length check below (D-3 fail-fast) -- under
        # data_contracts='report' the contract warning naming the defect has
        # already fired by the time this raises.
        _joined = (points.sjoin(field_support[['comp_grid_id', 'geometry']])
                   .groupby('point_id')['comp_grid_id'].max())
        args['indices_xy'] = _joined.sort_index().values

        if len(args['indices_xy']) != len(points):
            raise Exception(
                "Computational grid does not encompass all data points! "
                f"{len(points) - len(args['indices_xy'])} event(s) have no "
                "supported field cell: nonfinite coordinates, or located "
                "outside the model domain A (out-of-domain events are "
                "rejected, D-3).")

        args["xy_events"]=xy_events_total.transpose()
        return args,points

    def log_expected_likelihood(self, data, *, mass_table=None):
        """Expected log likelihood of ``data`` under the fitted posterior.

        Held-out ``data`` is a separate realization: all event-indexed state
        (pairs, ``cov_ind``, polygon mass-table rows) is rebuilt from it.
        Polygon excitation mode hard-requires ``mass_table=`` prepared for
        those held-out events via ``prepare_polygon_mass_table``; silent
        table construction is not allowed. Rectangle mode ignores
        ``mass_table``.
        """
        if type(data) is str:
            data = pd.read_csv(data)
        if 'day' in data.columns:
            data = data.drop(columns=['day'])
        for col in ['X', 'Y', 'T']:
            if not np.issubdtype(data[col].dtype, np.number):
                data[col] = pd.to_numeric(data[col], errors='coerce')
        # Held-out scoring never silently cleans: NaN / +inf / -inf all fail
        # loudly under every data_contracts mode (including "report").
        # dropna() alone is insufficient — it removes NaN but keeps ±inf.
        _xy_t = data[['X', 'Y', 'T']].to_numpy(dtype=float)
        _nonfinite_mask = ~np.isfinite(_xy_t)
        if _nonfinite_mask.any():
            _field_counts = {
                name: int(_nonfinite_mask[:, j].sum())
                for j, name in enumerate(('X', 'Y', 'T'))
                if int(_nonfinite_mask[:, j].sum()) > 0
            }
            _n_rows = int(_nonfinite_mask.any(axis=1).sum())
            raise DataContractError(
                f"{_n_rows} held-out row(s) have non-numeric or nonfinite "
                f"coordinates/times (field counts: {_field_counts}); "
                "held-out scoring rejects all nonfinite inputs and does not "
                "drop, clip, or repair them")
        if len(data) == 0:
            raise ValueError("No held-out events to score")

        # Only pass the minimal required arguments for likelihood
        test_args, points = self._scale_xyt(data, self.args.copy(),
                                            self.prepared_partitions.support_cells)

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
                axis_scales=np.asarray(self.args['axis_scales']),
            )
            test_args['coords'] = coords
            test_args['t_vals'] = t_vals
            test_args['x_vals'] = x_vals
            test_args['y_vals'] = y_vals

            # Polygon Hermite tables are per-parent-event. Reusing the training
            # table crashes when counts differ and silently mis-scores when
            # counts match but locations differ. Require an explicit held-out
            # table; never mutate self.args / self.excitation_support and never
            # build a table silently here.
            train_support = self.args.get('excitation_support')
            if (train_support is not None
                    and getattr(train_support, 'mode', None) == 'polygon'):
                if mass_table is None:
                    raise ValueError(
                        "Polygon log_expected_likelihood requires mass_table= "
                        "from bstpp.polygon_mass.prepare_polygon_mass_table(...) "
                        "built for the held-out events; silent rebuild is not "
                        "allowed.")
                test_args['excitation_support'] = (
                    self._excitation_support_for_events(
                        *self._xy_events_to_real(test_args),
                        mass_table=mass_table))
            elif mass_table is not None:
                raise ValueError(
                    "mass_table is only accepted for polygon excitation_support")

        if 'cov_ind' in self.args:
            # Same D-22 max-cov_ind tie rule as training ingestion. This also
            # closes a silent misalignment: a held-out event on a shared
            # covariate edge used to emit TWO rows here with no length check,
            # shifting every later event's covariate silently.
            test_args['cov_ind'] = (points.sjoin(self.spatial_cov)
                                    .groupby('point_id')['cov_ind'].max()
                                    .sort_index().values)
            if len(test_args['cov_ind']) != len(points):
                raise ValueError(
                    "Held-out cov_ind length does not match the held-out event "
                    f"count: got {len(test_args['cov_ind'])} assignments for "
                    f"{len(points)} events (incomplete covariate coverage or "
                    "failed membership).")

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

    def _sample_cells(self, geo_df, counts, rng=None):
        """Uniform points within cells: counts[k] points in geo_df row k.

        Shared by the Cox and pure-Hawkes background samplers so both paths
        draw locations from one tested primitive. rng=None means geopandas'
        own nondeterministic default -- the pure-Hawkes caller deliberately
        passes None to preserve its documented legacy behavior.

        Defined on Point_Process_Model because _sim_cox (also on this class)
        calls it: base-class code must not depend on child-only methods.
        LGCP_Model.simulate() reaches this through _sim_cox.
        """
        counts = np.asarray(counts)
        nz = counts > 0
        pts = geo_df[nz].sample_points(size=counts[nz], rng=rng).explode(index_parts=False)
        if len(pts) != counts.sum():
            raise RuntimeError(f"sample_points returned {len(pts)} points for "
                               f"{counts.sum()} requested; refusing to truncate")
        return np.stack((pts.x.values, pts.y.values), axis=1)

    def _decode_field_parameters(self, parameters):
        """Decode VAE latents to fields in a simulation parameter dict, in place.

        Same decode functions as the model layer (decode_fields.py). Fills
        f_t / f_a / f_xy from z_temporal / z_seasonal / z_spatial when the
        decoded field is absent, and b_0 from w for covariate models.
        Explicit decode remains useful because callers may pass z-only dicts
        (e.g. thinned sample-site subsets, or Predictive outputs that omit
        some deterministics depending on return_sites); do not assume NumPyro
        0.15 MCMC categorically omits every deterministic. Shared by
        Hawkes_Model.simulate() and LGCP_Model.simulate().
        """
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
        return parameters

    def _filter_simulated_points_to_domain(self, points):
        """Keep each simulated point once via authoritative domain membership.

        Polygon domains filter against ``PreparedDomain.union_geometry``
        (``covers``, including boundary). Rectangle/array domains filter
        against the prepared rectangle ``bounds`` with the same closed
        interval semantics -- never ``sjoin`` against the computational
        grid (``self.A``), which duplicates points on internal cell edges
        and vertices.
        """
        cols = ['X', 'Y', 'T', 'geometry']
        domain = self.prepared_domain
        if domain.is_polygon and domain.union_geometry is None:
            raise ValueError(
                "PreparedDomain.union_geometry is required for polygon "
                "domain simulation filtering")
        xs = np.asarray(points["X"], dtype=float)
        ys = np.asarray(points["Y"], dtype=float)
        mask = np.fromiter(
            (domain.covers_xy(x, y) for x, y in zip(xs, ys)),
            dtype=bool,
            count=len(points),
        )
        return points.loc[mask, cols]

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
            When provided, rng drives EVERY draw (Poisson count, inverse-CDF
            uniforms, cell multinomial, and GeoSeries.sample_points): one Generator
            gives a fully reproducible draw. rng=None falls back to np.random (plus
            geopandas' own unseeded sample_points), preserving legacy behavior.

        Known approximations (all vanish for a rectangle domain -- use a rectangle A for
        simulate-and-recover / SBC):
          - a finite spatial_window IS mirrored in the offspring thinning (real-unit
            box, within_real_box_window), matching the clipped compensator exactly, so
            finite-ws configurations are inside the calibration-supported regime.
          - For a GeoDataFrame domain in rectangle excitation mode, the excitation
            compensator integrates over the bounding rectangle A_. In polygon
            excitation mode the Hermite mass table charges the domain union.
            The Cox background itself is exact on the polygon since 3c-1: boundary
            cells are charged and sampled on the clipped support C_c ∩ A, so
            simulate()'s A-filter no longer discards background points.
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
            # test_sim_cox_array_domain_support_regression). 3c-1 (D-6):
            # the rows are the CLIPPED support cells C_c ∩ A -- the same
            # object whose areas built the integration arrays -- so
            # background points outside A are never drawn (full cells on
            # rectangle domains, where this is the legacy geometry).
            fi = np.asarray(self.args['integration_field_indices'])
            geo_df = (self.prepared_partitions.support_cells
                      .set_index('comp_grid_id').loc[fi])
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
        gen = rng if rng is not None else np.random
        N = gen.poisson(Ig * Ih)
        if N == 0:
            return np.empty((0, 3))
        # inverse-CDF on segment MASS (g_j * len_j), then uniform within the chosen segment
        w = g * seg_len
        cdf = np.cumsum(w) / w.sum()
        bins = np.searchsorted(cdf, gen.uniform(size=N), side='right')
        times = seg_lo[bins] + gen.uniform(size=N) * seg_len[bins]
        cells = gen.choice(len(h_mass), size=N, p=h_mass / h_mass.sum())
        counts = np.bincount(cells, minlength=len(h_mass))
        xy = self._sample_cells(geo_df, counts, rng=rng)
        # times and locations are independent given the factorization: pairing is arbitrary
        return np.column_stack((xy, times))


    def set_window(self, window=_UNSET, spatial_window=_UNSET):
        """Update computational cutoffs (internal temporal / real spatial).

        Omitted arguments (the private ``_UNSET`` default) leave the current
        value unchanged. Explicit ``None`` restores the untruncated default
        (temporal: full observation horizon ``T_INTERNAL``; spatial: no
        computational cutoff). ``set_window()`` with no arguments is a no-op.
        Candidate pairs are prepared locally and committed only on success.
        """
        if window is _UNSET and spatial_window is _UNSET:
            return

        if window is _UNSET:
            window = self.args['window']
        elif window is None:
            window = float(T_INTERNAL)
        else:
            window = float(window)
            if not (math.isfinite(window) and window > 0):
                raise ValueError(f"window must be finite and > 0; got {window}")

        if spatial_window is _UNSET:
            spatial_window = self.args.get('spatial_window')
        elif spatial_window is not None:
            spatial_window = float(spatial_window)
            if not (math.isfinite(spatial_window) and spatial_window > 0):
                raise ValueError(
                    f"spatial_window must be finite and > 0; got {spatial_window}")

        coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
            self.args['t_events'],
            self.args['xy_events'][0],
            self.args['xy_events'][1],
            window=window,
            spatial_window=spatial_window,
            axis_scales=np.asarray(self.args['axis_scales'])
        )

        self.args['window'] = window
        self.args['spatial_window'] = spatial_window
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
    def __init__(self, data, A, T, cox_background='cox',
                 temporal_trig=Temporal_Exponential,
                 spatial_trig=Spatial_Symmetric_Gaussian,
                 window=None, spatial_window=None,
                 temporal_cutoff_days=None,
                 temporal_cutoff_tol=None,
                 spatial_cutoff_tol=None,
                 cutoff_tol=None,
                 design_mean_lag_days=None,
                 design_sigma=None,
                 excitation_support=None,
                 min_sigma=None, max_sigma=None,
                 mass_table=None,
                 **kwargs):
        r"""
        Spatiotemporal Point Process Model given by,

        $$\lambda(t,s) = \mu(s,t) + \sum_{i:t_i<t}{\alpha f(t-t_i;\beta) \varphi(s-s_i;\sigma^2)}$$

        where $f$ is defined by spatial_trig, $\\varphi$ is defined by spatial_trig. If cox_background is true, $\mu$ is given by

        $$\mu(s,t) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes.
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(15,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$

        Otherwise, the $\mu$ is given by

        $$\mu(s,t) = exp(a_0 + X(s)w)$$

        The data is rescaled to a 1x1 spatial grid and an internal length-50 time
        window. Spatial trigger quantities (sigmax_2, spatial_window) are REAL-unit
        by contract. The preferred public temporal scale prior is ``mean_lag_days``
        (days); it is converted to the internal sample site ``beta``. Finite
        ``window`` / ``spatial_window`` values are computational cutoffs of
        infinite-support kernels (D-13/D-14), applied consistently to pairs,
        compensator, and simulator, with omitted mass in
        ``cutoff_provenance`` (OP-6).

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
            ``mean_lag_days``, ``temporal_cutoff_tol``, ``design_mean_lag_days``,
            and the temporal leg of shared ``cutoff_tol`` are valid only when
            this is exactly ``Temporal_Exponential``. ``Temporal_Power_Law`` and
            custom temporal triggers accept explicit ``window`` /
            ``temporal_cutoff_days`` and their own priors (e.g. ``beta`` /
            ``gamma``); exponential omitted mass remains ``None``.
        spatial_trig: class Trigger
            an implementation of Trigger to parameterize the spatial triggering mechanism.
            ``sigmax_2``, ``min_sigma`` / ``max_sigma``, ``spatial_cutoff_tol``,
            ``design_sigma``, and the spatial leg of shared ``cutoff_tol`` are
            valid only when this is exactly ``Spatial_Symmetric_Gaussian``.
            Custom rectangle spatial triggers use priors named by
            ``get_par_names()`` and an explicit ``spatial_window``; Gaussian
            omitted mass remains ``None``. Polygon mode still requires the
            exact ``Spatial_Symmetric_Gaussian`` class.
        window: float, optional
            Legacy temporal computational cutoff in INTERNAL time units
            ([0, 50]). Prefer ``temporal_cutoff_days``. Mutually exclusive with
            ``temporal_cutoff_days``. When omitted together with tolerance kwargs,
            defaults to the full horizon (no temporal cutoff).
        spatial_window: float, optional
            Spatial computational cutoff half-width, a REAL length: pairs kept
            iff max(|dx|, |dy|) <= spatial_window (per-axis square, D-21). Same
            cutoff applies to the compensator and offspring thinning. Prefer
            ``spatial_cutoff_tol`` with ``design_sigma`` for tolerance-based
            selection (default tol subsumes spatial_window >= 4*sigma). Defaults
            to None (no spatial cutoff).
        temporal_cutoff_days: float, optional
            Preferred physical temporal cutoff in real days (converted to
            internal units for the three legs). Wins over
            ``temporal_cutoff_tol`` when both are supplied (OP-6).
        temporal_cutoff_tol: float, optional
            Target omitted temporal mass eps_t; requires
            ``design_mean_lag_days``. Default package tol is exp(-5)
            (the old window >= 5*beta rule).
        spatial_cutoff_tol: float, optional
            Target omitted spatial mass eps_s for the square cutoff; requires
            ``design_sigma``. Default package tol is 1-erf(4/sqrt(2))**2
            (the old spatial_window >= 4*sigma rule).
        cutoff_tol: float, optional
            Shared default for temporal and spatial tolerances when the
            axis-specific tol is omitted.
        design_mean_lag_days: float, optional
            Design temporal scale (days) used to convert eps_t -> window at
            construction (beta is latent).
        design_sigma: float, optional
            Design spatial scale (real length) used to convert eps_s ->
            spatial_window at construction (sigma is latent).
        excitation_support: {'rectangle', 'polygon'}, optional
            Phase 3d support mode. Controls both offspring parenting eligibility and
            the excitation compensator charge region (D-18). Required on
            nonrectangular (GeoDataFrame) domains; defaults to 'rectangle' on
            axis-aligned array domains. 'polygon' discards offspring outside A
            before parenting and charges Hermite polygon mass; 'rectangle' retains
            the bounding-rectangle approximation.
        min_sigma: float, optional
            Lower bound on sigma = sqrt(sigmax_2) in domain CRS units. Required
            (explicit, finite, positive) for polygon mode. In rectangle mode,
            omit together with max_sigma to leave the prior unchanged.
        max_sigma: float, optional
            Upper bound on sigma in domain CRS units. Polygon mode defaults to
            5 km converted via the projected CRS when omitted. In rectangle mode,
            omit together with min_sigma to leave the prior unchanged. When
            bounds are active the sigmax_2 prior is truncated to
            [min_sigma**2, max_sigma**2] with interval support for NUTS/SVI.
        mass_table: PolygonMassTable, optional
            Prepared Hermite table from
            ``bstpp.polygon_mass.prepare_polygon_mass_table`` (required when
            ``excitation_support='polygon'``). Validated against prepared
            domain/events/windows after construction; never built
            synchronously here. Legacy tables lacking the current schema or
            exact event-hash metadata are rejected as incompatible.
        sp_var_mu: float
            Fixed log-amplitude multiplier applied to the spatial VAE decoder output; see
            Point_Process_Model for calibration guidance. Default 2.0.
        kwargs: dict
            parameters from Point_Process_Model, plus preferred prior
            ``mean_lag_days`` (numpyro Distribution in days) which is converted
            to the internal ``beta`` sample site. Legacy ``beta=`` (internal)
            remains accepted; do not pass both.
        """
        # Earliest valid preflight: explicit polygon mode without a table must
        # fail before expensive Point_Process_Model data/decoder preparation.
        # Full compatibility validation still runs after canonical domain/event
        # preparation when a table is supplied.
        if excitation_support == "polygon" and mass_table is None:
            raise ValueError(
                "Polygon excitation_support requires a prepared mass_table "
                "from bstpp.polygon_mass.prepare_polygon_mass_table(...); "
                "the constructor does not build Hermite tables.")

        self.model = spatiotemporal_hawkes_model
        if cox_background:
            name = 'cox_hawkes'
        else:
            name = 'hawkes'
        self._spatial_trig_cls = spatial_trig
        self._temporal_trig_cls = temporal_trig
        self._excitation_support_arg = excitation_support
        self._min_sigma_arg = min_sigma
        self._max_sigma_arg = max_sigma
        self._mass_table_arg = mass_table

        # Phase 3e: pop public-days prior before the base constructor treats
        # every Distribution kwarg as a sample-site prior name.
        mean_lag_days_prior = kwargs.pop('mean_lag_days', None)

        # Exact-type capability gates before expensive base construction.
        _reject_unsupported_kernel_kwargs(
            temporal_trig=temporal_trig,
            spatial_trig=spatial_trig,
            mean_lag_days_prior=mean_lag_days_prior,
            temporal_cutoff_tol=temporal_cutoff_tol,
            spatial_cutoff_tol=spatial_cutoff_tol,
            cutoff_tol=cutoff_tol,
            design_mean_lag_days=design_mean_lag_days,
            design_sigma=design_sigma,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
        )

        super().__init__(name, data, A, T, **kwargs)

        temporal_ok = _is_exact_temporal_exponential(temporal_trig)
        spatial_ok = _is_exact_spatial_gaussian(spatial_trig)

        if mean_lag_days_prior is not None:
            if 'beta' in self.args['priors']:
                raise ValueError(
                    "Pass mean_lag_days or beta, not both "
                    "(mean_lag_days is the public days-unit prior; beta is "
                    "the legacy internal-unit sample site).")
            self.args['priors']['beta'] = scale_temporal_prior_to_internal(
                mean_lag_days_prior, self.T)

        mode = resolve_excitation_support_mode(
            is_polygon_domain=self.prepared_domain.is_polygon,
            excitation_support=excitation_support)

        # Polygon Hermite mass integrates Spatial_Symmetric_Gaussian only.
        # Exact-type gate (not isinstance): a subclass may override the event
        # kernel while the compensator remains Gaussian -- an invalid likelihood.
        # Rectangle mode retains the existing custom-trigger interface.
        if mode == "polygon" and spatial_trig is not Spatial_Symmetric_Gaussian:
            raise TypeError(
                "Polygon excitation_support currently requires the exact "
                "Spatial_Symmetric_Gaussian spatial trigger (Hermite mass "
                "backend is Gaussian-only). Got "
                f"{getattr(spatial_trig, '__name__', repr(spatial_trig))}. "
                "Use excitation_support='rectangle' for a custom spatial "
                "trigger, or supply Spatial_Symmetric_Gaussian.")

        # Gaussian-only sigma prior / truncation. Custom rectangle spatial
        # triggers use priors named by get_par_names() instead.
        lo, hi, _ = resolve_sigma_bounds(
            mode=mode, min_sigma=min_sigma, max_sigma=max_sigma,
            crs=self.prepared_domain.crs)
        if spatial_ok:
            if 'sigmax_2' not in self.args['priors']:
                raise ValueError(
                    "Hawkes_Model requires a user-supplied sigmax_2 prior "
                    "(numpyro Distribution in squared real units) when "
                    "spatial_trig is Spatial_Symmetric_Gaussian.")
            if lo is not None and hi is not None:
                self.args['priors']['sigmax_2'] = truncate_sigmax_2_prior(
                    self.args['priors']['sigmax_2'], lo, hi)

        # Phase 3f WP1: NumericalConfig behind the args adapter (no public
        # signature change). Polygon builder settings come from the supplied
        # table (authoritative h_panel / gl_order); rectangle uses defaults.
        if mode == "polygon":
            numerical_config = NumericalConfig.create(
                panel_h_m=mass_table.h_panel,
                gl_order=mass_table.gl_order,
                support_mode=mode,
                min_sigma=lo,
                max_sigma=hi,
            )
        else:
            numerical_config = NumericalConfig.create(
                support_mode=mode,
                min_sigma=lo,
                max_sigma=hi,
            )
        self.numerical_config = numerical_config
        self.args["numerical_config"] = numerical_config

        self.args['t_trig'] = temporal_trig(self.args['priors'])
        self.args['sp_trig'] = spatial_trig(self.args['priors'])
        for trig, axis in (
                (self.args['t_trig'], "temporal"),
                (self.args['sp_trig'], "spatial")):
            for pname in trig.get_par_names():
                if pname not in self.args['priors']:
                    raise ValueError(
                        f"{axis} trigger "
                        f"{type(trig).__name__} requires a prior named "
                        f"{pname!r} (from get_par_names()).")

        # OP-6: resolve computational cutoffs (physical wins over tolerance).
        # Legacy path: bare window=None / spatial_window=None keeps the
        # untruncated defaults so fixed-cutoff pins stay bit-stable.
        # Unsupported kernels never receive tolerance / design-scale args
        # (gated above); physical cutoffs record selection but leave
        # kernel-specific omitted mass as None.
        self._design_mean_lag_days = (
            design_mean_lag_days if temporal_ok else None)
        self._design_sigma = design_sigma if spatial_ok else None
        cutoff_prov = resolve_computational_cutoffs(
            horizon_days=float(self.T),
            temporal_cutoff_days=temporal_cutoff_days,
            window_internal=float(window) if window is not None else None,
            spatial_window=(
                float(spatial_window) if spatial_window is not None else None),
            temporal_cutoff_tol=(
                temporal_cutoff_tol if temporal_ok else None),
            spatial_cutoff_tol=spatial_cutoff_tol if spatial_ok else None,
            cutoff_tol=(
                cutoff_tol if (temporal_ok and spatial_ok) else None),
            design_mean_lag_days=self._design_mean_lag_days,
            design_sigma=self._design_sigma,
        )
        # Install realized windows first. Public set_window re-resolves only
        # axes that are explicitly supplied; omitted (_UNSET) axes keep their
        # construction provenance verbatim. Restore the OP-6 construction
        # record (tolerance vs physical selection) afterward.
        self.set_window(
            cutoff_prov.temporal.window_internal,
            cutoff_prov.spatial.spatial_window,
            mass_table=mass_table if mode == "polygon" else _UNSET,
        )
        self.cutoff_provenance = cutoff_prov

    @staticmethod
    def _xy_events_to_real(args):
        """Event locations in real CRS units for mass-table / support identity.

        Prefers the float64 ``xy_events_real`` snapshot taken at ingestion
        (exact row-order identity for PolygonMassTable). Falls back to the
        affine roundtrip through ``xy_events`` / ``axis_scales`` only when
        that snapshot is absent.
        """
        if "xy_events_real" in args:
            xy_real = np.asarray(args["xy_events_real"], dtype=np.float64)
            return xy_real[0], xy_real[1]
        A_ = np.asarray(args['A_'], dtype=float)
        scales = np.asarray(args['axis_scales'], dtype=float)
        xy = np.asarray(args['xy_events'])
        return (xy[0] * scales[0] + A_[0, 0],
                xy[1] * scales[1] + A_[1, 0])

    def _event_xy_real(self):
        """Observed event locations in real CRS units (table / support order)."""
        return self._xy_events_to_real(self.args)

    def _excitation_support_for_events(self, event_x_real, event_y_real,
                                       *, mass_table=None):
        """Build excitation support for an alternate event set (held-out scoring).

        Reuses the fitted model's domain, spatial cutoff, and sigma bounds via
        the same ``build_excitation_support`` path as training install. Does not
        mutate ``self.args`` or the training support object. Polygon mode
        hard-requires a caller-supplied ``mass_table`` prepared for these
        events; silent ``prepare_polygon_mass_table`` is not allowed here.
        """
        mode = resolve_excitation_support_mode(
            is_polygon_domain=self.prepared_domain.is_polygon,
            excitation_support=self._excitation_support_arg)
        domain_gdf = (self.prepared_domain.domain
                      if self.prepared_domain.is_polygon else None)
        event_x_real = np.asarray(event_x_real, dtype=float)
        event_y_real = np.asarray(event_y_real, dtype=float)
        if mode == "polygon":
            if mass_table is None:
                raise ValueError(
                    "Polygon held-out excitation support requires mass_table= "
                    "from bstpp.polygon_mass.prepare_polygon_mass_table(...) "
                    "built for the held-out events; silent rebuild is not "
                    "allowed.")
        elif mass_table is not None:
            raise ValueError(
                "mass_table is only accepted for polygon excitation_support")
        return build_excitation_support(
            mode=mode,
            bounds=self.args['A_'],
            domain_gdf=domain_gdf,
            is_polygon_domain=self.prepared_domain.is_polygon,
            crs=self.prepared_domain.crs,
            spatial_window=self.args.get('spatial_window'),
            min_sigma=self._min_sigma_arg,
            max_sigma=self._max_sigma_arg,
            event_x_real=event_x_real,
            event_y_real=event_y_real,
            mass_table=mass_table,
            union_geometry=self.prepared_domain.union_geometry,
            numerical_config=self.args.get("numerical_config"),
        )

    def set_window(self, window=_UNSET, spatial_window=_UNSET, *,
                   mass_table=_UNSET):
        """Update computational cutoffs transactionally.

        Omitted arguments (private ``_UNSET``) leave the current value
        unchanged, including the existing ``TemporalCutoffRecord`` /
        ``SpatialCutoffRecord`` verbatim. Explicit ``window=None`` restores
        the full observation-horizon setting (``T_INTERNAL``); explicit
        ``spatial_window=None`` removes the spatial computational cutoff.
        ``set_window()`` with no arguments is a true no-op.

        Polygon mode: a temporal-only change (unchanged realized
        spatial_window) reuses the installed mass table. A changed
        spatial_window requires ``mass_table=`` from
        ``prepare_polygon_mass_table`` matching the new window; silent
        Hermite rebuild is not allowed. Candidate table, pairs, support,
        and provenance are prepared locally and committed only after
        validation succeeds. Design scales from construction are retained
        so realized omission on an explicitly updated axis is recomputed
        honestly.
        """
        if (window is _UNSET and spatial_window is _UNSET
                and mass_table is _UNSET):
            return

        if window is _UNSET:
            new_window = float(self.args['window'])
            temporal_clear = False
        elif window is None:
            new_window = float(T_INTERNAL)
            temporal_clear = True
        else:
            new_window = float(window)
            temporal_clear = False
            if not (math.isfinite(new_window) and new_window > 0):
                raise ValueError(
                    f"window must be finite and > 0; got {new_window}")

        if spatial_window is _UNSET:
            new_sw = self.args.get('spatial_window')
        elif spatial_window is None:
            new_sw = None
        else:
            new_sw = float(spatial_window)
            if not (math.isfinite(new_sw) and new_sw > 0):
                raise ValueError(
                    f"spatial_window must be finite and > 0; got {new_sw}")

        def _sw_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return float(a) == float(b)

        prev_sw = self.args.get('spatial_window')
        spatial_changed = not _sw_equal(prev_sw, new_sw)

        mode = resolve_excitation_support_mode(
            is_polygon_domain=self.prepared_domain.is_polygon,
            excitation_support=self._excitation_support_arg)

        table_arg = None
        if mode == "polygon":
            if spatial_changed:
                if mass_table is _UNSET or mass_table is None:
                    raise ValueError(
                        "Polygon set_window with a changed spatial_window "
                        "requires mass_table= from "
                        "bstpp.polygon_mass.prepare_polygon_mass_table(...) "
                        "matching the new spatial_window; silent rebuild is "
                        "not allowed.")
                table_arg = mass_table
            else:
                if mass_table is not _UNSET and mass_table is not None:
                    table_arg = mass_table
                elif getattr(self, "excitation_support", None) is not None:
                    table_arg = self.excitation_support.mass_table
                else:
                    table_arg = self._mass_table_arg
                if table_arg is None:
                    raise ValueError(
                        "Polygon set_window requires an installed or supplied "
                        "mass_table from prepare_polygon_mass_table(...).")
        elif mass_table is not _UNSET and mass_table is not None:
            raise ValueError(
                "mass_table is only accepted for polygon excitation_support")

        # Prospective NumericalConfig for a newly supplied mass_table (WP1.4b).
        # Built in local state before install; committed only with support.
        # Mode, CRS, and σ constructor args are stable across set_window.
        new_numerical_config = self.args.get("numerical_config")
        installing_new_table = (
            mode == "polygon"
            and mass_table is not _UNSET
            and mass_table is not None
        )
        if installing_new_table:
            lo_cfg, hi_cfg, _ = resolve_sigma_bounds(
                mode=mode,
                min_sigma=self._min_sigma_arg,
                max_sigma=self._max_sigma_arg,
                crs=self.prepared_domain.crs,
            )
            prev_cfg = self.args["numerical_config"]
            # Panel-ratio prefilter before rebuilding NumericalConfig so a
            # too-coarse supplied table still raises
            # assert_polygon_mass_table_budget's message before
            # NumericalConfigError (set_window error-precedence CF).
            assert_polygon_mass_table_budget(
                table_arg,
                sigma_min=lo_cfg,
                numerical_config=prev_cfg,
            )
            new_numerical_config = NumericalConfig.create(
                panel_h_m=table_arg.h_panel,
                gl_order=table_arg.gl_order,
                support_mode=mode,
                min_sigma=lo_cfg,
                max_sigma=hi_cfg,
                default_temporal_tol=prev_cfg.default_temporal_tol,
                default_spatial_tol=prev_cfg.default_spatial_tol,
            )

        # Prepare the full replacement state before mutating self.
        coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
            self.args['t_events'],
            self.args['xy_events'][0],
            self.args['xy_events'][1],
            window=new_window,
            spatial_window=new_sw,
            axis_scales=np.asarray(self.args['axis_scales'])
        )
        design_lag = getattr(self, "_design_mean_lag_days", None)
        design_sigma = getattr(self, "_design_sigma", None)
        # Per-axis provenance: an _UNSET axis keeps its existing record
        # verbatim; only an explicitly supplied axis is re-resolved.
        if window is _UNSET:
            temporal_record = self.cutoff_provenance.temporal
        elif temporal_clear:
            design_beta = (
                days_to_internal(float(design_lag), float(self.T))
                if design_lag is not None else None)
            temporal_record = TemporalCutoffRecord(
                selection="default_untruncated",
                window_internal=float(T_INTERNAL),
                window_days=float(self.T),
                requested_tol=None,
                design_mean_lag_days=(
                    float(design_lag) if design_lag is not None else None),
                design_beta_internal=design_beta,
                omitted_mass_at_design=None,
            )
        else:
            design_beta = (
                days_to_internal(float(design_lag), float(self.T))
                if design_lag is not None else None)
            omitted = (
                temporal_omitted_mass(new_window, design_beta)
                if design_beta is not None else None)
            temporal_record = TemporalCutoffRecord(
                selection="physical",
                window_internal=new_window,
                window_days=internal_to_days(new_window, float(self.T)),
                requested_tol=None,
                design_mean_lag_days=(
                    float(design_lag) if design_lag is not None else None),
                design_beta_internal=design_beta,
                omitted_mass_at_design=omitted,
            )

        if spatial_window is _UNSET:
            spatial_record = self.cutoff_provenance.spatial
        elif new_sw is None:
            spatial_record = SpatialCutoffRecord(
                selection="default_untruncated",
                spatial_window=None,
                requested_tol=None,
                design_sigma=(
                    float(design_sigma) if design_sigma is not None else None),
                omitted_mass_at_design=None,
            )
        else:
            omitted = (
                spatial_omitted_mass(new_sw, float(design_sigma))
                if design_sigma is not None else None)
            spatial_record = SpatialCutoffRecord(
                selection="physical",
                spatial_window=new_sw,
                requested_tol=None,
                design_sigma=(
                    float(design_sigma) if design_sigma is not None else None),
                omitted_mass_at_design=omitted,
            )
        cutoff_provenance = CutoffProvenance(
            temporal=temporal_record, spatial=spatial_record)

        x_real, y_real = self._event_xy_real()
        domain_gdf = (self.prepared_domain.domain
                      if self.prepared_domain.is_polygon else None)
        support = build_excitation_support(
            mode=mode,
            bounds=self.args['A_'],
            domain_gdf=domain_gdf,
            is_polygon_domain=self.prepared_domain.is_polygon,
            crs=self.prepared_domain.crs,
            spatial_window=new_sw,
            min_sigma=self._min_sigma_arg,
            max_sigma=self._max_sigma_arg,
            event_x_real=x_real,
            event_y_real=y_real,
            mass_table=table_arg,
            union_geometry=self.prepared_domain.union_geometry,
            numerical_config=new_numerical_config,
        )

        # Atomic commit: windows, pairs, support, provenance, and
        # NumericalConfig together (D-33 / WP1.4b).
        self.args['window'] = new_window
        self.args['spatial_window'] = new_sw
        self.args['coords'] = coords
        self.args['t_vals'] = t_vals
        self.args['x_vals'] = x_vals
        self.args['y_vals'] = y_vals
        self._mass_table_arg = None
        self.args['excitation_support'] = support
        self.excitation_support = support
        self.excitation_provenance = dict(support.provenance)
        self.cutoff_provenance = cutoff_provenance
        self.numerical_config = new_numerical_config
        self.args["numerical_config"] = new_numerical_config

    def export_polygon_mass_table(self, path):
        """Export the polygon-mode Hermite table for later refit reload."""
        support = self.args.get('excitation_support')
        if support is None or support.mass_table is None:
            raise ValueError(
                "No polygon mass table to export (rectangle mode, or support "
                "not installed).")
        return support.mass_table.export_npz(path)

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


    def _sim_hawkes_bg(self, parameters, rng=None):
        """Constant / covariate background via per-cell Poisson superposition.

        Cell rates are background_masses(...) -- the integrand of the
        background compensator atoms, whose sum the unit tests pin against
        constant/covariate_background_integral -- so the simulator and
        likelihood share one background-mass expression (no runtime assert:
        one shared computation plus tests, per review). Superposing per-cell
        Poisson draws is distributionally identical to Poisson(total) +
        multinomial. RNG: when rng is provided it drives EVERY draw here
        (Poisson counts, sample_points, uniform times) -- the historical
        unseeded-sample_points quirk is resolved; rng=None preserves the old
        np.random + unseeded behavior for legacy callers.
        """
        gen = rng if rng is not None else np.random
        a_0 = float(parameters['a_0'])
        T_int = self.args['T']
        if 'spatial_cov' in self.args:
            # 3c-3 (D-7): the clipped covariate support C_c ∩ A -- the same
            # object whose areas are cov_area -- so background points
            # outside A are never drawn (identical to spatial_cov when the
            # layer lies within A).
            geo_df = self.prepared_partitions.cov_support
            mu = np.exp(a_0 + np.asarray(parameters['b_0']))
            areas = np.asarray(self.args['cov_area'])
        elif self.prepared_domain.is_polygon:
            # D-30: sample from the canonical union geometry with A_area, not
            # raw overlapping domain rows (row areas double-count overlap).
            union = self.prepared_domain.union_geometry
            if union is None:
                raise ValueError(
                    "PreparedDomain.union_geometry is required for polygon "
                    "plain-Hawkes background sampling")
            geo_df = gpd.GeoDataFrame(
                {"geometry": [union]},
                crs=self.prepared_domain.crs,
            )
            mu = np.exp(a_0)
            areas = np.asarray([float(self.args['A_area'])], dtype=float)
        else:
            geo_df = self.A
            A_ = self.args['A_']
            mu = np.exp(a_0)
            areas = (geo_df.area / ((A_[0, 1]-A_[0, 0]) * (A_[1, 1]-A_[1, 0]))).values
        cell_rates = np.asarray(background_masses(mu, areas, T_int))
        num = gen.poisson(cell_rates)
        xy = self._sample_cells(geo_df, num, rng=rng)
        return np.column_stack((xy, gen.uniform(0, T_int, size=len(xy))))

    def _sim_offspring(self, bg, par, rng=None):
        # One Generator drives every draw when provided (offspring counts and
        # both trigger simulations); rng=None falls back to np.random so
        # legacy call sites and user-defined triggers keep working. Legacy
        # THIRD-PARTY triggers with the old simulate_trigger(pars) signature
        # are called without rng (detected by signature inspection) and stay
        # on np.random; both in-repo trigger classes are new-style.
        gen = rng if rng is not None else np.random

        # New-style vs old-style triggers are told apart by SIGNATURE
        # inspection, once per call -- not by a per-draw ``except TypeError``
        # fallback, which misclassified any TypeError raised INSIDE a
        # new-style trigger as an old signature and silently re-executed it
        # without rng (masked user bug + quietly abandoned reproducibility).
        sp_trig, t_trig = self.args['sp_trig'], self.args['t_trig']
        sp_accepts = accepts_rng_kwarg(sp_trig.simulate_trigger)
        t_accepts = accepts_rng_kwarg(t_trig.simulate_trigger)

        def _trig_draw(trig, accepts):
            if accepts:
                return trig.simulate_trigger(par, rng=rng)
            # Old simulate_trigger(pars) signature (legacy third-party
            # triggers): stays on np.random, the documented legacy behavior.
            return trig.simulate_trigger(par)

        i = 0
        while i < len(bg):
            for j in range(gen.poisson(lam=par['alpha'])):
                #simulate trigger: REAL-unit contract -- the spatial trigger
                #draws the offspring displacement directly in real coordinate
                #units, matching the real-unit kernel the likelihood evaluates
                #(the historical internal draw * per-axis box-span rescale is
                #gone; that rescale WAS the aspect-ratio anisotropy defect).
                sp_dif = _trig_draw(sp_trig, sp_accepts)
                t_dif = [_trig_draw(t_trig, t_accepts)]
                # window-consistent thinning: match the truncated-kernel likelihood
                # (Poisson(alpha) parents thinned by F(w) => expected offspring alpha*F(w))
                if t_dif[0] > self.args['window']:
                    continue
                # spatial window: the SAME real-unit box predicate the pair set
                # uses and the compensator integrates (single-sourced; the
                # spatial draw is already in real units under the contract)
                sw = self.args.get('spatial_window')
                if sw is not None and not within_real_box_window(
                        sp_dif[0], sp_dif[1], sw):
                    continue
                cand = bg[i] + np.append(sp_dif, t_dif)
                # D-18: parenting eligibility region ≡ compensator charge
                # region, via the single excitation_support object.
                support = self.args.get('excitation_support')
                if support is not None:
                    if not support.candidate_in_support(
                            float(cand[0]), float(cand[1])):
                        continue
                else:
                    # Legacy fallback (should not occur for Hawkes after 3d)
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
        rng: numpy.random.Generator
            Required. One Generator drives every draw (background and
            offspring paths); identically seeded fresh Generators give
            byte-identical simulations. The legacy ``rng=None`` →
            ``np.random`` fallback has been removed (caller-state ownership).
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns (real units)
                simulated data
        """
        if rng is None:
            raise ValueError(
                "simulate requires an explicit numpy.random.Generator via "
                "rng=; the legacy np.random fallback has been removed "
                "(caller-state ownership / pre-3f B2).")
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        # Shared decode of z latents -> fields (one copy, on the base class).
        parameters = self._decode_field_parameters(parameters)

        if self.args['model'] == 'cox_hawkes':
            bg = self._sim_cox(parameters, rng=rng)
        else:
            bg = self._sim_hawkes_bg(parameters, rng=rng)
        sample = self._sim_offspring(bg, parameters, rng=rng)
        #filter out offspring after cutoff
        sample = sample[sample.T[2]<self.args['T']]
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        #filter to time window
        points['T'] = (points['T']*self.T/self.args['T'])
        #filter to spatial domain (canonical union for polygon domains)
        return self._filter_simulated_points_to_domain(points)


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
        # Phase 3f WP1: rectangle NumericalConfig behind the adapter (LGCP has
        # no excitation support; defaults only).
        numerical_config = NumericalConfig.create(support_mode="rectangle")
        self.numerical_config = numerical_config
        self.args["numerical_config"] = numerical_config

    def set_window(self, window=_UNSET, spatial_window=_UNSET, *,
                   mass_table=_UNSET):
        """LGCP has no excitation cutoffs; reject all set_window calls.

        Computational excitation windows and pair construction apply only to
        Hawkes / Cox–Hawkes models. Zero-argument and explicit calls raise the
        same clear error and do not populate unused Hawkes state.
        """
        raise NotImplementedError(
            "LGCP_Model.set_window is not supported: computational excitation "
            "cutoffs and pair windows apply only to Hawkes / Cox-Hawkes "
            "models. LGCP has no excitation pairs or cutoff provenance.")

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
        rng: numpy.random.Generator
            Required. One Generator drives every draw (see _sim_cox);
            identically seeded fresh Generators give byte-identical
            simulations. The legacy ``rng=None`` → ``np.random`` fallback
            has been removed (caller-state ownership).
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns (real units)
                simulated data, clipped to the domain A
        """
        if rng is None:
            raise ValueError(
                "simulate requires an explicit numpy.random.Generator via "
                "rng=; the legacy np.random fallback has been removed "
                "(caller-state ownership / pre-3f B2).")
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        # Shared decode of z latents -> fields (one copy, on the base class);
        # mcmc.get_samples() dicts carry z sites only, so without this the
        # LGCP path raised KeyError('f_t') on any z-only parameter dict.
        parameters = self._decode_field_parameters(parameters)
        sample = self._sim_cox(parameters, rng=rng)
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        points['T'] = (points['T']*self.T/self.args['T'])
        # Clip to the true domain polygon, mirroring Hawkes_Model.simulate():
        # _sim_cox samples boundary cells over the FULL cell and documents that
        # "simulate()'s A-filter" clips them -- a promise this method previously
        # did not keep (returned points could fall outside a non-rectangular A).
        # Polygon domains use PreparedDomain.union_geometry (not row-wise A).
        return self._filter_simulated_points_to_domain(points)
