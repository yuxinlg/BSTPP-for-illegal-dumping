import os
import time
import jax.numpy as jnp
from jax.example_libraries.optimizers import inverse_time_decay

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer import Trace_ELBO, SVI, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal
from .vae_functions import (vae_decoder_temporal, vae_decoder_seasonal,
                             vae_decoder_spatial)
from .likelihood import (aggregate_pair_trigger_values,
                         constant_background_integral,
                         covariate_background_integral,
                         rectangular_excitation_compensator,
                         seasonal_time_integral,
                         spatial_refinement_integral)


def spatiotemporal_hawkes_model(args):
    # Model for Hawkes and Cox Hawkes

    coords = args['coords']
    t_vals = args['t_vals']
    x_vals = args['x_vals']
    y_vals = args['y_vals']
    t_events=args["t_events"]
    xy_events=args["xy_events"]

    # Do NOT recompute aligned_difference_pairs here; use precomputed values from args

    if args['model'] == 'hawkes':
      a_0 = numpyro.sample("a_0", args['priors']['a_0'])
      if 'spatial_cov' in args:
        w = numpyro.sample("w", args['priors']['w'])
        b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)
        # mu_xyt is defined per covariate cell; events index into it below.
        mu_xyt = numpyro.deterministic("mu_xyt", jnp.exp(a_0 + b_0))            # (n_cells,)
        # Background integral contracts cells against their areas.
        Itot_txy_back = numpyro.deterministic(
            "Itot_txy_back",
            covariate_background_integral(mu_xyt, args['cov_area'], args['T'])
        )
        # Use precomputed indices to evaluate the per-cell rate at each event.
        mu_xyt_events = mu_xyt[args['cov_ind']]                                  # (n_events,)
      else:
        b_0 = 0
        mu_xyt = numpyro.deterministic("mu_xyt", jnp.exp(a_0 + b_0))
        Itot_txy_back = numpyro.deterministic(
            "Itot_txy_back",
            constant_background_integral(mu_xyt, args['T'], args['A_area'])
        )
        mu_xyt_events = mu_xyt

    ####### LGCP BACKGROUND
    if args['model']=='cox_hawkes':
      # Intercept of linear combination
      a_0 = numpyro.sample("a_0", args['priors']['a_0'])

      # Generate gaussian vector to feed into VAE
      z_temporal = numpyro.sample("z_temporal",
        dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"]))
      )
      decoder_nn_temporal = vae_decoder_temporal(
        args["hidden_dim_temporal"],
        args["n_t"]
      )
      decoder_params = args["decoder_params_temporal"]
      # Approximate Gaussian Process with VAE
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
      # MARGINAL DIAGNOSTICS ONLY (rate_t, Itot_t): since the seasonal-diagonal
      # fix the likelihood's time integral is Itot_time (computed on the diagonal
      # a = sigma(t)); Itot_t * Itot_a * Itot_xy != Itot_txy by design.
      rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t + a_0))
      # calculate temporal integral over LGCP
      Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
      # Temporal part of log(mu(t,s))
      f_t_events=f_t[args["indices_t"]]

      # seasonal part of log(mu(t,s))
      z_seasonal = numpyro.sample("z_seasonal",
        dist.Normal(jnp.zeros(args["z_dim_seasonal"]), jnp.ones(args["z_dim_seasonal"]))
      )
      decoder_nn_seasonal = vae_decoder_seasonal(
        args["hidden_dim1_seasonal"],
        args["hidden_dim2_seasonal"],
        args["n_s"]
      )
      decoder_params = args["decoder_params_seasonal"]
      # Approximate Gaussian Process with VAE
      v_a = numpyro.deterministic("v_a", decoder_nn_seasonal[1](decoder_params, z_seasonal))
      f_a = numpyro.deterministic("f_a", v_a[0:args["n_s"]])
      # MARGINAL DIAGNOSTICS ONLY (rate_a, Itot_a): the seasonal integral enters
      # the likelihood only through Itot_time on the diagonal a = sigma(t), not as
      # a standalone factor; Itot_t * Itot_a * Itot_xy != Itot_txy by design.
      rate_a = numpyro.deterministic("rate_a",jnp.exp(f_a))
      # calculate integral over LGCP
      Itot_a = numpyro.deterministic("Itot_a", jnp.sum(rate_a)/args["n_s"]*args["S"])  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
      f_a_events = f_a[args["indices_a"]]

      # Generate gaussian vector to feed into VAE
      z_spatial = numpyro.sample("z_spatial",
        dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"]))
      )
      decoder_nn = vae_decoder_spatial(
        args["hidden_dim1_spatial"],
        args["hidden_dim2_spatial"],
        args["n_xy"]
      )
      decoder_params = args["decoder_params_spatial"]
      # Generate Gaussian Process from VAE
      f_xy = numpyro.deterministic("f_xy", jnp.exp(args['sp_var_mu']) * decoder_nn[1](decoder_params, z_spatial))
      f_xy_events=f_xy[args["indices_xy"]]

      # Calculate spatial intensity
      if 'spatial_cov' in args:
          # weights for linear combination
          # (args['spatial_cov'] is guaranteed 2-D by the constructor: X_s[:, None])
          w = numpyro.sample("w", args['priors']['w'])
          b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)

          f_xy_events = f_xy_events + b_0[args['cov_ind']]
          spatial_integral = spatial_refinement_integral(
              f_xy, args['integration_field_indices'], args['integration_areas'],
              b_0, args['integration_cov_indices'])
      else:
          # rate_xy = exp(f_xy) EXCLUDES the covariate term b_0; when covariates
          # are present the branch above folds b_0 into spatial_integral directly.
          rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy))  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
          spatial_integral = spatial_refinement_integral(
              f_xy, args['integration_field_indices'], args['integration_areas'])
      Itot_xy=numpyro.deterministic("Itot_xy", spatial_integral)

      # Total background integral on the seasonal diagonal a = sigma(t), computed
      # EXACTLY via the precomputed (n_t x n_s) overlap matrix season_overlap: for
      # each temporal cell, seasonal_mass integrates exp(f_a[sigma(t)]) over the cell
      # (W already carries the internal cell measure, so no extra /n_t*T factor).
      # This replaces the old midpoint rule exp(a_0 + f_t + f_a[season_idx_of_t]).
      seasonal_mass, Itot_time_val = seasonal_time_integral(
          a_0, f_t, f_a, args['season_overlap'])
      # rate_time is a DIAGNOSTIC ONLY: the exact per-cell average intensity.
      rate_time = numpyro.deterministic("rate_time",  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
          jnp.exp(a_0 + f_t) * seasonal_mass / (args["T"]/args["n_t"]))
      Itot_time = numpyro.deterministic("Itot_time", Itot_time_val)
      Itot_txy_back = numpyro.deterministic("Itot_txy_back", Itot_time * Itot_xy)

      ## Replace month effect with day GP
      ## Sample a weight for each day
      ## One-hot encode the day indices for each event
      ## Compute the day effect for each event
      ## Add to your log-intensity
      #f_t_events = f_t_events + day_effect

    #### EXPONENTIAL KERNEL for the excitation part
    #alpha is the reproduction rate
    alpha = numpyro.sample("alpha", args['priors']['alpha'])

    #spatial gaussian kernel parameters
    t_pars = args['t_trig'].sample_parameters()
    sp_pars = args['sp_trig'].sample_parameters()

    T,x_min,x_max,y_min,y_max = args['T'],args['x_min'],args['x_max'],args['y_min'],args['y_max']

    # Temporal trigger: use coords and t_vals
    _, t_trigger_vals = args['t_trig'].compute_trigger(t_pars, (coords, t_vals))

    # Spatial trigger: use coords, x_vals, y_vals
    _, s_trigger_vals = args['sp_trig'].compute_trigger(sp_pars, (coords, x_vals, y_vals))

    # Inner sum of the event term, eq. (23): pure aggregation atom
    # (kernel evaluation stays here at the model layer; see likelihood.py)
    l_hawkes_sum = aggregate_pair_trigger_values(coords, t_trigger_vals,
                                                 s_trigger_vals, t_events.shape[0])
    l_hawkes = numpyro.deterministic('l_hawkes', alpha * l_hawkes_sum)

    if args['model'] == 'hawkes':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+mu_xyt_events)))
    elif args['model']=='cox_hawkes':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+jnp.exp(a_0 + f_t_events + f_a_events + f_xy_events))))

    #### hawkes integral: truncated excitation compensator over the rectangle
    #### (eq. 14 specialized + eq. 27; honest scope in likelihood.py docstring)
    win = args.get('window', T)   # temporal truncation window; NOT the sampled covariate weights `w`
    Itot_excite = numpyro.deterministic(
        "Itot_excite",
        rectangular_excitation_compensator(alpha, t_events, xy_events, T, win,
                                           (x_min, x_max, y_min, y_max),
                                           t_pars, sp_pars,
                                           args['t_trig'], args['sp_trig']))
    ## total integral
    Itot_txy = numpyro.deterministic("Itot_txy",Itot_excite + Itot_txy_back)
    loglik=numpyro.deterministic('loglik',ell_1-Itot_txy)

    numpyro.factor("loglik_factor", loglik)


def spatiotemporal_LGCP_model(args):
    #temporal rate
    a_0 = numpyro.sample("a_0", args['priors']['a_0'])

    #zero mean temporal gp
    z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
    decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])
    decoder_params = args["decoder_params_temporal"]
    v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
    f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
    # --- Add month covariate effect ---
    #f_t_i=f_t[args["indices_t"]] + month_effect
    # MARGINAL DIAGNOSTICS ONLY (rate_t, Itot_t): since the seasonal-diagonal fix
    # the likelihood's time integral is Itot_time (computed on the diagonal
    # a = sigma(t)); Itot_t * Itot_a * Itot_xy != Itot_txy by design.
    rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
    Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)

    # seasonal part of log(mu(t,s))
    z_seasonal = numpyro.sample("z_seasonal",
                              dist.Normal(jnp.zeros(args["z_dim_seasonal"]),
                                          jnp.ones(args["z_dim_seasonal"]))
                             )
    decoder_nn_seasonal = vae_decoder_seasonal(args["hidden_dim1_seasonal"], args["hidden_dim2_seasonal"], args["n_s"])
    decoder_params = args["decoder_params_seasonal"]
    # Approximate Gaussian Process with VAE
    v_a = numpyro.deterministic("v_a", decoder_nn_seasonal[1](decoder_params, z_seasonal))
    f_a = numpyro.deterministic("f_a", v_a[0:args["n_s"]])
    # MARGINAL DIAGNOSTICS ONLY (rate_a, Itot_a): the seasonal integral enters the
    # likelihood only through Itot_time on the diagonal a = sigma(t), not as a
    # standalone factor; Itot_t * Itot_a * Itot_xy != Itot_txy by design.
    rate_a = numpyro.deterministic("rate_a",jnp.exp(f_a))
    # calculate temporal integral over LGCP
    Itot_a = numpyro.deterministic("Itot_a", jnp.sum(rate_a)/args["n_s"]*args["S"])  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)

    # zero mean spatial gp
    z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
    decoder_nn = vae_decoder_spatial(args["hidden_dim1_spatial"], args["hidden_dim2_spatial"], args["n_xy"])
    decoder_params = args["decoder_params_spatial"]
    f_xy = numpyro.deterministic("f_xy", jnp.exp(args['sp_var_mu']) * decoder_nn[1](decoder_params, z_spatial))
    # rate_xy = exp(f_xy) EXCLUDES the covariate term b_0; in the covariate branch
    # below spatial_integral folds b_0 in directly rather than using rate_xy.
    rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy))  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
    f_xy_i=f_xy[args["indices_xy"]]

    if 'spatial_cov' in args:
        # weights for linear combination
        w = numpyro.sample("w", args['priors']['w'])
        b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)
        # Use precomputed indices for event locations
        f_xy_i = f_xy[args["indices_xy"]] + b_0[args['cov_ind']]
        spatial_integral = spatial_refinement_integral(
            f_xy, args['integration_field_indices'], args['integration_areas'],
            b_0, args['integration_cov_indices'])
    else:
        f_xy_i = f_xy[args["indices_xy"]]
        spatial_integral = spatial_refinement_integral(
            f_xy, args['integration_field_indices'], args['integration_areas'])

    Itot_xy=numpyro.deterministic("Itot_xy", spatial_integral)


    f_t_i = f_t[args["indices_t"]]
    f_a_i = f_a[args["indices_a"]]
    loglik = jnp.sum(a_0 + f_t_i + f_a_i + f_xy_i)
    # EXACT time integral on the seasonal diagonal a = sigma(t) via the precomputed
    # (n_t x n_s) overlap matrix season_overlap (replaces the midpoint rule at
    # season_idx_of_t; W already carries the internal cell measure).
    seasonal_mass, Itot_time_val = seasonal_time_integral(
        a_0, f_t, f_a, args['season_overlap'])
    # rate_time is a DIAGNOSTIC ONLY: the exact per-cell average intensity.
    rate_time = numpyro.deterministic("rate_time",  # noqa: F841 -- deterministic trace site (posterior interface, CLAUDE.md)
        jnp.exp(a_0 + f_t) * seasonal_mass / (args["T"]/args["n_t"]))
    Itot_time = numpyro.deterministic("Itot_time", Itot_time_val)
    Itot_txy = numpyro.deterministic("Itot_txy", Itot_time * Itot_xy)
    loglik-=Itot_txy
    numpyro.deterministic("loglik",loglik)

    numpyro.factor("loglik_factor", loglik)


def run_mcmc(rng_key, model_mcmc, args):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model_mcmc, init_strategy=init_strategy)#, max_tree_depth=(7,9))
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, args)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc

def get_samples(rng_key, model, guide, svi_result, args, sites):
    predictive = Predictive(model, guide=guide, params=svi_result.params,
                            return_sites=sites,
                            num_samples=args["num_samples"],
                            # original: True
                            parallel=True)
    if 'coords' in args:
        print("Number of posterior samples:", args["num_samples"])
        print("Number of pairs:", args['coords'].shape[0])
    posterior_samples = predictive(rng_key, args=args)
    return posterior_samples

def run_SVI(rng_key, model, args, num_steps, lr, sites, auto_guide = AutoMultivariateNormal, init_strategy=init_to_median,init_state=None):
    start = time.time()
    optimizer = numpyro.optim.Adam(inverse_time_decay(lr,num_steps,4))
    guide = auto_guide(model,init_loc_fn=init_strategy)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps, args, stable_update=True, init_state=init_state)
    posterior_samples = get_samples(rng_key,model,guide,svi_result,args,sites)
    print("\nSVI elapsed time:", time.time() - start)
    return svi,svi_result ,posterior_samples
