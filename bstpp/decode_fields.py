"""Pure application of the pretrained PriorVAE decoder surrogates (guide eq. 22).

Kept SEPARATE from likelihood.py deliberately: the guide treats the decoder
surrogates as the prior-side model substitution (eq. 22), a distinct layer
from the discrete likelihood (eqs. 23-27), and the module boundary mirrors
that distinction. Pure functions: jnp only, no numpyro, no `args` dict --
trace-site registration (v_t, f_t, ...) stays at the model layer.

The network dimensions are pinned by the pretrained artifacts (n_t = 50,
n_s = 24, 25x25 spatial): they are arguments here only so the functions stay
honest about what they consume, not because other values would work with the
shipped decoder parameter pickles.
"""
import jax.numpy as jnp

from .vae_functions import (vae_decoder_temporal, vae_decoder_seasonal,
                            vae_decoder_spatial)


def decode_temporal_field(z, decoder_params, hidden_dim, n_t):
    """Full temporal decoder output v (callers slice f_t = v[:n_t]).

    The slice stays with the caller because v_t and f_t are separate trace
    sites at the model layer.
    """
    decoder = vae_decoder_temporal(hidden_dim, n_t)
    return decoder[1](decoder_params, z)


def decode_seasonal_field(z, decoder_params, hidden_dim1, hidden_dim2, n_s):
    """Full seasonal decoder output v (callers slice f_a = v[:n_s])."""
    decoder = vae_decoder_seasonal(hidden_dim1, hidden_dim2, n_s)
    return decoder[1](decoder_params, z)


def decode_spatial_field(z, decoder_params, hidden_dim1, hidden_dim2, n_xy,
                         sp_var_mu):
    """Spatial field f_xy = exp(sp_var_mu) * decoder(z).

    exp(sp_var_mu) is a DECODER-PAIRING CALIBRATION: it restores the
    log-amplitude factored out of the spatial draws during VAE training
    (guide Appendix C). It is a computational convention paired with these
    decoder parameters -- not a modeling commitment, not a likelihood
    parameter, and not a sampled prior. It lives inside this function so
    that no caller can apply the gain inconsistently: before extraction,
    three call sites (both model functions and simulate()) each applied it
    independently.
    """
    decoder = vae_decoder_spatial(hidden_dim1, hidden_dim2, n_xy)
    return jnp.exp(sp_var_mu) * decoder[1](decoder_params, z)
