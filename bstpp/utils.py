# general libraries
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops
#from jax.experimental import stax

from functools import partial


#@title

def square_mean(a,b):
  return np.mean((a-b)**2)

def sq_diff(a,b):
  return np.sum((a-b)**2)
  
## helper functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx.astype(int)

def find_nearest_2D(array, value):
    array = np.asarray(array)
    idx = (np.sqrt(np.sum((array-value)**2,1))).argmin()
    return idx

    
def randdist(x, pdf, nvals):
    """Produce nvals random samples from pdf(x), assuming constant spacing in x."""
    # get cumulative distribution from 0 to 1
    cumpdf = np.cumsum(pdf)
    cumpdf *= 1/cumpdf[-1]

    # input random values
    randv = np.random.uniform(size=nvals)

    # find where random values would go
    idx1 = np.searchsorted(cumpdf, randv)
    # get previous value, avoiding division by zero below
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1.at[idx0==0].set(1)

    # do linear interpolation in x
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    if x.size==x.shape[0]:
        randdist = x[idx0]*(1-frac1) + x[idx1]*frac1
    elif x.size==x.shape[0]*2:
        randdist = x[idx0,:]*(1-frac1).reshape(idx0.size,1) + x[idx1,:]*frac1.reshape(idx0.size,1) 
    randdist_y = pdf[idx0]*(1-frac1) + pdf[idx1]*frac1
    indices = randdist>0
    return idx0, idx1, randdist[indices], randdist_y[indices]



    #@title
def dist_euclid(x, z):
    x = jnp.array(x) 
    z = jnp.array(z)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1)
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        x_d = x[:,d]
        z_d = z[:,d]
        delta += (x_d[:,jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-6):
    dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k

    
def GP(args, jitter=1e-4, y=None, var=None, length=None, sigma=None, noise=False):
    
    x = args["x"]
    #obs_idx = args["obs_idx"]
    gp_kernel=args["gp_kernel"] 

    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4,1))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
    
    k = gp_kernel(x, x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.05))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)



#@title
def rej_sampling_new(N, grid, gp_function, n):
  f_max=np.max(gp_function);
  ids=np.arange(0, n)
  if N<100:
    N_max=N*100
  else:
    N_max=N*10;
  index=np.random.choice(ids,N_max)

  candidate_points=grid[index];

  #plt.plot(args['x_t'],gp_function)
  #plt.plot(args['x_t'][index],gp_function[index],'x')
  U=np.random.uniform(0, f_max, N_max);
  #plt.plot(x_t[index],U,'o')
  indices=jnp.where(U<gp_function[index]);
  #plt.plot(x_t[index][indices],U[indices],'+' )
  #plt.plot(x_t[index][indices],np.zeros(indices[0].size),'+' )
  accepted_points=grid[index][indices][0:N]
  accepted_f=gp_function[index][indices][0:N]
  return jnp.array(index[indices][0:N]), accepted_points, accepted_f



def find_index_b(events, grid):
  n=events.shape[0];
  ind=np.zeros(n)*np.nan
  for i in range(n):  
    if events.size>events.shape[0]:
      ind[i]=jnp.nanargmin(np.sqrt(np.sum((events[i,:]-grid)**2,1)))
    else:
      ind[i]=jnp.nanargmin(np.abs(events[i]-grid))
  return ind.astype(int)
 

#@title
def find_index(events, grid):
  if isinstance(events,np.ndarray):
    n=events.shape[0];
    ind=np.zeros(n)*np.nan
    for i in range(n):  
      if events.size>events.shape[0]:
        ind[i]=jnp.nanargmin(np.sqrt(np.sum((events[i,:]-grid)**2,1)))
      else:
        ind[i]=jnp.nanargmin(np.abs(events[i]-grid))
  else:
    ind=jnp.nanargmin(np.abs(events-grid))
  return ind.astype(int)
  

  #@title

def difference_matrix(a, window=15):
    x = jnp.reshape(a, (a.shape[0], 1))
    x2 = x - x.T
    mask = (x2 > 0) & (x2 <= window)
    indices = jnp.where(mask)
    values = x2[indices]
    coords = jnp.stack(indices, axis=-1)
    return coords, values

def difference_matrix_partial(a, partial_index, window=15):
    x = jnp.reshape(a, (a.shape[0], 1))
    x2 = x[partial_index] - x.T
    mask = (x2 > 0) & (x2 <= window)
    indices = jnp.where(mask)
    values = x2[indices]
    coords = jnp.stack(indices, axis=-1)
    return coords, values

def difference_pairs(a, window=15):
    x = jnp.reshape(a, (a.shape[0], 1))
    x2 = x - x.T
    mask = (x2 > 0) & (x2 <= window)
    indices = jnp.where(mask)
    values = x2[indices]
    coords = jnp.stack(indices, axis=-1)
    return coords, values

def difference_pairs_2d(x, y, window=15):
    x = jnp.reshape(x, (x.shape[0], 1))
    y = jnp.reshape(y, (y.shape[0], 1))
    dx = x - x.T
    dy = y - y.T
    mask = (dx > 0) & (dx <= window)


    
    indices = jnp.where(mask)
    coords = jnp.stack(indices, axis=-1)
    dx_vals = dx[indices]
    dy_vals = dy[indices]
    return coords, dx_vals, dy_vals

def aligned_difference_pairs(t, x, y, window, spatial_window=None):
    """Emit the (receiver i, source j) excitation pairs with 0 < t[i]-t[j] <= window.

    Contract (identical to the former dense n x n implementation):
      - returns (coords, t_vals, x_vals, y_vals) as jnp arrays;
      - coords is (P, 2) with coords[:, 0] = i (receiver / later event) and
        coords[:, 1] = j (source / earlier event), in the ORIGINAL event order;
      - t_vals = t[i] - t[j], strictly > 0 and <= window (equal times excluded);
      - x_vals = x[i] - x[j], y_vals = y[i] - y[j];
      - if spatial_window is not None, only pairs with
        sqrt(x_vals**2 + y_vals**2) <= spatial_window are kept.

    The pair ORDERING within the returned arrays is NOT part of the contract:
    the likelihood aggregates them with segment_sum on coords[:, 0], which is
    order-invariant. This is why a sort-based construction is a drop-in.

    Construction is O(n log n + P) in time and O(n + P) in memory (P = number of
    emitted pairs): sort by time, then for each event use searchsorted to find the
    contiguous block of earlier events within the window, instead of materializing
    the dense (n, n) difference matrices.
    """
    window = float(window)
    if spatial_window is not None:
        spatial_window = float(spatial_window)

    t = np.asarray(t).reshape(-1)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = t.shape[0]

    if n == 0:
        return (jnp.zeros((0, 2), dtype=jnp.int32), jnp.zeros((0,)),
                jnp.zeros((0,)), jnp.zeros((0,)))

    order = np.argsort(t, kind='stable')
    ts = t[order]

    # For receiver at sorted position k, valid sources are the sorted positions
    # [lo[k], hi[k]): ts[j] >= ts[k]-window (dt <= window, side='left' keeps
    # dt == window) and ts[j] < ts[k] (strictly earlier, so dt > 0 and ties are
    # excluded because side='left' places equal times at/after hi[k]).
    lo = np.searchsorted(ts, ts - window, side='left')
    hi = np.searchsorted(ts, ts, side='left')
    counts = np.maximum(hi - lo, 0)

    P = int(counts.sum())
    if P == 0:
        return (jnp.zeros((0, 2), dtype=jnp.int32), jnp.zeros((0,)),
                jnp.zeros((0,)), jnp.zeros((0,)))

    rows = np.repeat(np.arange(n), counts)              # receiver positions (sorted)
    starts = np.repeat(lo, counts)                       # block start per emitted pair
    offs = np.arange(P) - np.repeat(np.cumsum(counts) - counts, counts)
    srcs = starts + offs                                 # source positions (sorted)

    i_idx = order[rows]                                  # receiver, ORIGINAL order
    j_idx = order[srcs]                                  # source, ORIGINAL order

    t_vals = t[i_idx] - t[j_idx]
    x_vals = x[i_idx] - x[j_idx]
    y_vals = y[i_idx] - y[j_idx]

    if spatial_window is not None:
        keep = np.sqrt(x_vals**2 + y_vals**2) <= spatial_window
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        t_vals, x_vals, y_vals = t_vals[keep], x_vals[keep], y_vals[keep]

    coords = np.stack((i_idx, j_idx), axis=-1)
    return (jnp.asarray(coords), jnp.asarray(t_vals),
            jnp.asarray(x_vals), jnp.asarray(y_vals))

def sq_diff(a,b):
  return np.sum((a-b)**2)


def square_mean(a,b):
  return np.mean((a-b)**2)
