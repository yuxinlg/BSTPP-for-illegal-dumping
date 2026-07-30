"""
Helper functions for extracting spatial grid information from Point Process Models.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .preparation import SLIVER_AREA_INTERNAL


def _posterior_b0_mean(model):
    """Posterior-mean covariate effect per covariate polygon, shape (n_cov,).

    Prefers ``samples['b_0']``; otherwise derives ``X @ mean(w)`` from the
    prepared covariate matrix. Does not mutate model state.
    """
    if 'b_0' in model.samples:
        return np.asarray(model.samples['b_0']).mean(axis=0)
    if 'w' in model.samples and 'spatial_cov' in model.args:
        X = np.asarray(model.args['spatial_cov'])
        w_mean = np.asarray(model.samples['w']).mean(axis=0)
        return np.asarray(X @ w_mean)
    raise Exception(
        "include_cov=True requires samples['b_0'] or samples['w'] with "
        "prepared spatial covariates")


def _covariate_comp_grid_refinement(model):
    """Local support×covariate refinement with intersection areas.

    Mirrors the prepared cox/LGCP ``int_df`` construction without writing
    into ``model.args`` or mutating ``spatial_cov`` / ``comp_grid``.
    """
    support = model.prepared_partitions.support_cells[
        ['comp_grid_id', 'geometry']]
    intersect = gpd.overlay(
        support, model.spatial_cov, how='intersection', keep_geom_type=True)
    A_ = np.asarray(model.prepared_domain.bounds)
    denom = (A_[0, 1] - A_[0, 0]) * (A_[1, 1] - A_[1, 0])
    refine = intersect.copy()
    refine['area'] = refine.area / denom
    refine = refine[refine['area'] > SLIVER_AREA_INTERNAL]
    if 'cov_ind' not in refine.columns:
        raise Exception(
            "Covariate layer is missing cov_ind required for grid mapping")
    return refine


def get_grid_post_mean(model, include_cov=False):
    """
    Extract posterior mean values for each grid cell in the computational grid.
    
    This function computes the mean posterior spatial intensity for each cell in the
    25x25 computational grid and returns a GeoDataFrame with grid coordinates,
    posterior mean values, and polygon geometries.
    
    Parameters
    ----------
    model : Point_Process_Model
        A fitted Point Process Model instance (e.g., Hawkes_Model, LGCP_Model)
        that has been run through inference (has 'samples' attribute).
    include_cov : bool, default=False
        If True and model has spatial covariates, include covariate effects
        in the posterior mean calculation. Otherwise, only return the GP component.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the following columns:
        - grid_row : int
            Row coordinate (1-25), where 1 is the top row
        - grid_col : int
            Column coordinate (1-25), where 1 is the left column
        - post_mean : float
            Mean posterior spatial intensity for this grid cell
        - comp_grid_id : int
            Original computational grid ID (0-624)
        - geometry : shapely.geometry.Polygon
            Polygon geometry of the grid cell
    
    Raises
    ------
    Exception
        If model has not been run through inference (no 'samples' attribute).
    Exception
        If model does not support spatial GP (not 'cox_hawkes' or 'lgcp') and include_cov=False.
    Exception
        If include_cov=True but model has no spatial covariates.
    """
    if 'samples' not in dir(model):
        raise Exception("MCMC posterior sampling has not been performed yet.")
    
    n_xy = model.args['n_xy']  # Should be 25
    
    # Check model type and covariates
    if model.args['model'] not in ['cox_hawkes', 'lgcp'] and not include_cov:
        raise Exception("Nothing to extract: spatial background is constant")
    if include_cov and 'spatial_cov' not in model.args:
        raise Exception("No spatial covariates are in the model and include_cov was set to True")
    
    # Compute posterior mean
    if model.args['model'] in ['cox_hawkes', 'lgcp'] and include_cov:
        # Include both GP and covariate effects
        b0_mean = _posterior_b0_mean(model)
        cov_inds = model.args['int_df']['cov_ind'].values
        f_xy_mean = np.asarray(model.samples["f_xy"]).mean(axis=0)
        post_mean = (
            b0_mean[cov_inds]
            + f_xy_mean[model.args['int_df']['comp_grid_id'].values]
        )

        # Create result from int_df
        result_gdf = model.args['int_df'].copy()
        result_gdf['post_mean'] = post_mean

        # Add grid coordinates based on comp_grid_id
        result_gdf['grid_row'] = n_xy - (result_gdf['comp_grid_id'] // n_xy)
        result_gdf['grid_col'] = (result_gdf['comp_grid_id'] % n_xy) + 1

        # Reorder columns
        result_gdf = result_gdf[['grid_row', 'grid_col', 'post_mean', 'comp_grid_id', 'geometry']]

    elif include_cov:
        # Plain Hawkes covariate effects on the computational grid.
        # Build a local support×covariate refinement; never require a prior
        # plot to have written spatial_cov['post_mean'], and do not mutate
        # model.spatial_cov / comp_grid / prepared state.
        result_gdf = model.comp_grid.copy()
        refine = _covariate_comp_grid_refinement(model)
        b0_mean = _posterior_b0_mean(model)
        refine = refine.copy()
        refine['post_mean'] = b0_mean[refine['cov_ind'].values]
        # Area-weighted mean of covariate effects per computational cell.
        wsum = refine.groupby('comp_grid_id').apply(
            lambda g: float(np.average(g['post_mean'].to_numpy(),
                                       weights=g['area'].to_numpy()))
        )
        result_gdf['post_mean'] = result_gdf['comp_grid_id'].map(wsum).fillna(0)

        # Add grid coordinates
        result_gdf['grid_row'] = n_xy - (result_gdf['comp_grid_id'] // n_xy)
        result_gdf['grid_col'] = (result_gdf['comp_grid_id'] % n_xy) + 1

        # Reorder columns
        result_gdf = result_gdf[['grid_row', 'grid_col', 'post_mean', 'comp_grid_id', 'geometry']]
        
    else:
        # Only GP component (no covariates)
        f_xy_post = model.samples["f_xy"]
        f_xy_post_mean = jnp.mean(f_xy_post, axis=0)
        
        # Create result from comp_grid
        result_gdf = model.comp_grid.copy()
        result_gdf['post_mean'] = np.asarray(f_xy_post_mean)
        
        # Add grid coordinates
        # comp_grid_id goes from 0 to n_xy^2 - 1
        # Grid is created with y (rows) first, then x (columns)
        # The loop structure: for y in cols: for x in cols:
        #   - y=0 (bottom) to y=24/25 (top)
        #   - x=0 (left) to x=24/25 (right)
        # So: row = comp_grid_id // n_xy (0=bottom, 24=top)
        #     col = comp_grid_id % n_xy (0=left, 24=right)
        # To get (1,1) at top-left: grid_row = n_xy - row, grid_col = col + 1
        result_gdf['grid_row'] = n_xy - (result_gdf['comp_grid_id'] // n_xy)
        result_gdf['grid_col'] = (result_gdf['comp_grid_id'] % n_xy) + 1
        
        # Reorder columns
        result_gdf = result_gdf[['grid_row', 'grid_col', 'post_mean', 'comp_grid_id', 'geometry']]
    
    # Ensure grid_row and grid_col are integers
    result_gdf['grid_row'] = result_gdf['grid_row'].astype(int)
    result_gdf['grid_col'] = result_gdf['grid_col'].astype(int)
    
    # Sort by grid_row (descending, so top row first) then grid_col (ascending)
    result_gdf = result_gdf.sort_values(['grid_row', 'grid_col'], ascending=[False, True])
    
    return result_gdf.reset_index(drop=True)


def identify_hotspots_coldspots(result_gdf, quantile_method='auto'):
    """
    Identify hotspots and coldspots from the result GeoDataFrame based on post_mean values.
    
    Uses quantile-based thresholds. By default, tries to use matplotlib's default method
    for continuous colormaps (2nd and 98th percentiles), or falls back to 10% quantiles.
    
    Parameters
    ----------
    result_gdf : gpd.GeoDataFrame
        GeoDataFrame from get_grid_post_mean() with 'post_mean' column.
    quantile_method : str, default='auto'
        Method for determining thresholds:
        - 'auto': Try matplotlib's default (2nd/98th percentiles), fallback to 10%
        - 'matplotlib': Use 2nd and 98th percentiles (matplotlib default)
        - '10pct': Use 10th and 90th percentiles (top 10% and bottom 10%)
        - 'custom': Use custom percentiles (not implemented, would need additional args)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'hotspots': gpd.GeoDataFrame with hotspot cells
        - 'coldspots': gpd.GeoDataFrame with coldspot cells
        - 'hotspot_threshold': float, threshold value for hotspots
        - 'coldspot_threshold': float, threshold value for coldspots
        - 'hotspot_pct': float, percentile used for hotspot threshold
        - 'coldspot_pct': float, percentile used for coldspot threshold
    """
    if 'post_mean' not in result_gdf.columns:
        raise ValueError("result_gdf must contain 'post_mean' column")
    
    post_mean_values = result_gdf['post_mean'].values
    
    # Determine quantile thresholds
    if quantile_method == 'auto':
        # Try matplotlib's default method (2nd and 98th percentiles)
        try:
            # Check if PercentileNorm is available and use its default
            hotspot_pct = 98.0
            coldspot_pct = 2.0
            hotspot_threshold = np.percentile(post_mean_values, hotspot_pct)
            coldspot_threshold = np.percentile(post_mean_values, coldspot_pct)
        except Exception:
            # Fallback to 10%
            hotspot_pct = 90.0
            coldspot_pct = 10.0
            hotspot_threshold = np.percentile(post_mean_values, hotspot_pct)
            coldspot_threshold = np.percentile(post_mean_values, coldspot_pct)
    elif quantile_method == 'matplotlib':
        hotspot_pct = 98.0
        coldspot_pct = 2.0
        hotspot_threshold = np.percentile(post_mean_values, hotspot_pct)
        coldspot_threshold = np.percentile(post_mean_values, coldspot_pct)
    elif quantile_method == '10pct':
        hotspot_pct = 90.0
        coldspot_pct = 10.0
        hotspot_threshold = np.percentile(post_mean_values, hotspot_pct)
        coldspot_threshold = np.percentile(post_mean_values, coldspot_pct)
    else:
        raise ValueError(f"Unknown quantile_method: {quantile_method}")
    
    # Identify hotspots (above threshold) and coldspots (below threshold)
    hotspots = result_gdf[result_gdf['post_mean'] >= hotspot_threshold].copy()
    coldspots = result_gdf[result_gdf['post_mean'] <= coldspot_threshold].copy()
    
    # Sort hotspots by post_mean (descending) and coldspots (ascending)
    hotspots = hotspots.sort_values('post_mean', ascending=False).reset_index(drop=True)
    coldspots = coldspots.sort_values('post_mean', ascending=True).reset_index(drop=True)
    
    return {
        'hotspots': hotspots,
        'coldspots': coldspots,
        'hotspot_threshold': hotspot_threshold,
        'coldspot_threshold': coldspot_threshold,
        'hotspot_pct': hotspot_pct,
        'coldspot_pct': coldspot_pct
    }


def plot_daily_event_counts(result_gdf, events_gdf, plot_mode='combined', figsize=(8, 5)):
    """
    Plot daily event counts for hotspots and coldspots.
    
    Parameters
    ----------
    result_gdf : gpd.GeoDataFrame
        GeoDataFrame from get_grid_post_mean() with 'post_mean' column.
    events_gdf : gpd.GeoDataFrame
        Event dataset with geometry and 'start_time' column (datetime or timestamp).
    plot_mode : str, default='combined'
        - 'combined': Plot all hotspots combined and all coldspots combined
        - 'top5': Plot top 5 hotspots and top 5 coldspots individually
    figsize : tuple, default=(8, 5)
        Figure size (width, height) in inches.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    # First identify hotspots and coldspots
    hotspot_coldspot_dict = identify_hotspots_coldspots(result_gdf)
    hotspots = hotspot_coldspot_dict['hotspots']
    coldspots = hotspot_coldspot_dict['coldspots']
    
    print(f"Found {len(hotspots)} hotspots and {len(coldspots)} coldspots")
    
    # Ensure events_gdf has geometry and start_time
    if 'geometry' not in events_gdf.columns:
        raise ValueError("events_gdf must have 'geometry' column. "
                        "If you have coordinates, create geometry with: "
                        "gpd.points_from_xy(events_gdf['lon'], events_gdf['lat'])")
    if 'start_time' not in events_gdf.columns:
        raise ValueError("events_gdf must have 'start_time' column")
    
    print(f"Events dataframe has {len(events_gdf)} events")
    print(f"Events CRS: {events_gdf.crs}, Hotspots CRS: {hotspots.crs if len(hotspots) > 0 else 'N/A'}")
    
    # Check if geometries are valid
    if events_gdf.geometry.isna().any():
        print(f"Warning: {events_gdf.geometry.isna().sum()} events have null geometries")
    if hasattr(events_gdf.geometry, 'is_valid'):
        invalid_geoms = ~events_gdf.geometry.is_valid
        if invalid_geoms.any():
            print(f"Warning: {invalid_geoms.sum()} events have invalid geometries")
    
    # Convert start_time to datetime if needed
    events_gdf = events_gdf.copy()
    if not pd.api.types.is_datetime64_any_dtype(events_gdf['start_time']):
        events_gdf['start_time'] = pd.to_datetime(events_gdf['start_time'])
    
    # Ensure CRS is set and match
    if len(hotspots) > 0 and hotspots.crs is None:
        raise ValueError("hotspots GeoDataFrame must have a CRS set")
    if events_gdf.crs is None:
        if len(hotspots) > 0:
            # Try to set CRS from hotspots if events don't have one
            print(f"Setting events CRS to {hotspots.crs}")
            events_gdf = events_gdf.set_crs(hotspots.crs)
        else:
            raise ValueError("events_gdf must have a CRS set when hotspots are empty")
    elif len(hotspots) > 0 and events_gdf.crs != hotspots.crs:
        print(f"Transforming events CRS from {events_gdf.crs} to {hotspots.crs}")
        events_gdf = events_gdf.to_crs(hotspots.crs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_mode == 'combined':
        # Combine all hotspot polygons and all coldspot polygons
        from shapely.ops import unary_union
        
        # Check if we have any hotspots/coldspots
        if len(hotspots) == 0:
            hotspot_union_gdf = gpd.GeoDataFrame(geometry=[], crs=hotspots.crs)
        else:
            # Create union of all hotspot polygons
            try:
                hotspot_union_geom = unary_union(hotspots.geometry.values)
                if hotspot_union_geom.is_empty:
                    hotspot_union_gdf = gpd.GeoDataFrame(geometry=[], crs=hotspots.crs)
                else:
                    hotspot_union = gpd.GeoSeries([hotspot_union_geom], crs=hotspots.crs)
                    hotspot_union_gdf = gpd.GeoDataFrame(geometry=hotspot_union, crs=hotspots.crs)
            except Exception:
                # Fallback: use individual polygons if union fails
                hotspot_union_gdf = hotspots[['geometry']].copy()
        
        if len(coldspots) == 0:
            coldspot_union_gdf = gpd.GeoDataFrame(geometry=[], crs=coldspots.crs)
        else:
            # Create union of all coldspot polygons
            try:
                coldspot_union_geom = unary_union(coldspots.geometry.values)
                if coldspot_union_geom.is_empty:
                    coldspot_union_gdf = gpd.GeoDataFrame(geometry=[], crs=coldspots.crs)
                else:
                    coldspot_union = gpd.GeoSeries([coldspot_union_geom], crs=coldspots.crs)
                    coldspot_union_gdf = gpd.GeoDataFrame(geometry=coldspot_union, crs=coldspots.crs)
            except Exception:
                # Fallback: use individual polygons if union fails
                coldspot_union_gdf = coldspots[['geometry']].copy()
        
        # Drop index columns BEFORE spatial join to avoid conflicts
        events_gdf_clean = events_gdf.drop(columns=[col for col in events_gdf.columns 
                                                    if col in ['index_right', 'index_left']], errors='ignore')
        hotspot_union_gdf_clean = hotspot_union_gdf.drop(columns=[col for col in hotspot_union_gdf.columns 
                                                                    if col in ['index_right', 'index_left']], errors='ignore')
        coldspot_union_gdf_clean = coldspot_union_gdf.drop(columns=[col for col in coldspot_union_gdf.columns 
                                                                    if col in ['index_right', 'index_left']], errors='ignore')
        
        # Find events in hotspots and coldspots using spatial join
        if len(hotspot_union_gdf_clean) > 0:
            # Try with individual polygons if union doesn't work
            try:
                events_in_hotspots = events_gdf_clean.sjoin(hotspot_union_gdf_clean, how='inner', predicate='within')
                print(f"Found {len(events_in_hotspots)} events in hotspots (using union with 'within')")
            except Exception as e:
                print(f"Warning: Union 'within' failed for hotspots: {e}")
                # Try with individual polygons instead of union
                try:
                    hotspots_clean = hotspots.drop(columns=[col for col in hotspots.columns 
                                                            if col in ['index_right', 'index_left']], errors='ignore')
                    events_in_hotspots = events_gdf_clean.sjoin(hotspots_clean[['geometry']], how='inner', predicate='within')
                    # Remove duplicates if an event falls in multiple hotspot polygons
                    # Keep only geometry and original event columns (exclude index_right from join)
                    event_cols = [col for col in events_in_hotspots.columns 
                                 if col not in ['index_right', 'geometry']]
                    if 'index_right' in events_in_hotspots.columns:
                        events_in_hotspots = events_in_hotspots.drop_duplicates(subset=event_cols)
                        events_in_hotspots = events_in_hotspots.drop(columns=['index_right'], errors='ignore')
                    print(f"Found {len(events_in_hotspots)} events in hotspots (using individual polygons with 'within')")
                except Exception as e2:
                    print(f"Warning: Individual polygons 'within' also failed: {e2}")
                    # Try with intersects as final fallback
                    try:
                        events_in_hotspots = events_gdf_clean.sjoin(hotspot_union_gdf_clean, how='inner', predicate='intersects')
                        print(f"Found {len(events_in_hotspots)} events in hotspots (using union with 'intersects')")
                    except Exception as e3:
                        print(f"Warning: All spatial join methods failed: {e3}")
                        events_in_hotspots = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
        else:
            print("No hotspot polygons to join with")
            events_in_hotspots = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
        
        if len(coldspot_union_gdf_clean) > 0:
            # Try with individual polygons if union doesn't work
            try:
                events_in_coldspots = events_gdf_clean.sjoin(coldspot_union_gdf_clean, how='inner', predicate='within')
                print(f"Found {len(events_in_coldspots)} events in coldspots (using union with 'within')")
            except Exception as e:
                print(f"Warning: Union 'within' failed for coldspots: {e}")
                # Try with individual polygons instead of union
                try:
                    coldspots_clean = coldspots.drop(columns=[col for col in coldspots.columns 
                                                              if col in ['index_right', 'index_left']], errors='ignore')
                    events_in_coldspots = events_gdf_clean.sjoin(coldspots_clean[['geometry']], how='inner', predicate='within')
                    # Remove duplicates if an event falls in multiple coldspot polygons
                    # Keep only geometry and original event columns (exclude index_right from join)
                    event_cols = [col for col in events_in_coldspots.columns 
                                 if col not in ['index_right', 'geometry']]
                    if 'index_right' in events_in_coldspots.columns:
                        events_in_coldspots = events_in_coldspots.drop_duplicates(subset=event_cols)
                        events_in_coldspots = events_in_coldspots.drop(columns=['index_right'], errors='ignore')
                    print(f"Found {len(events_in_coldspots)} events in coldspots (using individual polygons with 'within')")
                except Exception as e2:
                    print(f"Warning: Individual polygons 'within' also failed: {e2}")
                    # Try with intersects as final fallback
                    try:
                        events_in_coldspots = events_gdf_clean.sjoin(coldspot_union_gdf_clean, how='inner', predicate='intersects')
                        print(f"Found {len(events_in_coldspots)} events in coldspots (using union with 'intersects')")
                    except Exception as e3:
                        print(f"Warning: All spatial join methods failed: {e3}")
                        events_in_coldspots = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
        else:
            print("No coldspot polygons to join with")
            events_in_coldspots = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
        
        # Aggregate by day
        has_data = False
        if len(events_in_hotspots) > 0:
            events_in_hotspots['date'] = events_in_hotspots['start_time'].dt.date
            daily_hotspots = events_in_hotspots.groupby('date').size().reset_index(name='count')
            daily_hotspots['date'] = pd.to_datetime(daily_hotspots['date'])
            daily_hotspots = daily_hotspots.sort_values('date')
            ax.plot(daily_hotspots['date'], daily_hotspots['count'], 
                   color='#990000', label='All Hotspots Combined', linewidth=1.5)
            has_data = True
        else:
            print("Warning: No events found in hotspots")
        
        if len(events_in_coldspots) > 0:
            events_in_coldspots['date'] = events_in_coldspots['start_time'].dt.date
            daily_coldspots = events_in_coldspots.groupby('date').size().reset_index(name='count')
            daily_coldspots['date'] = pd.to_datetime(daily_coldspots['date'])
            daily_coldspots = daily_coldspots.sort_values('date')
            ax.plot(daily_coldspots['date'], daily_coldspots['count'], 
                   color='#011F5B', label='All Coldspots Combined', linewidth=1.5)
            has_data = True
        else:
            print("Warning: No events found in coldspots")
        
        if not has_data:
            ax.text(0.5, 0.5, 'No events found in hotspots or coldspots.\nCheck spatial join and CRS matching.', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    elif plot_mode == 'top5':
        # Plot top 5 hotspots and top 5 coldspots
        top5_hotspots = hotspots.head(5)
        top5_coldspots = coldspots.head(5)
        
        # Alpha values: 0.95, 0.85, 0.75, 0.65, 0.55
        alphas = [0.95, 0.85, 0.75, 0.65, 0.55]
        
        # Drop index columns from events_gdf before joining
        events_gdf_clean = events_gdf.drop(columns=[col for col in events_gdf.columns 
                                                    if col in ['index_right', 'index_left']], errors='ignore')
        
        # Plot top 5 hotspots
        for idx, (_, hotspot_row) in enumerate(top5_hotspots.iterrows()):
            if hotspot_row.geometry.is_empty or not hotspot_row.geometry.is_valid:
                # Skip invalid geometries
                label = f'Hotspot {idx+1}' if idx < 5 else None
                ax.plot([], [], color='#990000', alpha=alphas[idx], label=label, linewidth=1.5)
                continue
            
            hotspot_gdf = gpd.GeoDataFrame(geometry=[hotspot_row.geometry], crs=hotspots.crs)
            hotspot_gdf_clean = hotspot_gdf.drop(columns=[col for col in hotspot_gdf.columns 
                                                          if col in ['index_right', 'index_left']], errors='ignore')
            try:
                events_in_hotspot = events_gdf_clean.sjoin(hotspot_gdf_clean, how='inner', predicate='within')
            except Exception:
                # If spatial join fails, try with intersects as fallback
                try:
                    events_in_hotspot = events_gdf_clean.sjoin(hotspot_gdf_clean, how='inner', predicate='intersects')
                except Exception:
                    events_in_hotspot = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
            
            if len(events_in_hotspot) > 0:
                events_in_hotspot = events_in_hotspot.copy()
                events_in_hotspot['date'] = events_in_hotspot['start_time'].dt.date
                daily_events = events_in_hotspot.groupby('date').size().reset_index(name='count')
                daily_events['date'] = pd.to_datetime(daily_events['date'])
                daily_events = daily_events.sort_values('date')
                
                label = f'Hotspot {idx+1}' if idx < 5 else None
                ax.plot(daily_events['date'], daily_events['count'], 
                       color='#990000', alpha=alphas[idx], label=label, linewidth=1.5)
            else:
                label = f'Hotspot {idx+1}' if idx < 5 else None
                ax.plot([], [], color='#990000', alpha=alphas[idx], label=label, linewidth=1.5)
        
        # Plot top 5 coldspots
        for idx, (_, coldspot_row) in enumerate(top5_coldspots.iterrows()):
            if coldspot_row.geometry.is_empty or not coldspot_row.geometry.is_valid:
                # Skip invalid geometries
                label = f'Coldspot {idx+1}' if idx < 5 else None
                ax.plot([], [], color='#011F5B', alpha=alphas[idx], label=label, linewidth=1.5)
                continue
            
            coldspot_gdf = gpd.GeoDataFrame(geometry=[coldspot_row.geometry], crs=coldspots.crs)
            coldspot_gdf_clean = coldspot_gdf.drop(columns=[col for col in coldspot_gdf.columns 
                                                            if col in ['index_right', 'index_left']], errors='ignore')
            try:
                events_in_coldspot = events_gdf_clean.sjoin(coldspot_gdf_clean, how='inner', predicate='within')
            except Exception:
                # If spatial join fails, try with intersects as fallback
                try:
                    events_in_coldspot = events_gdf_clean.sjoin(coldspot_gdf_clean, how='inner', predicate='intersects')
                except Exception:
                    events_in_coldspot = gpd.GeoDataFrame(columns=events_gdf_clean.columns, crs=events_gdf_clean.crs)
            
            if len(events_in_coldspot) > 0:
                events_in_coldspot = events_in_coldspot.copy()
                events_in_coldspot['date'] = events_in_coldspot['start_time'].dt.date
                daily_events = events_in_coldspot.groupby('date').size().reset_index(name='count')
                daily_events['date'] = pd.to_datetime(daily_events['date'])
                daily_events = daily_events.sort_values('date')
                
                label = f'Coldspot {idx+1}' if idx < 5 else None
                ax.plot(daily_events['date'], daily_events['count'], 
                       color='#011F5B', alpha=alphas[idx], label=label, linewidth=1.5)
            else:
                label = f'Coldspot {idx+1}' if idx < 5 else None
                ax.plot([], [], color='#011F5B', alpha=alphas[idx], label=label, linewidth=1.5)
    
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}. Must be 'combined' or 'top5'")
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Event Count')
    ax.set_title('Daily Event Counts: Hotspots vs Coldspots')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

