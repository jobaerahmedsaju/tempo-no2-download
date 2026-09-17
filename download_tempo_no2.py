# Download TEMPO NO2 L2 V04 via earthaccess, spatially mask to bbox, subset variables.
# Batched + resumable: processes granules in chunks, tracks progress/failures to disk

import sys
import earthaccess
import netCDF4 as nc
import numpy as np
import os
import re
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

earthaccess.login(persist=True)

tempo_dir = '/vscratch/grp-kangsun/jobaerah/TEMPONO2/'
raw_dir = os.path.join(tempo_dir, 'raw_tmp')
os.makedirs(raw_dir, exist_ok=True)

start_dt = '2023-08-01'
end_dt = '2026-09-01'
bbox = (-130, 23, -63, 52)  # west, south, east, north
west, south, east, north = bbox
batch_size = 200

progress_path = os.path.join(tempo_dir, 'batch_progress.pkl')
failed_path = os.path.join(tempo_dir, 'failed_granules.pkl')

geo_vars = ['latitude', 'longitude', 'latitude_bounds', 'longitude_bounds',
            'solar_zenith_angle', 'time', 'viewing_zenith_angle']
product_vars = ['main_data_quality_flag', 'vertical_column_stratosphere',
                 'vertical_column_troposphere', 'vertical_column_troposphere_uncertainty']
support_vars = ['albedo', 'amf_cloud_fraction', 'eff_cloud_fraction',
                 'pbl_height', 'surface_pressure', 'terrain_height',
                 'tropopause_pressure', 'vertical_column_total',
                 'vertical_column_total_uncertainty']


def get_fill_value(var):
    """Return var's own _FillValue, or NetCDF's default fill for its dtype."""
    if '_FillValue' in var.ncattrs():
        return var.getncattr('_FillValue')
    dtype = np.dtype(var.dtype)
    return nc.default_fillvals[f'{dtype.kind}{dtype.itemsize}']


def copy_group_vars(src_group, dst_group, var_list, spatial_mask):
    """Copy var_list from src_group to dst_group; mask out-of-bbox pixels to fill_value."""
    for attr in src_group.ncattrs():
        dst_group.setncattr(attr, src_group.getncattr(attr))

    for vname in var_list:
        if vname not in src_group.variables:
            continue
        var = src_group.variables[vname]
        for dname, dlen in zip(var.dimensions, var.shape):
            if dname not in dst_group.dimensions:
                dst_group.createDimension(dname, dlen)

        spatial_var = 'mirror_step' in var.dimensions and 'xtrack' in var.dimensions
        fill_value = get_fill_value(var) if spatial_var else getattr(var, '_FillValue', None)

        kwargs = {'zlib': True, 'complevel': 4}
        if fill_value is not None:
            kwargs['fill_value'] = fill_value

        out_var = dst_group.createVariable(vname, var.dtype, var.dimensions, **kwargs)

        var.set_auto_maskandscale(False)  # keep packed values as-is
        data = np.array(var[:], copy=True)

        if spatial_var:
            mask = spatial_mask
            while mask.ndim < data.ndim:
                mask = mask[..., None]  # broadcast to extra dims (bounds/corners)
            data = np.where(mask, data, fill_value)

        out_var[:] = data
        attrs = {k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'}
        out_var.setncatts(attrs)


def granule_ur(granule):
    """Extract a granule's unique filename-like identifier (GranuleUR) from its metadata."""
    return granule['umm']['GranuleUR']


def match_key(s):
    """Extract the timestamp_ScanGranule core, robust to naming/extension differences
    between GranuleUR and the actual downloaded filename."""
    m = re.search(r'\d{8}T\d{6}Z_S\d+G\d+', s)
    return m.group(0) if m else None


def output_path_for(granule_ur_str):
    """Build the dated output path (tempo_dir/product/YYYY/MM/DD/file.nc) for a granule UR."""
    fn = granule_ur_str
    if not fn.endswith('.nc'):
        fn = fn + '.nc'
    parts = fn.split('_')
    product_name = '_'.join(parts[:4])
    date_str = parts[4][:8]
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    return os.path.join(tempo_dir, product_name, year, month, day, fn)


def already_done(granule):
    """True if this granule's output file already exists on disk (for resume/skip)."""
    return os.path.exists(output_path_for(granule_ur(granule)))


def subset_one(downloaded_file):
    """Spatially mask + variable-subset one downloaded raw file; delete the raw file after."""
    file_name = os.path.basename(downloaded_file).replace('_subsetted', '')
    output_file = output_path_for(file_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        with nc.Dataset(downloaded_file, 'r') as src:
            lat = src.groups['geolocation']['latitude'][:]
            lon = src.groups['geolocation']['longitude'][:]
            lat_data = np.ma.getdata(lat)
            lon_data = np.ma.getdata(lon)

            valid = (~np.ma.getmaskarray(lat) & ~np.ma.getmaskarray(lon) &
                      np.isfinite(lat_data) & np.isfinite(lon_data))
            spatial_mask = (valid & (lon_data >= west) & (lon_data <= east) &
                             (lat_data >= south) & (lat_data <= north))

            with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
                for attr in src.ncattrs():
                    dst.setncattr(attr, src.getncattr(attr))

                geo_group = dst.createGroup('geolocation')
                product_group = dst.createGroup('product')
                support_group = dst.createGroup('support_data')

                copy_group_vars(src.groups['geolocation'], geo_group, geo_vars, spatial_mask)
                copy_group_vars(src.groups['product'], product_group, product_vars, spatial_mask)
                copy_group_vars(src.groups['support_data'], support_group, support_vars, spatial_mask)
    except Exception:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise
    finally:
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)  # clean up raw file, success or fail

    return output_file


def load_pickle(path, default):
    """Load a pickled object from path, or return default if the file doesn't exist yet."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return default


def save_pickle(path, obj):
    """Pickle obj to path, overwriting any existing file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    all_granules = earthaccess.search_data(
        short_name='TEMPO_NO2_L2', version='V04',
        temporal=(start_dt, end_dt), bounding_box=bbox
    )
    print(f'Found {len(all_granules)} granules', flush=True)

    start_batch = load_pickle(progress_path, 0)
    all_failed = load_pickle(failed_path, [])  # accumulate across restarts
    if start_batch:
        print(f'Resuming from batch {start_batch}; {len(all_failed)} prior failures',
              flush=True)

    n_batches = (len(all_granules) + batch_size - 1) // batch_size

    for batch_idx in range(start_batch, n_batches):
        batch = all_granules[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        todo = [g for g in batch if not already_done(g)]  # resume mid-batch too
        skipped = len(batch) - len(todo)

        print(f'\n--- Batch {batch_idx + 1}/{n_batches} '
              f'({len(todo)} to process, {skipped} done) ---', flush=True)

        if todo:
            try:
                local_files = earthaccess.download(todo, local_path=raw_dir, threads=16)
            except Exception as e:
                # nothing downloaded -- don't advance progress, stop for manual restart
                print(f'Batch download failed: {e}', flush=True)
                all_failed.extend(todo)
                save_pickle(failed_path, all_failed)
                raise

            # match by extracted timestamp/scan/granule key, not exact filename equality
            # or list position -- neither is reliable, and earthaccess can return
            # Exception objects in place of a path for a failed item
            by_key = {match_key(granule_ur(g)): g for g in todo}
            pending = {}
            unmatched = []
            for lf in local_files:
                if not isinstance(lf, (str, os.PathLike)):
                    continue  # exception object or other non-path -- counted as missing below
                key = match_key(os.path.basename(lf))
                g = by_key.get(key)
                if g is None:
                    unmatched.append(lf)
                else:
                    pending[lf] = g

            if unmatched:
                print(f'{len(unmatched)} downloaded files unmatched to a granule', flush=True)

            matched_keys = {match_key(granule_ur(g)) for g in pending.values()}
            missing = [g for g in todo if match_key(granule_ur(g)) not in matched_keys]
            if missing:
                # partial gap in an otherwise-ok batch -- log, don't fail whole batch
                print(f'{len(missing)}/{len(todo)} granules missing from download', flush=True)
                all_failed.extend(missing)

            if pending:
                with ProcessPoolExecutor(max_workers=30) as executor:  # processes: HDF5 unsafe w/ threads
                    futures = {executor.submit(subset_one, lf): g
                                for lf, g in pending.items()}
                    done = 0
                    for fut in as_completed(futures):
                        g = futures[fut]
                        try:
                            fut.result()
                            done += 1
                        except Exception as e:
                            print(f'Failed: {granule_ur(g)}: {e}', flush=True)
                            all_failed.append(g)

                print(f'Batch {batch_idx + 1}: {done}/{len(pending)} subsetted', flush=True)

        # per-granule failures don't block progress; only a total download exception does
        save_pickle(failed_path, all_failed)
        save_pickle(progress_path, batch_idx + 1)

    print(f'\nAll batches complete. {len(all_failed)} total failures', flush=True)
    print(f'Failed granules saved to {failed_path}', flush=True)
    if os.path.exists(progress_path):
        os.remove(progress_path)
