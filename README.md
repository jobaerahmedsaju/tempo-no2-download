# TEMPO NO2 Download

Batched, resumable downloader for TEMPO NO2 L2 V04 data via [earthaccess](https://github.com/nsidc/earthaccess).

For each granule intersecting the target bounding box and time range, the script:
1. Downloads the raw file via earthaccess
2. Spatially masks pixels outside the bounding box to the variable's fill value
3. Subsets to a fixed set of geolocation, product, and support-data variables
4. Saves the result under `<output_dir>/<product>/<YYYY>/<MM>/<DD>/`, deletes the raw file

Progress and per-granule failures are pickled to disk (`batch_progress.pkl`, `failed_granules.pkl`), so an interrupted run can be restarted from where it left off.

## Requirements

```
pip install -r requirements.txt
```

You'll also need a NASA Earthdata Login account (https://urs.earthdata.nasa.gov/). `earthaccess.login(persist=True)` will prompt for credentials on first run and cache them locally.

## Configuration

Edit the constants near the top of `download_tempo_no2.py`:

- `tempo_dir` — output directory
- `start_dt`, `end_dt` — date range
- `bbox` — (west, south, east, north)
- `batch_size` — granules per batch
- `geo_vars`, `product_vars`, `support_vars` — variables to keep

## Usage

```
python download_tempo_no2.py
```

Re-running the script resumes from the last completed batch.
