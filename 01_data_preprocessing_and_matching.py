"""
01_data_preprocessing_and_matching.py
- Level-3 NetCDF artifact filtering (_FillValue, missing_value, 0.05 calibration flag).
- Monthly mean aggregation of Kd_490 with duplicate prevention.
- Vectorized spatial proximity matching (Euclidean distance <= 0.40 deg) with Chlorophyll-a via cKDTree.
- Cyclical coordinate encoding (sin/cos) and harmonization into final tabular CSV.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import xarray as xr

# ==============================================================================
# CONFIGURATION & DATA PATHS 
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MICROPLASTIC_CSV_PATH = os.path.join(DATA_DIR, "merged_microplastic_data_3class.csv")
CHLOROPHYLL_FOLDER_PATH = os.path.join(DATA_DIR, "chlorophyll_csvs")
KD490_FOLDER_PATH = os.path.join(DATA_DIR, "kd490_netcdf")
OUTPUT_ALIGNED_CSV = os.path.join(DATA_DIR, "merged_dataset_with_diffuse_attenuation_coefficient.csv")


def extract_kd490_monthly_means(netcdf_folder: str, output_csv: str = None) -> pd.DataFrame:
    """Computes monthly spatial averages for Kd_490 from NetCDF archives."""
    print("Step 1: Extracting monthly spatial averages for Kd_490...")
    nc_files = sorted(glob.glob(os.path.join(netcdf_folder, "*.nc")))
    records = []

    for filepath in tqdm(nc_files, desc="Processing Kd_490 NetCDF"):
        try:
            basename = os.path.basename(filepath)
            if ".CU." in basename:
                continue

            match = re.search(r"\.(\d{4})(\d{2})\d{2}_", basename)
            if not match:
                continue
            yyyy, mm = match.groups()
            date_str = f"{mm}/{yyyy}"

            ds = xr.open_dataset(filepath)
            var_name = "Kd_490" if "Kd_490" in ds else list(ds.data_vars)[0]
            data = ds[var_name]

            data = data.where(data.notnull())
            data = data.where(data != 0.05)

            mean_val = float(data.mean().values)
            records.append({"month_year": date_str, "Kd_490_mean": mean_val})
            ds.close()
        except Exception as e:
            print(f"Skipping {filepath} due to error: {e}")

    df_kd = pd.DataFrame(records)
    df_kd = df_kd.groupby("month_year", as_index=False)["Kd_490_mean"].mean()

    if output_csv:
        df_kd.to_csv(output_csv, index=False)
        print(f"Kd_490 monthly averages saved to: {output_csv}")

    return df_kd


def match_chlorophyll(microplastic_csv: str, chlorophyll_csv_folder: str, max_dist_deg: float = 0.40) -> pd.DataFrame:
    """Matches in-situ stations to nearest Chlorophyll-a pixels via cKDTree within monthly bins."""
    print("Step 2: Spatiotemporal matching with Chlorophyll-a...")
    micro_df = pd.read_csv(microplastic_csv)
    micro_df.columns = [c.strip() for c in micro_df.columns]

    date_col = [c for c in micro_df.columns if "date" in c.lower()][0]
    lat_col_mp = [c for c in micro_df.columns if "lat" in c.lower()][0]
    lon_col_mp = [c for c in micro_df.columns if "lon" in c.lower()][0]

    micro_df["date_clean"] = micro_df[date_col].astype(str).str.strip()

    all_matches = []
    chl_files = sorted(glob.glob(os.path.join(chlorophyll_csv_folder, "*.csv")))

    for filepath in tqdm(chl_files, desc="Matching Chlorophyll CSVs"):
        filename = os.path.basename(filepath)
        match = re.search(r"\.(\d{4})(\d{2})\d{2}_\d{8}\.", filename)
        if not match:
            continue
        year, month = match.groups()
        month_key = f"{month}/{year}"

        micro_month = micro_df[micro_df["date_clean"] == month_key]
        if micro_month.empty:
            continue

        chl_df = pd.read_csv(filepath)
        chl_df.columns = [c.strip().lower() for c in chl_df.columns]
        lat_col = [c for c in chl_df.columns if "lat" in c][0]
        lon_col = [c for c in chl_df.columns if "lon" in c][0]
        chl_col = [c for c in chl_df.columns if "chlor" in c][0]

        chl_df = chl_df.dropna(subset=[lat_col, lon_col, chl_col])
        if chl_df.empty:
            continue

        chl_coords = chl_df[[lat_col, lon_col]].values
        tree = cKDTree(chl_coords)

        for _, mp_row in micro_month.iterrows():
            mp_coord = np.array([mp_row[lat_col_mp], mp_row[lon_col_mp]])
            dist, idx = tree.query(mp_coord, distance_upper_bound=max_dist_deg)

            if not np.isinf(dist):
                row_dict = mp_row.to_dict()
                row_dict["chlor_a"] = float(chl_df.iloc[idx][chl_col])
                row_dict["date_matched"] = month_key
                all_matches.append(row_dict)

    matched_df = pd.DataFrame(all_matches)
    return matched_df


def build_final_dataset(microplastic_csv: str, chlorophyll_folder: str, kd490_folder: str, output_path: str):
    df_matched = match_chlorophyll(microplastic_csv, chlorophyll_folder)
    df_kd = extract_kd490_monthly_means(kd490_folder)

    print("Step 3: Merging all environmental proxies...")
    final_df = pd.merge(
        df_matched,
        df_kd,
        left_on="date_matched",
        right_on="month_year",
        how="left"
    )

    final_df["date"] = final_df["date_matched"]
    if "month_year" in final_df.columns:
        final_df = final_df.drop(columns=["month_year"])
    if "date_matched" in final_df.columns:
        final_df = final_df.drop(columns=["date_matched"])

    lat_col = [c for c in final_df.columns if "lat" in c.lower() and "sin" not in c.lower() and "cos" not in c.lower()][0]
    lon_col = [c for c in final_df.columns if "lon" in c.lower() and "sin" not in c.lower() and "cos" not in c.lower()][0]

    final_df["latitude"] = pd.to_numeric(final_df[lat_col], errors="coerce")
    final_df["longitude"] = pd.to_numeric(final_df[lon_col], errors="coerce")

    final_df["lat_sin"] = np.sin(np.radians(final_df["latitude"]))
    final_df["lat_cos"] = np.cos(np.radians(final_df["latitude"]))
    final_df["lon_sin"] = np.sin(np.radians(final_df["longitude"]))
    final_df["lon_cos"] = np.cos(np.radians(final_df["longitude"]))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Final aligned dataset successfully saved to: {output_path} (Total Rows: {len(final_df)})")


if __name__ == "__main__":
    build_final_dataset(
        microplastic_csv=MICROPLASTIC_CSV_PATH,
        chlorophyll_folder=CHLOROPHYLL_FOLDER_PATH,
        kd490_folder=KD490_FOLDER_PATH,
        output_path=OUTPUT_ALIGNED_CSV
    )