"""
dataloader.py
-------------
Functions for fetching station metadata from the TOAR-II API
and loading pre-downloaded CSV files from disk.
"""
import os
import pandas as pd
import requests
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://toar-data.fz-juelich.de/api/v2/stationmeta"

GLOBALMETA_FIELDS: list[str] = [
    "htap_region_tier1_year2010",
    "dominant_landcover_year2012",
    "landcover_description_25km_year2012",
    "dominant_ecoregion_year2017",
    "ecoregion_description_25km_year2017",
    "climatic_zone_year2016",
    "distance_to_major_road_year2020",
    "mean_stable_nightlights_1km_year2013",
    "mean_stable_nightlights_5km_year2013",
    "max_stable_nightlights_25km_year2013",
    "max_stable_nightlights_25km_year1992",
    "mean_population_density_250m_year2015",
    "mean_population_density_5km_year2015",
    "max_population_density_25km_year2015",
    "mean_population_density_250m_year1990",
    "mean_population_density_5km_year1990",
    "max_population_density_25km_year1990",
    "mean_nox_emissions_10km_year2015",
    "mean_nox_emissions_10km_year2000",
    "mean_topography_srtm_alt_90m_year1994",
    "mean_topography_srtm_alt_1km_year1994",
    "max_topography_srtm_relative_alt_5km_year1994",
    "min_topography_srtm_relative_alt_5km_year1994",
    "stddev_topography_srtm_relative_alt_5km_year1994",
]

# ---------------------------------------------------------------------------
# CSV loading functions
# ---------------------------------------------------------------------------
def load_station_data(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """Load a pre-downloaded station metadata CSV from the data directory.

    Args:
        filename: Name of the CSV file (e.g. 'stationglobalmetadata.csv').
        data_dir: Path to the data directory. Defaults to 'data/'.

    Returns:
        DataFrame with station metadata.

    Raises:
        FileNotFoundError: If the file does not exist at the resolved path.
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path, comment="#")

def get_station_coord(station_code: str) -> dict:
    """Fetch coordinates and area type for a single station from the TOAR-II API.

    Args:
        station_code: The TOAR station identifier (e.g. 'GB0682A').

    Returns:
        Dict with keys: 'lat', 'lon', 'alt', 'type_of_area'.

    Raises:
        requests.HTTPError: If the API returns a non-200 status.
        KeyError: If expected fields are missing in the response.
    """
    response = requests.get(f"{BASE_URL}/{station_code}", timeout=30)
    response.raise_for_status()
    data_point = response.json()
    return {
        "lat": data_point["coordinates"]["lat"],
        "lon": data_point["coordinates"]["lng"],
        "alt": data_point["coordinates"]["alt"],
        "type_of_area": data_point["type_of_area"],
    }


def get_test_data_from_station_code(station_codes: dict[str, list[str]]) -> pd.DataFrame:
    """Fetch test station data for hand-labelled stations and return a DataFrame.

    Args:
        station_codes: Dict mapping area type ('urban', 'suburban', 'rural')
                       to a list of station codes.

    Returns:
        DataFrame with columns: lat, lon, area_code, altitude,
        type_of_area_toar, type_of_area_gmap. Rows with
        type_of_area_toar == 'unknown' are excluded.
    """
    records = []
    for area_type, codes in station_codes.items():
        for code in codes:
            try:
                coord = get_station_coord(code)
                records.append({
                    "lat": coord["lat"],
                    "lon": coord["lon"],
                    "area_code": code,
                    "altitude": coord["alt"],
                    "type_of_area_toar": coord["type_of_area"],
                    "type_of_area_gmap": area_type,
                })
            except (requests.HTTPError, KeyError) as e:
                print(f"Warning: skipping station '{code}' — {e}")

    df = pd.DataFrame(records)
    return df[df["type_of_area_toar"] != "unknown"].reset_index(drop=True)


def get_N_first_station_data(
    data_points: list | None = None,
    page_size: int = 1000,   
    n_pages: int = 26,      
) -> pd.DataFrame:
    """Fetch station metadata from the TOAR-II API.

    Fetches `page_size * n_pages` stations in total (default ~26 000).

    Args:
        data_points: Pre-fetched list of station dicts. If None, data is
                     fetched from the API using `page_size` and `n_pages`.
        page_size: Number of records per API request (page size).
        n_pages: Number of pages (API requests) to perform.

    Returns:
        DataFrame with station coordinates and global metadata fields.

    Raises:
        requests.HTTPError: If any API request returns a non-200 status.
    """
    if data_points is None:
        data_points = []
        for i in range(n_pages):
            url = f"{BASE_URL}/?limit={page_size}&offset={i * page_size}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data_points.extend(response.json())

    records = []
    for dp in data_points:
        try:
            record = {
                "lat": dp["coordinates"]["lat"],
                "lon": dp["coordinates"]["lng"],
                "area_code": dp["codes"][0],
                "timezone": dp["timezone"],
                "type_of_area": dp["type_of_area"],
                "altitude": dp["coordinates"]["alt"],
            }
            for field in GLOBALMETA_FIELDS:
                record[field] = dp["globalmeta"].get(field)
            records.append(record)
        except (KeyError, IndexError) as e:
            print(f"Warning: skipping malformed entry — {e}")

    return pd.DataFrame(records)


def get_data_from_station_codes(
    station_codes: list[str] | str,
    output_format: str = "dataframe",
) -> pd.DataFrame | list[dict]:
    """Fetch station metadata for an explicit list of station codes.

    Calls ``{BASE_URL}/{code}`` once per code and parses the response
    identically to :func:`get_N_first_station_data`, so the returned
    DataFrame has the same columns:

        lat, lon, area_code, timezone, type_of_area, altitude,
        <all GLOBALMETA_FIELDS>

    Args:
        station_codes: Single station code string or list of station codes.
        output_format: ``'dataframe'`` (default) returns a ``pd.DataFrame``;
                       ``'dicts'`` returns a raw list of record dicts.

    Returns:
        DataFrame or list of dicts with one entry per successfully fetched
        station.  Stations that raise an HTTP or parsing error are skipped
        with a printed warning.

    Raises:
        ValueError: If *output_format* is not ``'dataframe'`` or ``'dicts'``.
    """
    if output_format not in ("dataframe", "dicts"):
        raise ValueError("output_format must be 'dataframe' or 'dicts'.")

    if isinstance(station_codes, str):
        station_codes = [station_codes]

    records = []
    for code in station_codes:
        try:
            response = requests.get(f"{BASE_URL}/{code}", timeout=30)
            response.raise_for_status()
            dp = response.json()
            record = {
                "lat": dp["coordinates"]["lat"],
                "lon": dp["coordinates"]["lng"],
                "area_code": dp["codes"][0],
                "timezone": dp["timezone"],
                "type_of_area": dp["type_of_area"],
                "altitude": dp["coordinates"]["alt"],
            }
            for field in GLOBALMETA_FIELDS:
                record[field] = dp["globalmeta"].get(field)
            records.append(record)
        except (requests.HTTPError, KeyError, IndexError) as e:
            print(f"Warning: skipping station '{code}' — {e}")

    if output_format == "dicts":
        return records
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
#__main__ guard for running the script directly to refresh data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    print("Fetching station metadata from TOAR-II API...")
    df = get_N_first_station_data(page_size=1000, n_pages=26)
    today = datetime.today().strftime("%d.%m.%Y")
    output_path = os.path.join("data", f"stationglobalmetadata_{today}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} stations to '{output_path}'")

