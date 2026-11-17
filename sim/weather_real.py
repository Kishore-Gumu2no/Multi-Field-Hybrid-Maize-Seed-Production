"""
Real weather ingestion.

Reads a NASA POWER daily CSV export and turns it into (a) DSSAT .WTH files,
one per year, and (b) the in-memory season dictionaries the scheduling
simulator consumes. This replaces the synthetic generator used in the
provisional study, so that both the crop model and the optimiser see the
same real daily record.

Expected POWER columns (community=AG, temporal=daily):
    YEAR, DOY, T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN
The export carries a header block terminated by a line containing
'-END HEADER-'; everything above it is metadata.
"""

import os
import re

import numpy as np
import pandas as pd

from .core import daily_gdd
from .dssat import write_wth

# Each quantity may arrive under any of several POWER column names: the
# viewer has renamed some over time (PRECTOTCORR is now shown simply as
# "Precipitation") and offers alternative products for the same variable.
# Listed in order of preference; the first one present is used.
POWER_COLS = {
    "tmax": ["T2M_MAX", "T2MMAX"],
    "tmin": ["T2M_MIN", "T2MMIN"],
    "rain": ["PRECTOTCORR", "PRECTOT", "PRECTOTCORR_SUM", "PRECIPITATION"],
    "srad": ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DWN_SUM", "CLRSKY_SFC_SW_DWN"],
}


def read_power_csv(path):
    """
    Parse a NASA POWER daily CSV into a tidy frame with columns
    year, doy, tmax, tmin, rain, srad (and lat/lon/elev as attributes).

    POWER writes -999 for missing values; those are interpolated, and the
    number of filled days is recorded on the frame so it can be reported.
    """
    with open(path, errors="ignore") as fh:
        raw = fh.read()

    # POWER writes e.g. "Location: latitude  14.6031   longitude 75.5976"
    # and "elevation from MERRA-2: Average for ... region = 597.04 meters"
    lat = lon = elev = None
    m = re.search(r"latitude\s+([-\d.]+)", raw, re.I)
    if m:
        lat = float(m.group(1))
    m = re.search(r"longitude\s+([-\d.]+)", raw, re.I)
    if m:
        lon = float(m.group(1))
    m = re.search(r"elevation.*?=\s*([-\d.]+)\s*met", raw, re.I | re.S)
    if m:
        elev = float(m.group(1))
    else:
        m = re.search(r"elevation[^\d-]*([-\d.]+)", raw, re.I)
        if m:
            elev = float(m.group(1))

    # locate the end of the metadata block
    lines = raw.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if "END HEADER" in ln.upper():
            start = i + 1
            break
    else:
        # no header marker: find the line that looks like the column header
        for i, ln in enumerate(lines):
            if ln.upper().startswith(("YEAR", "\"YEAR")):
                start = i
                break

    df = pd.read_csv(path, skiprows=start)
    df.columns = [c.strip().upper().strip('"') for c in df.columns]

    resolved, missing = {}, []
    for dst, candidates in POWER_COLS.items():
        hit = next((c for c in candidates if c in df.columns), None)
        if hit is None:
            missing.append(f"{dst} (any of {candidates})")
        else:
            resolved[dst] = hit
    if missing:
        raise ValueError(
            "POWER export is missing: " + "; ".join(missing) +
            f". Columns present: {sorted(df.columns)}")

    out = pd.DataFrame({
        "year": df["YEAR"].astype(int),
        "doy": df["DOY"].astype(int),
    })
    n_filled = 0
    for dst, src in resolved.items():
        s = pd.to_numeric(df[src], errors="coerce")
        bad = s.isna() | (s <= -900)
        n_filled += int(bad.sum())
        s = s.mask(bad).interpolate(limit_direction="both")
        out[dst] = s.values

    out["rain"] = out["rain"].clip(lower=0.0)
    out["srad"] = out["srad"].clip(lower=0.5)
    # keep tmin strictly below tmax; POWER occasionally crosses them
    bad = out.tmin >= out.tmax
    out.loc[bad, "tmin"] = out.loc[bad, "tmax"] - 0.5

    out.attrs.update(lat=lat, lon=lon, elev=elev,
                     n_filled=n_filled, source=os.path.basename(path),
                     columns_used=dict(resolved))
    return out


def years_available(df, min_days=360):
    """Years with a near-complete record."""
    counts = df.groupby("year").size()
    return sorted(counts[counts >= min_days].index.tolist())


def to_season(df, year, n_days=None):
    """
    One calendar year as the dict the scheduling simulator expects:
    tmax, tmin, rain and derived gdd, indexed by day-of-year - 1.
    """
    sub = df[df.year == year].sort_values("doy")
    tmax = sub.tmax.to_numpy(float)
    tmin = sub.tmin.to_numpy(float)
    rain = sub.rain.to_numpy(float)
    if n_days:
        tmax, tmin, rain = tmax[:n_days], tmin[:n_days], rain[:n_days]
    return dict(tmax=tmax, tmin=tmin, rain=rain,
                gdd=daily_gdd(tmax, tmin), year=int(year))


def export_wth(df, wsta="KARN", outdir=None, years=None):
    """Write one DSSAT .WTH per year; returns the list of years written."""
    lat = df.attrs.get("lat") or 14.47
    lon = df.attrs.get("lon") or 75.92
    elev = df.attrs.get("elev") or 600.0
    years = years or years_available(df)
    written = []
    for y in years:
        sub = df[df.year == y].sort_values("doy")
        if len(sub) < 360:
            continue
        write_wth(sub, wsta, int(y), lat, lon, elev, outdir=outdir)
        written.append(int(y))
    return written


def describe(df):
    """One-line provenance summary for the manuscript."""
    yrs = years_available(df)
    return (f"{df.attrs.get('source')}: {len(yrs)} complete years "
            f"({min(yrs)}-{max(yrs)}), lat {df.attrs.get('lat')}, "
            f"lon {df.attrs.get('lon')}, elev {df.attrs.get('elev')} m, "
            f"{df.attrs.get('n_filled')} values interpolated; "
            f"Tmax {df.tmax.min():.1f}-{df.tmax.max():.1f} C, "
            f"Tmin {df.tmin.min():.1f}-{df.tmin.max():.1f} C, "
            f"mean annual rainfall "
            f"{df.groupby('year').rain.sum().mean():.0f} mm")
