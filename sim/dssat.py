"""
DSSAT CERES-Maize driver.

Runs the real crop model (DSSAT-CSM v4.8, built from source) to obtain
anthesis dates for a given cultivar, sowing date and weather year, and
provides the machinery to calibrate and validate the fast thermal-time
surrogate that the scheduling optimiser actually calls.

Why a surrogate is necessary: the rolling-horizon policy evaluates on the
order of 10^4-10^5 (field, sowing day, offset, scenario) combinations per
season. A DSSAT run takes ~0.1-1 s, so calling it inside the optimiser is
not tractable. Instead we run DSSAT across a design grid, calibrate the
thermal-time model against those runs, and report the agreement in days.
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

DSSAT_HOME = "/DSSAT48"
DSSAT_BIN = os.path.join(DSSAT_HOME, "dscsm048")
CULTIVAR_FILE = os.path.join(DSSAT_HOME, "Genotype", "MZCER048.CUL")


# ---------------------------------------------------------------------------
# Cultivar coefficients
# ---------------------------------------------------------------------------

def read_cultivars(path=CULTIVAR_FILE):
    """
    Parse MZCER048.CUL.

    P1     thermal time, emergence to end of juvenile phase (degree-days, Tb 8 C)
    P2     photoperiod sensitivity (days delay per hour above 12.5 h)
    P5     thermal time, silking to physiological maturity
    G2, G3 kernel number and filling-rate coefficients (not used here)
    PHINT  phyllochron interval (degree-days per leaf)

    Flowering date is governed mainly by P1, P2 and PHINT, which is why two
    inbred parent lines with different coefficients flower at different times
    from the same sowing date -- the mechanism this study depends on.
    """
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith(("!", "*", "@")) or not line.strip():
                continue
            m = re.match(
                r"^(\S{6})\s+(.{1,20}?)\s{2,}(\S+)\s+(\S{6})\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                line.rstrip())
            if not m:
                continue
            rows.append(dict(
                var=m.group(1), name=m.group(2).strip(), eco=m.group(4),
                P1=float(m.group(5)), P2=float(m.group(6)), P5=float(m.group(7)),
                G2=float(m.group(8)), G3=float(m.group(9)), PHINT=float(m.group(10)),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment file generation
# ---------------------------------------------------------------------------

TEMPLATE_MZX = os.path.join(DSSAT_HOME, "Maize", "UFGA8201.MZX")


def build_mzx(ingeno, year, doy, wsta, template=TEMPLATE_MZX):
    """
    Build an experiment file by editing a real, working DSSAT experiment in
    place rather than generating one from scratch.

    DSSAT's reader is strictly column-positional and the shipped files use
    CRLF endings; a hand-written file is very easy to misalign. Starting from
    UFGA8201.MZX and substituting only the cultivar, planting date, weather
    station and simulation start keeps every other column byte-identical, and
    holds soil, field, initial conditions and management fixed across runs --
    which is exactly the controlled comparison we want.
    """
    with open(template, "rb") as fh:
        raw = fh.read().decode("latin-1")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    pdate = _yyyddd(year, doy)
    sdate = _yyyddd(year, max(1, doy - 1))
    out = []
    section = ""
    for ln in lines:
        s = ln
        if s.startswith("*"):
            section = s.strip().upper()
        # Keep only the first treatment. This must be scoped to the
        # TREATMENTS block: rows in the irrigation and fertiliser blocks have
        # the same leading-integer shape and deleting them corrupts the file.
        if section.startswith("*TREATMENTS") and "TNAME" not in s \
                and re.match(r"^\s*[2-9]\d*\s+\d+\s+\d+\s+\d+\s+\S", s):
            continue
        # cultivar
        if re.match(r"^\s*1 MZ \S{6}", s):
            s = re.sub(r"^(\s*1 MZ )(\S{6})", r"\g<1>" + ingeno, s)
        # planting date (PLANTING DETAILS row)
        elif re.match(r"^\s*1 \d{5}\s+-99\s+[\d.]+\s+[\d.]+\s+S\s", s):
            s = re.sub(r"^(\s*1 )(\d{5})", r"\g<1>" + pdate, s)
        # simulation start date
        elif re.match(r"^\s*1 GE\s", s):
            s = re.sub(r"(\s+S\s+)(\d{5})", r"\g<1>" + sdate, s)
        # Switch water and nitrogen balances off, so phenology is simulated
        # under potential conditions. Two reasons: the template's irrigation
        # and fertiliser events are pegged to its own planting date and become
        # invalid once we move sowing, and for calibrating a phenology
        # surrogate we want thermal time uncontaminated by stress.
        elif re.match(r"^\s*1 OP\s", s):
            s = re.sub(r"^(\s*1 OP\s+)(\S+)(\s+)(\S+)", r"\g<1>N\g<3>N", s)
        elif re.match(r"^\s*1 MA\s", s):
            s = re.sub(r"^(\s*1 MA\s+)(\S+)(\s+)(\S+)(\s+)(\S+)",
                       r"\g<1>\g<2>\g<3>N\g<5>N", s)
        # automatic planting window, if present
        elif re.match(r"^\s*1 PL\s+\d{5}\s+\d{5}", s):
            s = re.sub(r"(\s*1 PL\s+)(\d{5})(\s+)(\d{5})",
                       r"\g<1>" + pdate + r"\g<3>" + pdate, s)
        # weather station in the FIELDS row (4-char code, keep width)
        if re.match(r"^\s*1 \S{8} \S{4}\s", s) and "ID_FIELD" not in s:
            s = re.sub(r"^(\s*1 \S{8} )(\S{4})", r"\g<1>" + wsta[:4], s)
        out.append(s)
    return nl.join(out)


MZX_TEMPLATE = """*EXP.DETAILS: {exid}MZ SYNC STUDY

*GENERAL
@PEOPLE
 anthesis synchrony scheduling study
@ADDRESS
 simulated
@SITE
 {wsta}

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 0 0 SYNC                       1  1  0  1  1  0  1  0  0  0  0  0  1

*CULTIVARS
@C CR INGENO CNAME
 1 MZ {ingeno} parent line

*FIELDS
@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME
 1 SYNC0001 {wsta}     -99     0 DR000     0     0 00000 -99    180  {soil}  field
@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR
 1         {lon:>15.4f} {lat:>15.4f} {elev:>9.0f}                 0     0     0     0   -99   -99

*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    MZ {icdat}   100     0     1     1   -99  1000    .8     0   100    15 init
@C  ICBL  SH2O  SNH4  SNO3
 1     5  .250   1.0   5.0
 1    15  .250   1.0   5.0
 1    30  .250   1.0   4.0
 1    60  .250    .5   3.0
 1    90  .250    .5   2.0
 1   120  .250    .5   2.0
 1   180  .250    .5   2.0

*PLANTING DETAILS
@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME
 1 {pdate}   -99   7.0   7.0     S     R    75     0     5   -99   -99   -99   -99     0                        -99

*FERTILIZERS (INORGANIC)
@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME
 1 {pdate} FE005 AP002     5    60   -99   -99   -99   -99   -99 basal

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL
 1 GE              1     1     S {sdate}  2150 SYNC                     MZCER
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              Y     Y     N     N     N     N     N     Y     M
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL
 1 ME              M     M     E     R     S     C     R     1     G     S     2
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     N     R     R     M
@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT
 1 OU              N     Y     Y     1     N     N     N     N     N     N     0     N     N

@  AUTOMATIC MANAGEMENT
@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN
 1 PL          {pdate} {pdate}    40   100    30    40    10
@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF
 1 IR             30    50   100 GS000 IR001    10  1.00
@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF
 1 NI             30    50    25 FE001 GS000
@N RESIDUES    RIPCN RTIME RIDEP
 1 RE            100     1    20
@N HARVEST     HFRST HLAST HPCNP HRCNP
 1 HA              0 {hlast}   100     0
"""


def _yyyddd(year, doy):
    return f"{year % 100:02d}{doy:03d}"


def run_dssat(ingeno, year, doy, wsta, soil="IBMZ910014",
              lat=14.47, lon=75.92, elev=600, workdir=None, keep=False):
    """
    Run DSSAT for one cultivar sown on one day; return days from planting to
    anthesis, or None if the model did not reach anthesis.

    `wsta` is the 4-character weather station code; {wsta}{yy}01.WTH must
    already exist in /DSSAT48/Weather.
    """
    tmp = workdir or tempfile.mkdtemp(prefix="dssat_", dir="/DSSAT48")
    os.makedirs(tmp, exist_ok=True)
    exid = "SYNC0001"
    pdate = _yyyddd(year, doy)
    sdate = _yyyddd(year, max(1, doy - 10))
    icdat = sdate
    hlast = _yyyddd(year + 1, 60) if doy > 200 else _yyyddd(year, 365)

    mzx = build_mzx(ingeno, year, doy, wsta)
    path = os.path.join(tmp, "SYNC0001.MZX")
    with open(path, "w", newline="") as fh:
        fh.write(mzx)

    try:
        subprocess.run([DSSAT_BIN, "MZCER048", "A", "SYNC0001.MZX"],
                       cwd=tmp, capture_output=True, timeout=120,
                       input=b"\n")
        summ = os.path.join(tmp, "Summary.OUT")
        if not os.path.exists(summ):
            return None
        with open(summ, errors="ignore") as fh:
            lines = [l for l in fh if l.strip() and not l.startswith(("*", "!", "@"))]
        if not lines:
            return None
        # Treatment and field names contain spaces, so splitting on whitespace
        # does not line up with the header. The date block is unambiguous
        # instead: SDAT PDAT EDAT ADAT MDAT HDAT, each a 7-digit YYYYDDD.
        dates = re.findall(r"\b([12]\d{3}[0-3]\d{2})\b", lines[-1])
        if len(dates) < 4:
            return None
        pd_, ad_ = dates[1], dates[3]
        if ad_ in ("-99", "", "0"):
            return None
        d = _days_between(pd_, ad_)
        return d if 0 < d < 365 else None
    finally:
        if not keep and workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def _days_between(a, b):
    """Days between two YYYYDDD dates."""
    ya, da = int(a[:4]), int(a[4:])
    yb, db = int(b[:4]), int(b[4:])
    base = pd.Timestamp(year=ya, month=1, day=1) + pd.Timedelta(days=da - 1)
    end = pd.Timestamp(year=yb, month=1, day=1) + pd.Timedelta(days=db - 1)
    return int((end - base).days)


# ---------------------------------------------------------------------------
# Weather file writing
# ---------------------------------------------------------------------------

def write_wth(df, wsta, year, lat, lon, elev, outdir=None):
    """
    Write a DSSAT .WTH file for one calendar year.

    `df` needs columns: doy, srad (MJ/m2/d), tmax, tmin, rain (mm).
    """
    outdir = outdir or os.path.join(DSSAT_HOME, "Weather")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{wsta}{year % 100:02d}01.WTH")
    tav = (df.tmax.mean() + df.tmin.mean()) / 2
    amp = (df.groupby(df.doy // 30).tmax.mean().max()
           - df.groupby(df.doy // 30).tmin.mean().min()) / 2
    with open(path, "w") as fh:
        fh.write(f"*WEATHER DATA : {wsta} synthesis\n\n")
        fh.write("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT\n")
        fh.write(f"  {wsta} {lat:>8.3f} {lon:>8.3f} {elev:>5.0f} "
                 f"{tav:>5.1f} {amp:>5.1f} {2.0:>5.1f} {2.0:>5.1f}\n")
        fh.write("@DATE  SRAD  TMAX  TMIN  RAIN\n")
        for _, r in df.iterrows():
            fh.write(f"{year % 100:02d}{int(r.doy):03d} {r.srad:>5.1f} "
                     f"{r.tmax:>5.1f} {r.tmin:>5.1f} {r.rain:>5.1f}\n")
    return path
