#!/usr/bin/env python3
"""
Bake a global 10 m wind field from GFS into two small static files that
Cast & Forecast can fetch straight off GitHub Pages.

    wind-global.json   ~250 B   metadata: grid shape, cycle, units
    wind-global.bin    ~260 KB  Int16 u then v, tenths of a knot

Why this exists: Open-Meteo is a point-forecast API and meters requests by the
number of coordinates in them, so asking it for a world grid burns the quota in
minutes. GFS ships the whole field in one GRIB2 file. It just needs decoding,
which is what this script does on a CI runner instead of on the phone.

Grid convention, chosen so the browser has nothing to work out:
  row 0   = latitude -90  (SOUTH first, increasing north)
  col 0   = longitude -180
  last col= longitude +180 (a deliberate duplicate of col 0 so interpolation
            wraps across the dateline without a seam)

Usage:  python tools/make_wind.py --step 1.0 --out .
"""
import argparse, datetime as dt, json, os, struct, sys, urllib.request, urllib.error

NOMADS = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl"
          "?dir=%2Fgfs.{ymd}%2F{hh}%2Fatmos"
          "&file=gfs.t{hh}z.pgrb2full.0p50.f000"
          "&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on")
NATIVE = 0.5            # degrees, the resolution we download
MS_TO_KN = 1.9438445
UA = "cast-and-forecast-windbot/1.0 (+https://github.com/Taylor2498/Weather)"


def log(*a):
    print(*a, flush=True)


def fetch_cycle(max_back=6):
    """GFS publishes 00/06/12/18Z about 3-5 h late. Walk back until one answers
    with something that actually starts with the GRIB magic bytes — an unready
    cycle returns an HTML error page with a 200, so the status code is not
    enough to go on."""
    now = dt.datetime.now(dt.timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0,
                       hour=(now.hour // 6) * 6)
    for i in range(max_back):
        c = base - dt.timedelta(hours=6 * i)
        url = NOMADS.format(ymd=c.strftime("%Y%m%d"), hh=c.strftime("%H"))
        log(f"trying cycle {c:%Y-%m-%d %HZ}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"  no: {e}")
            continue
        if data[:4] != b"GRIB":
            log(f"  no: not GRIB ({len(data)} bytes, starts {data[:40]!r})")
            continue
        log(f"  got {len(data)/1e6:.2f} MB")
        return c, data
    raise SystemExit("no GFS cycle available - is NOMADS up?")


def decode(grib_bytes):
    """Pull the 10u/10v fields out with the ecCodes bindings. Deliberately not
    xarray/cfgrib: fewer moving parts to break on a runner, and this file has
    exactly two messages in it."""
    import numpy as np
    import eccodes

    tmp = "gfs_wind.grb2"
    with open(tmp, "wb") as f:
        f.write(grib_bytes)

    fields = {}
    meta = {}
    with open(tmp, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                name = eccodes.codes_get(gid, "shortName")
                if name in ("10u", "10v"):
                    ni = eccodes.codes_get(gid, "Ni")
                    nj = eccodes.codes_get(gid, "Nj")
                    lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                    lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                    vals = eccodes.codes_get_values(gid).reshape(nj, ni)
                    fields[name] = vals
                    meta = dict(ni=ni, nj=nj, lat1=lat1, lon1=lon1)
                    log(f"  {name}: {nj}x{ni}, first point {lat1},{lon1}")
            finally:
                eccodes.codes_release(gid)
    os.remove(tmp)

    missing = {"10u", "10v"} - set(fields)
    if missing:
        raise SystemExit(f"GRIB was missing {missing}")
    return fields["10u"], fields["10v"], meta


def regrid(u, v, meta, step):
    import numpy as np

    # GFS scans north-to-south; we want south-first so the browser can index
    # straight off latitude without a flip.
    if meta["lat1"] > 0:
        u, v = u[::-1, :], v[::-1, :]

    # Longitude arrives as 0..359.5. Roll it to -180..179.5.
    half = meta["ni"] // 2
    u, v = np.roll(u, half, axis=1), np.roll(v, half, axis=1)

    stride = int(round(step / NATIVE))
    if stride < 1:
        raise SystemExit(f"--step cannot be finer than the {NATIVE}deg source")
    u, v = u[::stride, ::stride], v[::stride, ::stride]

    # Duplicate the -180 column at +180 so bilinear interpolation crosses the
    # dateline continuously instead of falling off the edge of the array.
    u = np.concatenate([u, u[:, :1]], axis=1)
    v = np.concatenate([v, v[:, :1]], axis=1)

    return u * MS_TO_KN, v * MS_TO_KN


def validate(u, v, nx, ny):
    import numpy as np

    if u.shape != (ny, nx) or v.shape != (ny, nx):
        raise SystemExit(f"shape mismatch: {u.shape} vs expected {(ny, nx)}")
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise SystemExit("field contains NaN or inf")
    spd = np.sqrt(u * u + v * v)
    mx, mean = float(spd.max()), float(spd.mean())
    # A global 10 m field always has *some* wind and never has 300 kt of it.
    if not (5 < mx < 250):
        raise SystemExit(f"peak wind {mx:.1f} kt is not physical")
    if not (2 < mean < 40):
        raise SystemExit(f"mean wind {mean:.1f} kt is not physical")
    if abs(u).max() < 0.5:
        raise SystemExit("u component is flat - wrong field?")
    log(f"  sanity: peak {mx:.1f} kt, mean {mean:.1f} kt")
    return mx, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=1.0,
                    help="output grid spacing in degrees (0.5 = native, 1.0 default)")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    cycle, raw = fetch_cycle()
    u, v, meta = decode(raw)
    u, v = regrid(u, v, meta, a.step)
    ny, nx = u.shape
    mx, mean = validate(u, v, nx, ny)

    # Int16 in tenths of a knot: 260 KB for a 1-degree world, no base64 tax,
    # and one Int16Array view decodes it in the browser.
    import numpy as np
    packed = np.concatenate([u.ravel(), v.ravel()])
    packed = np.clip(np.round(packed * 10), -32767, 32767).astype("<i2")

    os.makedirs(a.out, exist_ok=True)
    binp = os.path.join(a.out, "wind-global.bin")
    with open(binp, "wb") as f:
        f.write(packed.tobytes())

    metadata = {
        "format": "cf-wind-1",
        "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nx": nx, "ny": ny,
        "lo1": -180.0, "la1": -90.0,
        "dx": a.step, "dy": a.step,
        "scale": 0.1, "unit": "kn",
        "peak": round(mx, 1), "mean": round(mean, 1),
        "source": "NOAA GFS 0.5deg via NOMADS",
    }
    with open(os.path.join(a.out, "wind-global.json"), "w") as f:
        json.dump(metadata, f, separators=(",", ":"))

    log(f"wrote wind-global.bin {os.path.getsize(binp)/1024:.0f} KB "
        f"({nx}x{ny} @ {a.step}deg) for cycle {metadata['cycle']}")


if __name__ == "__main__":
    main()
