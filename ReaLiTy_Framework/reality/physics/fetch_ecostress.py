"""Re-fetch the ECOSTRESS readings behind reflectance_lut.yaml.

Run this to audit or regenerate the measured values:

    python reality/physics/fetch_ecostress.py asphalt concrete

Each `measured_pct` in the LUT is a spectrum from the NASA/JPL ECOSTRESS Spectral
Library v1.0 (speclib.jpl.nasa.gov), linearly interpolated at 0.905 um. The site
serves its search and raw spectra from the site root, not /library.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

BASE = "https://speclib.jpl.nasa.gov"
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}
TARGET_UM = 0.905


class _KeepPost(urllib.request.HTTPRedirectHandler):
    """307/308 must re-issue the POST rather than degrade it to a GET."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code in (307, 308):
            return urllib.request.Request(newurl, data=req.data, headers=req.headers,
                                          method="POST")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_KeepPost)


def listing(searchtype: str, classsel: str = "All", maxhits: int = 200) -> List[Tuple[str, str]]:
    """Return (filename, row text) for every spectrum of a given type."""
    params = {"searchtype": searchtype, "classsel": classsel, "subclass": "All",
              "mname": "", "xstart": "", "xstop": "", "maxhits": maxhits, "wavelength": "Any"}
    req = urllib.request.Request(f"{BASE}/ecospeclib_list",
                                 data=urllib.parse.urlencode(params).encode(), headers=HEADERS)
    html = _OPENER.open(req, timeout=90).read().decode("utf-8", "replace")
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        match = re.search(r"getplot\('([^']+)'\)", tr)
        if match:
            rows.append((match.group(1), re.sub(r"<[^>]+>", " ", tr).strip()))
    return rows


def reflectance_at(filename: str, target_um: float = TARGET_UM):
    """Interpolate one spectrum's reflectance (percent) at ``target_um``."""
    url = f"{BASE}/ecospeclibdata/{filename}"
    text = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}), timeout=90
    ).read().decode("utf-8", "replace")

    meta: Dict[str, str] = {}
    points = []
    for line in text.splitlines():
        pair = re.match(r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s*$", line)
        if pair:
            points.append((float(pair.group(1)), float(pair.group(2))))
        elif ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    points.sort()
    if not points or not points[0][0] <= target_um <= points[-1][0]:
        return None, meta  # e.g. the gold/brass plates are thermal-infrared only
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target_um <= x1:
            t = 0.0 if x1 == x0 else (target_um - x0) / (x1 - x0)
            return y0 + t * (y1 - y0), meta
    return None, meta


if __name__ == "__main__":
    keywords = [k.lower() for k in sys.argv[1:]] or ["asphalt", "concrete"]
    for searchtype in ("manmade", "vegetation", "soil", "nonphotosyntheticvegetation"):
        for filename, row in listing(searchtype):
            if not any(k in row.lower() for k in keywords):
                continue
            value, meta = reflectance_at(filename)
            printed = "n/a  " if value is None else f"{value:6.2f}"
            print(f"{printed}%  {meta.get('Name', '?'):34s} {searchtype:12s} {filename}")
