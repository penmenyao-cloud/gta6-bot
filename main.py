#!/usr/bin/env python3
"""Hour-by-hour Twitch viewer forecast for a game launch day, version 2.

The program reads every case file in the input directory and writes one
forecast file per case. Each forecast covers hours 2..24 of the release
day at five quantiles.

Method: for each forecast hour separately, fit a local-linear
regression of log viewers on log hour-1 viewers across the past
launches bundled with the case, weighting launches by how close their
hour-1 count is to this case's (a Gaussian kernel of bandwidth 1 in log
space). The fitted centre is then shrunk toward the case's own hour-1
level: fully trusted for small hour-1 counts, which usually mean the
game had not unlocked yet at UTC midnight, and shrunk 30% of the way
toward persistence for large hour-1 counts, which mean the launch was
already live and hyped. Spreads come from honest out-of-sample error:
for every pooled launch in turn, the same procedure is refit without
it, and the quantiles are the median and robust spread of those
leave-one-out residuals. Everything is deterministic: no clock, no
randomness, and input files are processed in filename order.
"""

import json
import math
import os
import sys
from datetime import datetime

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
FIRST_HOUR = 1
LAST_HOUR = 24
MAX_VIEWERS = 1_000_000_000
MIN_QUANTILE = 0.01
KERNEL_BW = 1.0
MIN_POOL = 5
FALLBACK_SIGMA = 0.15

# Exact normal quantiles for the standard levels; other levels use the
# rational approximation in norm_ppf.
STANDARD_Z = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}

# Shrinkage schedule: log hour-1 at or below SMALL_ANCHOR keeps the
# kernel centre untouched; at or above BIG_ANCHOR the centre moves 30%
# of the way toward the hour-1 level; between them it interpolates.
SMALL_ANCHOR_LOG = math.log(5000.0)
BIG_ANCHOR_LOG = math.log(50000.0)
BIG_ANCHOR_ALPHA = 0.7


def parse_timestamp(text):
    """Parse an ISO-8601 timestamp, returning None when unusable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def norm_ppf(tau):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    for level, z in STANDARD_Z.items():
        if abs(tau - level) < 1e-12:
            return z
    # Coefficients of the Acklam approximation, accurate to ~1e-9.
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    low = 0.02425
    if tau < low:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if tau > 1.0 - low:
        q = math.sqrt(-2.0 * math.log(1.0 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def shrink_alpha(log_anchor):
    """Weight on the kernel centre (the rest goes to persistence)."""
    if log_anchor <= SMALL_ANCHOR_LOG:
        return 1.0
    if log_anchor >= BIG_ANCHOR_LOG:
        return BIG_ANCHOR_ALPHA
    frac = (log_anchor - SMALL_ANCHOR_LOG) / (BIG_ANCHOR_LOG - SMALL_ANCHOR_LOG)
    return 1.0 + frac * (BIG_ANCHOR_ALPHA - 1.0)


def series_from_comparable(comparable):
    """Log viewer counts by release-day hour for one past launch.

    Returns None when the launch has no positive hour-1 count, so a
    history that never covers a release day (such as the target game's
    own pre-release history) contributes nothing.
    """
    if not isinstance(comparable, dict):
        return None
    start = parse_timestamp(comparable.get("day_start"))
    if start is None:
        return None
    hours = {}
    for point in comparable.get("hourly") or []:
        if not isinstance(point, dict):
            continue
        at = parse_timestamp(point.get("from"))
        if at is None:
            continue
        hour = int((at - start).total_seconds() // 3600) + 1
        if FIRST_HOUR <= hour <= LAST_HOUR and hour not in hours:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                hours[hour] = math.log(float(viewers))
    if FIRST_HOUR not in hours:
        return None
    return hours


def kernel_centre(x0, xs, ys):
    """Local-linear kernel fit of y on x, evaluated at x0.

    A Gaussian kernel of bandwidth KERNEL_BW weights nearby launches;
    when the total weight is tiny (an anchor far from every launch),
    the fit falls back to an unweighted regression over the pool. The
    fitted value is clamped to the pool's observed range expanded by
    1.5 log units on each side, so an anchor beyond every pooled
    launch (a record-breaking hour 1) can extrapolate only modestly
    instead of running away on a noisy local slope.
    """
    weights = [math.exp(-0.5 * ((x - x0) / KERNEL_BW) ** 2) for x in xs]
    if sum(weights) < 3.0:
        weights = [1.0] * len(xs)
    total = sum(weights)
    xbar = sum(w * x for w, x in zip(weights, xs)) / total
    ybar = sum(w * y for w, y in zip(weights, ys)) / total
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(weights, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(weights, xs, ys))
    slope = sxy / sxx if sxx > 0.0 else 0.0
    fitted = ybar + slope * (x0 - xbar)
    return min(max(fitted, min(ys) - 1.5), max(ys) + 1.5)


def robust_sigma(residuals):
    """Robust scale of residuals: 1.4826 times the median absolute
    deviation, with a standard-deviation fallback for degenerate
    spreads and a floor so intervals never collapse to a point."""
    centre = median(residuals)
    mad = median([abs(r - centre) for r in residuals])
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        if len(residuals) > 1:
            mean = sum(residuals) / len(residuals)
            sigma = math.sqrt(sum((r - mean) ** 2 for r in residuals) / len(residuals))
        else:
            sigma = 0.5
    return max(sigma, 0.05)


def forecast_hour(log_anchor, pool, hour, z_values):
    """Five log quantiles for one hour from a pool of past launches.

    pool holds (log hour-1, log viewers at this hour) pairs. The centre
    is the shrunken kernel fit; the spread is the median and robust
    scale of nested leave-one-out residuals of the same procedure.
    """
    if len(pool) < MIN_POOL:
        sigma = FALLBACK_SIGMA * math.sqrt(hour - FIRST_HOUR)
        return [log_anchor + z * sigma for z in z_values]
    xs = [pair[0] for pair in pool]
    ys = [pair[1] for pair in pool]
    alpha = shrink_alpha(log_anchor)
    centre = kernel_centre(log_anchor, xs, ys)
    centre = alpha * centre + (1.0 - alpha) * log_anchor
    residuals = []
    for index in range(len(pool)):
        sub_xs = xs[:index] + xs[index + 1:]
        sub_ys = ys[:index] + ys[index + 1:]
        held_alpha = shrink_alpha(xs[index])
        refit = kernel_centre(xs[index], sub_xs, sub_ys)
        refit = held_alpha * refit + (1.0 - held_alpha) * xs[index]
        residuals.append(ys[index] - refit)
    offset = median(residuals)
    sigma = robust_sigma(residuals)
    return [centre + offset + z * sigma for z in z_values]


def hour_one_count(case):
    """The known first-hour viewer count, with fallbacks."""
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == FIRST_HOUR:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    known = case.get("known") or []
    if known and isinstance(known[0], dict):
        viewers = known[0].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    hourly = case.get("history_hourly") or []
    if hourly and isinstance(hourly[-1], dict):
        viewers = hourly[-1].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    return 1.0


def forecast_levels(case):
    """The five quantile levels to forecast, from the case or default."""
    levels = case.get("levels")
    if (
        isinstance(levels, list)
        and len(levels) == 5
        and all(isinstance(t, (int, float)) and 0.0 < t < 1.0 for t in levels)
    ):
        return sorted(float(t) for t in levels)
    return list(DEFAULT_LEVELS)


def clean_quantiles(log_quantiles):
    """Turn log quantiles into valid viewer counts: finite, positive,
    capped, rounded, and non-decreasing."""
    cleaned = []
    running = 0.0
    for log_value in log_quantiles:
        value = math.exp(max(-20.0, min(25.0, log_value)))
        rounded = round(min(MAX_VIEWERS, max(MIN_QUANTILE, value)), 2)
        rounded = max(rounded, running)
        cleaned.append(rounded)
        running = rounded
    return cleaned


def forecast_case(case):
    """Forecast hours 2..24 for one case, as the output document."""
    anchor = max(1.0, hour_one_count(case))
    log_anchor = math.log(anchor)
    levels = forecast_levels(case)
    z_values = [norm_ppf(tau) for tau in levels]
    series = []
    for comparable in case.get("comparables") or []:
        hours = series_from_comparable(comparable)
        if hours is not None:
            series.append(hours)
    rows = []
    for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
        pool = [
            (hours[FIRST_HOUR], hours[hour])
            for hours in series
            if hour in hours
        ]
        log_quantiles = forecast_hour(log_anchor, pool, hour, z_values)
        rows.append({"hour": hour, "q": clean_quantiles(log_quantiles)})
    return {"hours": rows}


def fallback_forecast(anchor):
    """Persistence forecast used when a case cannot be forecast normally."""
    log_anchor = math.log(max(1.0, anchor))
    rows = []
    for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
        sigma = FALLBACK_SIGMA * math.sqrt(hour - FIRST_HOUR)
        z_values = [STANDARD_Z[tau] for tau in DEFAULT_LEVELS]
        rows.append({
            "hour": hour,
            "q": clean_quantiles([log_anchor + z * sigma for z in z_values]),
        })
    return {"hours": rows}


def run(input_dir, output_dir):
    """Forecast every case file in the input directory.

    A case that fails to load or forecast still gets a fallback
    forecast, so one bad file never aborts the remaining cases.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in names:
        stem = name[:-3]
        case = None
        try:
            with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
        except Exception as exc:
            anchor = 1.0
            if isinstance(case, dict):
                try:
                    anchor = max(1.0, hour_one_count(case))
                except Exception:
                    anchor = 1.0
            print(f"warning: {name} failed ({exc}); writing fallback", file=sys.stderr)
            result = fallback_forecast(anchor)
        with open(os.path.join(output_dir, stem + ".out"), "w", encoding="utf-8") as handle:
            json.dump(result, handle)
            handle.write("\n")


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    run(input_dir, output_dir)


if __name__ == "__main__":
    main()
