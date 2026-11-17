#!/usr/bin/env python3
"""Hour-by-hour Twitch viewer forecast for a game launch day.

The program reads every case file in the input directory and writes one
forecast file per case. Each forecast covers hours 2..24 of the release
day at five quantiles.

Method: anchor on the case's first-hour viewer count, then apply the
empirical distribution of log-ratios (hour h versus hour 1) pooled
across the past launches bundled with the case. Interval widths get a
floor that grows with the square root of the horizon, so late hours are
never stated with false precision. Everything is deterministic: no
clock, no randomness, and input files are processed in filename order.
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
SPREAD_FLOOR = 0.15
INNER_Z = 0.674
OUTER_Z = 1.645


def parse_timestamp(text):
    """Parse an ISO-8601 timestamp, returning None when unusable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def interpolated_quantile(ordered, tau):
    """Linear-interpolation quantile of an already sorted list."""
    count = len(ordered)
    if count == 0:
        raise ValueError("quantile of an empty list")
    if count == 1:
        return ordered[0]
    position = tau * (count - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, count - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def release_day_hours(comparable):
    """Map hour numbers 1..24 to viewer counts for a comparable launch."""
    start = parse_timestamp(comparable.get("day_start"))
    if start is None:
        return {}
    hours = {}
    for point in comparable.get("hourly") or []:
        at = parse_timestamp(point.get("from"))
        if at is None:
            continue
        hour = int((at - start).total_seconds() // 3600) + 1
        if FIRST_HOUR <= hour <= LAST_HOUR and hour not in hours:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                hours[hour] = float(viewers)
    return hours


def pooled_log_ratios(comparables):
    """Per-hour lists of log(viewers_h / viewers_1) across comparables.

    A comparable contributes to an hour only when it has a positive
    viewer count for both hour 1 and that hour, so sparse histories
    simply contribute fewer hours instead of breaking the pool.
    """
    pools = {hour: [] for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1)}
    for comparable in comparables or []:
        hours = release_day_hours(comparable)
        first = hours.get(FIRST_HOUR)
        if not first:
            continue
        for hour, viewers in hours.items():
            if hour == FIRST_HOUR:
                continue
            pools[hour].append(math.log(viewers / first))
    for hour in pools:
        pools[hour].sort()
    return pools


def hour_one_count(case):
    """The known first-hour viewer count, with fallbacks."""
    for row in case.get("known") or []:
        if row.get("hour") == FIRST_HOUR:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    known = case.get("known") or []
    if known:
        viewers = known[0].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    hourly = case.get("history_hourly") or []
    if hourly:
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


def widen_for_horizon(log_quantiles, horizon):
    """Enforce a minimum log-spread that grows with the horizon."""
    median = log_quantiles[2]
    sigma = SPREAD_FLOOR * math.sqrt(horizon)
    widened = list(log_quantiles)
    widened[0] = min(widened[0], median - OUTER_Z * sigma)
    widened[1] = min(widened[1], median - INNER_Z * sigma)
    widened[3] = max(widened[3], median + INNER_Z * sigma)
    widened[4] = max(widened[4], median + OUTER_Z * sigma)
    return widened


def forecast_case(case):
    """Forecast hours 2..24 for one case, as the output document."""
    anchor = max(1.0, hour_one_count(case))
    levels = forecast_levels(case)
    pools = pooled_log_ratios(case.get("comparables"))
    rows = []
    for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
        pool = pools[hour]
        if pool:
            log_quantiles = [interpolated_quantile(pool, tau) for tau in levels]
            log_quantiles = widen_for_horizon(log_quantiles, hour - FIRST_HOUR)
            values = [anchor * math.exp(q) for q in log_quantiles]
        else:
            sigma = SPREAD_FLOOR * math.sqrt(hour - FIRST_HOUR)
            spreads = [-OUTER_Z, -INNER_Z, 0.0, INNER_Z, OUTER_Z]
            values = [anchor * math.exp(z * sigma) for z in spreads]
        cleaned = []
        running = 0.0
        for value in values:
            rounded = round(min(MAX_VIEWERS, max(MIN_QUANTILE, value)), 2)
            rounded = max(rounded, running)
            cleaned.append(rounded)
            running = rounded
        rows.append({"hour": hour, "q": cleaned})
    return {"hours": rows}


def fallback_forecast(anchor):
    """Persistence forecast used when a case cannot be forecast normally."""
    rows = []
    for hour in range(FIRST_HOUR + 1, LAST_HOUR + 1):
        sigma = SPREAD_FLOOR * math.sqrt(hour - FIRST_HOUR)
        spreads = [-OUTER_Z, -INNER_Z, 0.0, INNER_Z, OUTER_Z]
        values = [
            max(MIN_QUANTILE, min(MAX_VIEWERS, anchor * math.exp(z * sigma)))
            for z in spreads
        ]
        cleaned = []
        running = 0.0
        for value in values:
            rounded = max(round(value, 2), running)
            cleaned.append(rounded)
            running = rounded
        rows.append({"hour": hour, "q": cleaned})
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
