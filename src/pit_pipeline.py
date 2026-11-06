"""Initial point-in-time monthly feature pipeline for EECS Honors Milestone 1.

Public FRED/ALFRED API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
This is a PROTOTYPE, not a validated full-history empirical dataset.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import requests
import pandas as pd

API_URL = 'https://api.stlouisfed.org/fred/series/observations'
FF_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'

# Initial subset. Broader inventory / political risk appear in metadata/data_inventory.csv.
SERIES = {
    'UNRATE': {'name': 'unemployment_rate_pp', 'transform': 'level', 'frequency': 'monthly'},
    'CPIAUCSL': {'name': 'cpi_yoy_pct', 'transform': 'yoy_pct', 'frequency': 'monthly'},
    'INDPRO': {'name': 'industrial_production_yoy_pct', 'transform': 'yoy_pct', 'frequency': 'monthly'},
    'PAYEMS': {'name': 'payrolls_yoy_pct', 'transform': 'yoy_pct', 'frequency': 'monthly'},
    'FEDFUNDS': {'name': 'fed_funds_rate_pct', 'transform': 'level', 'frequency': 'monthly'},
    'DGS10': {'name': 'ten_year_treasury_pct', 'transform': 'level', 'frequency': 'daily'},
    'GS10': {'name': 'ten_year_treasury_pct', 'transform': 'level', 'frequency': 'monthly'},
    'VIXCLS': {'name': 'vix_level', 'transform': 'level', 'frequency': 'daily'},
    'USEPUINDXM': {'name': 'us_epu_level', 'transform': 'level', 'frequency': 'monthly'},
}


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def month_ends(start_ym: str, end_ym: str) -> list[date]:
    a = datetime.strptime(start_ym, '%Y-%m')
    b = datetime.strptime(end_ym, '%Y-%m')
    if (b.year, b.month) < (a.year, a.month):
        raise ValueError('end must be >= start')
    dates = []
    year, month = a.year, a.month
    while (year, month) <= (b.year, b.month):
        dates.append(month_end(year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return dates


@dataclass
class FredClient:
    api_key: str
    request_spacing_seconds: float = 0.65

    def __post_init__(self):
        self._last_call = 0.0
        self._first_vintages = {}
        self._session = requests.Session()

    def _get(self, endpoint: str, series_id: str, params: dict) -> dict:
        if not self.api_key:
            raise ValueError('FRED_API_KEY must be set for live downloads')
        parameters = {'series_id': series_id, 'api_key': self.api_key,
                      'file_type': 'json', **params}
        for attempt in range(6):
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.request_spacing_seconds:
                time.sleep(self.request_spacing_seconds - elapsed)
            try:
                response = self._session.get(endpoint, params=parameters, timeout=45)
            except requests.RequestException as exc:
                raise RuntimeError(f'FRED network error for {series_id}: {type(exc).__name__}') from exc
            finally:
                self._last_call = time.monotonic()
            # FRED version 1 allows up to 120 requests/minute; back off if throttled.
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(60, 5 * (2 ** attempt)))
                continue
            if response.status_code >= 400:
                try:
                    error = response.json()
                    message = error.get('error_message', error.get('message', 'Unknown API error'))
                except (ValueError, AttributeError):
                    message = response.text[:250]
                # Never include the request URL in exceptions because it embeds the secret key.
                raise RuntimeError(f'FRED HTTP {response.status_code} for {series_id}: {message}')
            result = response.json()
            if 'error_code' in result:
                raise RuntimeError(f"FRED {series_id}: {result.get('error_message', 'Unknown API error')}")
            return result
        raise RuntimeError(f'FRED query exhausted retries for {series_id}')

    def earliest_vintage(self, series_id: str) -> date | None:
        """First ALFRED snapshot date; NOT necessarily original external availability."""
        if series_id not in self._first_vintages:
            result = self._get('https://api.stlouisfed.org/fred/series/vintagedates',
                               series_id, {'limit': 1, 'sort_order': 'asc'})
            vintage_dates = result.get('vintage_dates', [])
            self._first_vintages[series_id] = (date.fromisoformat(vintage_dates[0])
                                               if vintage_dates else None)
        return self._first_vintages[series_id]

    def observations(self, series_id: str, cutoff: date) -> list[dict]:
        cutoff_str = cutoff.isoformat()
        result = self._get(API_URL, series_id, {
            'realtime_start': cutoff_str,
            'realtime_end': cutoff_str,
            'observation_start': '1900-01-01',
            'observation_end': cutoff_str,
            'sort_order': 'asc',
        })
        return result.get('observations', [])


def _valid_observations(rows: list[dict], cutoff: date) -> pd.DataFrame:
    """Defense-in-depth: reject after-cutoff observations/vintage intervals."""
    out = []
    for r in rows:
        try:
            obs_date = date.fromisoformat(r['date'])
            first_known = date.fromisoformat(r.get('realtime_start', r['date']))
            last_known = date.fromisoformat(r.get('realtime_end', '9999-12-31'))
            value = float(r['value'])
        except (KeyError, TypeError, ValueError):
            continue
        if obs_date > cutoff or first_known > cutoff or last_known < cutoff:
            continue
        out.append({'observation_date': obs_date, 'value': value,
                    'vintage_start': first_known.isoformat()})
    if not out:
        return pd.DataFrame(columns=['observation_date', 'value', 'vintage_start'])
    df = pd.DataFrame(out).sort_values(['observation_date', 'vintage_start'])
    return df.drop_duplicates('observation_date', keep='last').reset_index(drop=True)


def _months_between(later: date, earlier: date) -> int:
    return 12 * (later.year - earlier.year) + later.month - earlier.month


def feature_from_rows(rows: list[dict], cutoff: date, transform: str) -> dict:
    obs = _valid_observations(rows, cutoff)
    if obs.empty:
        return {'value': None, 'last_observation_date': None, 'vintage_start': None,
                'age_months': None, 'status': 'no_asof_observation'}
    last = obs.iloc[-1]
    obs_date = last['observation_date']
    value = float(last['value'])
    status = 'ok'
    if transform == 'yoy_pct':
        previous_key = (obs_date.year - 1, obs_date.month)
        previous = obs.loc[obs['observation_date'].map(lambda d: (d.year, d.month)) == previous_key]
        if previous.empty or float(previous.iloc[-1]['value']) == 0:
            value = None
            status = 'insufficient_history'
        else:
            value = (value / float(previous.iloc[-1]['value']) - 1.0) * 100.0
    elif transform == 'mom_change':
        prior_year = obs_date.year if obs_date.month != 1 else obs_date.year - 1
        prior_month = obs_date.month - 1 if obs_date.month != 1 else 12
        previous = obs.loc[obs['observation_date'].map(lambda d: (d.year, d.month)) == (prior_year, prior_month)]
        if previous.empty:
            value = None
            status = 'insufficient_history'
        else:
            value -= float(previous.iloc[-1]['value'])
    elif transform != 'level':
        raise ValueError(f'unknown transform: {transform}')
    return {'value': value, 'last_observation_date': obs_date.isoformat(),
            'vintage_start': last['vintage_start'],
            'age_months': _months_between(cutoff, obs_date), 'status': status}


def build_panel(client, start_ym: str, end_ym: str, ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At each cutoff refetch a vintage snapshot; no future dates in feature values."""
    output, audit = [], []
    selected = list(ids)
    bad = sorted(set(selected) - set(SERIES))
    if bad:
        raise ValueError(f'unknown or not yet implemented series: {bad}')
    for cutoff in month_ends(start_ym, end_ym):
        record = {'forecast_origin': cutoff.isoformat()}
        for sid in selected:
            spec = SERIES[sid]
            # Avoid asking ALFRED for a snapshot before the series' first stored
            # vintage. Do not backfill that period from today's revised values.
            first = client.earliest_vintage(sid) if hasattr(client, 'earliest_vintage') else None
            if first is not None and cutoff < first:
                feature = {'value': None, 'last_observation_date': None,
                           'vintage_start': None, 'age_months': None,
                           'status': 'predates_first_alfred_vintage'}
            else:
                feature = feature_from_rows(client.observations(sid, cutoff), cutoff, spec['transform'])
            record[spec['name']] = feature['value']
            record[f"{spec['name']}__obs_date"] = feature['last_observation_date']
            audit.append({'forecast_origin': cutoff.isoformat(), 'series_id': sid,
                          'feature_name': spec['name'], 'transform': spec['transform'],
                          **feature})
        output.append(record)
    return pd.DataFrame(output), pd.DataFrame(audit)


def load_current_ff_monthly_target(url: str = FF_URL) -> pd.DataFrame:
    """Returns ex-post realized Mkt-RF labels only. Not a historical vintage feature."""
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if len(candidates) != 1:
            raise RuntimeError('Expected one Fama/French CSV in ZIP')
        content = zf.read(candidates[0]).decode('utf-8-sig', errors='replace')
    records = []
    # French CSV begins with description, then YYYYMM,Mkt-RF,SMB,HML,RF rows, then annual section.
    for line in content.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5 or not re.fullmatch(r'\d{6}', parts[0]):
            continue
        try:
            year, month = int(parts[0][:4]), int(parts[0][4:])
            if not (1 <= month <= 12):
                continue
            mkt_rf = float(parts[1])
        except ValueError:
            continue
        records.append({'target_month': f'{year:04d}-{month:02d}',
                        'target_next_month_excess_return_pct': mkt_rf})
    if not records:
        raise RuntimeError('No monthly Fama/French factors parsed')
    return pd.DataFrame(records).drop_duplicates('target_month')


def attach_realized_target(features: pd.DataFrame, monthly_targets: pd.DataFrame) -> pd.DataFrame:
    labeled = features.copy()
    dates = pd.to_datetime(labeled['forecast_origin'])
    labeled['target_month'] = (dates.dt.to_period('M') + 1).astype(str)
    return labeled.merge(monthly_targets, on='target_month', how='left', validate='many_to_one')


def main() -> None:
    parser = argparse.ArgumentParser(description='ALFRED as-of monthly feature pilot')
    parser.add_argument('--start', default='2020-01', help='Start forecast-origin month YYYY-MM')
    parser.add_argument('--end', default='2020-03', help='End forecast-origin month YYYY-MM')
    parser.add_argument('--series', default='UNRATE,CPIAUCSL,INDPRO,FEDFUNDS,GS10',
                        help='Comma-separated FRED IDs from implemented initial subset')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--include-realized-target', action='store_true',
                        help='Download current Fama/French Mkt-RF target labels (not a PIT feature)')
    args = parser.parse_args()
    key = os.getenv('FRED_API_KEY', '')
    if not key:
        parser.error('Set the environment variable FRED_API_KEY before running the live pipeline.')
    ids = [s.strip() for s in args.series.split(',') if s.strip()]
    features, audit = build_panel(FredClient(key), args.start, args.end, ids)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    if args.include_realized_target:
        features = attach_realized_target(features, load_current_ff_monthly_target())
    features.to_csv(folder / 'features_asof.csv', index=False)
    audit.to_csv(folder / 'asof_audit.csv', index=False)
    (folder / 'run_manifest.json').write_text(json.dumps({
        'forecast_cutoff': 'last calendar day 23:59 US Eastern (date-level approximation)',
        'data_provider': 'FRED/ALFRED observations with realtime_start = realtime_end = forecast_origin',
        'historical_coverage_warning': 'ALFRED vintage coverage varies; availability and first public introduction must be independently audited',
        'political_index_warning': 'EPU/GPR retroactively constructed histories may not have existed at early forecast dates',
        'pre_inception_warning': 'Missing features predating the first ALFRED vintage are not backfilled using revised values',
        'target_warning': 'Current Kenneth French Mkt-RF file supplies realized ex-post labels only, not vintage features',
        'series': ids, 'start': args.start, 'end': args.end,
        'includes_realized_target': args.include_realized_target,
    }, indent=2), encoding='utf-8')
    print(f"Saved {len(features)} origins, {len(audit)} audit records to {folder}")


if __name__ == '__main__':
    main()