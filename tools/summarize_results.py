"""Summarize explicitly selected experiment arms without hiding provenance."""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='*', help='Registered arm paths relative to workspace root')
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'tools/experiment_arms.json').read_text(encoding='utf-8'))
    known = {a['path']: a for a in registry['arms']}
    paths = [p.replace('\\', '/').rstrip('/').removeprefix('./') for p in args.arms]
    if not paths:
        paths = [p for p, a in known.items() if a['role'] == 'primary']
    records = []
    for path in paths:
        if path not in known or not known[path]['audit_selected']:
            parser.error(f'Arm is not in the reviewed selection: {path}. Inspect excluded raw evidence directly.')
        pattern = '*.json' if args.arms else 'batch_isl16k_c*.json'
        for file in sorted((ROOT / path).glob(pattern)):
            data = json.loads(file.read_text(encoding='utf-8'))
            if not args.arms and data.get('max_concurrency') not in (1, 4, 16, 64):
                continue
            complete = data.get('completed') == data.get('num_prompts') and data.get('failed') == 0
            if not complete or data.get('duration', 0) <= 0:
                raise ValueError(f'Incomplete/invalid workload: {file.relative_to(ROOT)}')
            hits = data.get('new_prefix_cache_hits')
            cache = 'intentional reuse; telemetry not captured' if file.name.startswith('prefix_') else 'unknown' if hits is None else f'recorded delta={hits}; scrape validity not independently verified'
            records.append({
                'arm': path, 'point': file.stem, 'role': known[path]['role'],
                'output_tok_s': round(data['output_throughput'], 3),
                'median_ttft_ms': round(data['median_ttft_ms'], 3),
                'median_tpot_ms': round(data['median_tpot_ms'], 3),
                'completed': data['completed'], 'cache_evidence': cache,
            })
    if args.csv:
        if records:
            writer = csv.DictWriter(sys.stdout, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    else:
        print('Named deployment results; not quality-adjusted or isolated architecture effects.')
        for r in records:
            print(f"{r['arm']}/{r['point']}: {r['output_tok_s']:.1f} out tok/s; "
                  f"TTFT {r['median_ttft_ms']:.1f} ms; TPOT {r['median_tpot_ms']:.1f} ms")
        print(f'{len(records)} points. See experiments.md and report.md for controls and limitations.')


if __name__ == '__main__':
    main()
