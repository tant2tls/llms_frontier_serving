"""Offline integrity audit; does not establish telemetry validity or causality."""
from pathlib import Path
import json
import math
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def slug(value):
    value = re.sub(r'[`*_]', '', value).lower()
    return ''.join(c for c in value if c == '-' or c.isspace() or unicodedata.category(c)[0] in 'LN').replace(' ', '-')


def main():
    issues = []
    docs = list(ROOT.glob('*.md'))
    docs += [ROOT / 'archive/README.md', ROOT / 'final_presentation/README.md']
    for name in ('GLM-5.3-Flash', 'deepseek_v4_flash', 'Qwen3.8-Flash-Next-FP8'):
        docs += [ROOT / name / 'README.md', ROOT / name / 'report.md']
    links = 0
    for doc in docs:
        body = doc.read_text(encoding='utf-8')
        for match in re.finditer(r'\]\(([^)]+)\)', body):
            target = match[1].strip('<>')
            if re.match(r'^[a-z]+:', target):
                continue
            path, _, anchor = target.partition('#')
            resolved = (doc.parent / path).resolve() if path else doc
            links += 1
            if not resolved.exists():
                issues.append(f'{doc.relative_to(ROOT)}: missing link {target}')
            elif anchor and resolved.suffix == '.md' and 'archive' not in resolved.parts and 'history' not in resolved.parts:
                headings = re.findall(r'^#{1,6}\s+(.+)$', resolved.read_text(encoding='utf-8'), re.M)
                if anchor not in {slug(h) for h in headings}:
                    issues.append(f'{doc.relative_to(ROOT)}: missing anchor {target}')
    registry = json.loads((ROOT / 'tools/experiment_arms.json').read_text(encoding='utf-8'))
    registered = {a['path'] for a in registry['arms']}
    actual = {str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.glob('*/results/*') if p.is_dir()}
    if registered != actual:
        issues.append(f'Arm inventory mismatch: {sorted(registered ^ actual)}')
    points = 0
    missing_hits = 0
    for arm in registry['arms']:
        if not arm['audit_selected']:
            continue
        for file in sorted((ROOT / arm['path']).glob('*.json')):
            d = json.loads(file.read_text(encoding='utf-8'))
            points += 1
            ok = (d.get('completed') == d.get('num_prompts') and d.get('failed') == 0
                  and d.get('duration', 0) > 0 and d.get('total_output_tokens', 0) > 0)
            if not ok:
                issues.append(f'Incomplete/invalid work: {file.relative_to(ROOT)}')
            elif not math.isclose(d['output_throughput'], d['total_output_tokens']/d['duration'], abs_tol=.01):
                issues.append(f'Throughput arithmetic: {file.relative_to(ROOT)}')
            if 'new_prefix_cache_hits' not in d:
                missing_hits += 1
            elif not file.name.startswith('prefix_') and d['new_prefix_cache_hits'] != 0:
                issues.append(f'Recorded cold-prefix contamination: {file.relative_to(ROOT)}')
    slides = (ROOT / 'slides.md').read_text(encoding='utf-8')
    notes = re.findall(r'<!--(.*?)-->', slides, re.S)
    timings = re.findall(r'SLIDE (\d+) — (\d+):(\d+)', slides)
    seconds = sum(60*int(m)+int(s) for _, m, s in timings)
    if len(notes) != 24 or len(timings) != 16 or seconds != 1080:
        issues.append(f'Slide notes/timing mismatch: {len(notes)} notes, {len(timings)} main, {seconds}s')
    script = (ROOT / 'script.md').read_text(encoding='utf-8')
    spoken = re.findall(r'<!-- SCRIPT (\d+) -->\s*(.*?)\s*<!-- END SCRIPT -->', script, re.S)
    if [int(n) for n, _ in spoken] != list(range(1, 25)):
        issues.append('script.md must contain ordered spoken scripts 1–24')
    for (n, body), note in zip(spoken, notes):
        if body not in note:
            issues.append(f'Spoken script {n} differs from slides.md notes; rebuild the deck')
    script_timing = re.findall(r'\*\*(\d+):(\d+) · cumulative (\d+):(\d+)', script)
    cumulative = 0
    for m, s, cm, cs in script_timing:
        cumulative += 60*int(m)+int(s)
        if cumulative != 60*int(cm)+int(cs):
            issues.append('script.md cumulative timing mismatch')
    if len(script_timing) != 16 or cumulative != 1080:
        issues.append('script.md needs 16 main timings totaling 18 minutes')
    print(json.dumps({
        'current_documents': len(docs), 'local_links_checked': links,
        'registered_arms': len(registered),
        'selected_arms': sum(a['audit_selected'] for a in registry['arms']),
        'selected_points_checked': points, 'points_without_cache_hit_field': missing_hits,
        'main_talk_minutes': seconds/60, 'speaker_notes': len(notes), 'spoken_scripts': len(spoken), 'issues': issues,
        'scope': 'Saved arithmetic/completeness, links, inventory and notes only. No independent telemetry, causality or repeat-based validation.',
    }, indent=2))
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
