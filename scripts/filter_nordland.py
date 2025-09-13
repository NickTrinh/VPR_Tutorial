import os
import shutil
from pathlib import Path

BASE = Path('/u/erdos/csga/ntrinhvanminh/vpr-research/VPR_Tutorial')
SRC_ROOT = BASE / 'images' / 'Nordland'
DST_ROOT = BASE / 'images' / 'Nordland_filtered'
NAME_LIST = BASE / 'images' / 'Nordland_HF' / 'dataset_imageNames' / 'nordland_imageNames.txt'
SEASONS = ['spring','summer','fall','winter']
PRIORITY = {'.png': 0, '.jpg': 1, '.jpeg': 2}

# Load target stems
with open(NAME_LIST, 'r') as f:
    STEMS = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()]

# Build recursive index per season: stem -> best path (png > jpg > jpeg)
index = {s: {} for s in SEASONS}
for s in SEASONS:
    root = SRC_ROOT / s
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in PRIORITY:
            continue
        stem = p.stem
        cur = index[s].get(stem)
        if cur is None or PRIORITY[ext] < PRIORITY[cur.suffix.lower()]:
            index[s][stem] = p

# Reset destination
if DST_ROOT.exists():
    shutil.rmtree(DST_ROOT)
for s in SEASONS:
    (DST_ROOT / s).mkdir(parents=True, exist_ok=True)

# Copy exactly the requested stems
copied = {s: 0 for s in SEASONS}
missing = {s: 0 for s in SEASONS}
for stem in STEMS:
    for s in SEASONS:
        src = index[s].get(stem)
        if src is None:
            missing[s] += 1
            continue
        dst = DST_ROOT / s / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied[s] += 1

print('Copied per season:', copied)
print('Missing per season:', missing)
# Verify counts
for s in SEASONS:
    c = sum(1 for _ in (DST_ROOT / s).glob('*.*'))
    print(s, 'final count:', c)
