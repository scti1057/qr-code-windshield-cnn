import json
from collections import defaultdict
from pathlib import Path

split_path = Path("data/splits/patches_split_seed42.json")

def source_id(p: str) -> str:
    name = Path(p).name
    return name.split("__")[0]  # anpassen falls dein Naming anders ist

data = json.loads(split_path.read_text(encoding="utf-8"))
splits = data["splits"]

seen = {}
overlaps = defaultdict(list)

for split_name, rows in splits.items():
    for r in rows:
        sid = source_id(r["path"])
        if sid in seen and seen[sid] != split_name:
            overlaps[sid].append((seen[sid], split_name))
        else:
            seen[sid] = split_name

print("Unique source images:", len(seen))
print("Overlapping sources across splits:", len(overlaps))
if overlaps:
    example = list(overlaps.items())[:10]
    print("Examples:", example)
