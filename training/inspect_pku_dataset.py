import json
import lzma
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset_dir = project_root / "data" / "raw" / "pku_saferlhf"

train_file = dataset_dir / "train.jsonl.xz"
test_file = dataset_dir / "test.jsonl.xz"

def inspect_file(path, count=3):
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print(f"SIZE: {path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"{'=' * 70}")

    with lzma.open(path, "rt", encoding="utf-8") as f:
        for i in range(count):
            line = f.readline()

            if not line:
                break

            record = json.loads(line)

            print(f"\nRECORD {i + 1}")
            print(f"Keys: {list(record.keys())}")

            for key, value in record.items():
                if isinstance(value, str):
                    preview = value.replace("\n", " ")[:500]
                    print(f"{key}: {preview}")
                else:
                    print(f"{key}: {value}")

inspect_file(train_file)
inspect_file(test_file)