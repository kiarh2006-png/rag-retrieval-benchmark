"""Run one-factor-at-a-time parameter experiments; each config changes one parameter."""
import copy
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_benchmark import run


def main():
    with open(ROOT / "configs/baseline.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    experiments = [
        ("baseline", {}),
        ("topk_10", {"top_k": 10}),       # only top_k changes
        ("rrf_k_20", {"rrf_k": 20}),      # only RRF k changes
    ]
    for name, changes in experiments:
        cfg = copy.deepcopy(base)
        cfg.update(changes)
        cfg["experiment_name"] = name
        print("\n" + "=" * 72 + "\nEXPERIMENT:", name, "CHANGES:", changes)
        run(cfg)


if __name__ == "__main__":
    main()
