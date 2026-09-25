"""Find AutoDMP's best trial from its raw HpBandSter `results.json` log, by its own `objective` field.

Reads the log's own JSON-lines format directly, rather than via `hpbandster.core.result` (see `find_best.py` for
that heavier alternative, which also validates against `tuner_analyze.py`'s own `get_candidates()` ranking).

Usage: python find_best_json.py path/to/tuner_logs/<design>/results.json
"""
import argparse
import json


def find_best(results_json_path: str):
    best_config_id, min_obj = None, float("inf")
    with open(results_json_path) as f:
        for line in f:
            data = json.loads(line)
            config_id = data[0]
            info = data[3]
            if info and "info" in info and "objective" in info["info"]:
                obj = info["info"]["objective"]
                if obj < min_obj:
                    min_obj, best_config_id = obj, config_id
    return best_config_id, min_obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json")
    args = ap.parse_args()
    best_config_id, min_obj = find_best(args.results_json)
    print(f"Best config by 'objective': {best_config_id}, val: {min_obj}")
