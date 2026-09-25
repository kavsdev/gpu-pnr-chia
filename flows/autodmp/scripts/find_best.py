"""Find AutoDMP's best trial via its own `hpbandster`/`tuner_analyze.get_candidates()` ranking (the authoritative
way AutoDMP itself ranks trials, not a guessed sort order -- see `find_best_json.py` for a lighter, dependency-free
alternative that reads the same log directly).

Usage: python find_best.py --tuner-dir /path/to/AutoDMP/tuner --log-dir /path/to/tuner_logs/<design>
"""
import argparse
import sys


def find_best(tuner_dir: str, log_dir: str):
    sys.path.append(tuner_dir)
    import hpbandster.core.result as hpres
    from tuner_analyze import get_candidates

    res = hpres.logged_results_to_HBS_result(log_dir)
    all_runs = res.get_all_runs()
    best_run, min_obj = None, float("inf")
    for run in all_runs:
        if run.info is None:
            continue
        obj = run.info.get("objective", float("inf"))
        if obj < min_obj:
            min_obj, best_run = obj, run
    print("Best run by 'objective' field:", best_run.config_id if best_run else "None")
    print("Min obj:", min_obj)
    try:
        print(get_candidates(res, num=1))
    except Exception as e:
        print("get_candidates failed:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuner-dir", required=True, help="AutoDMP's tuner/ dir, needs tuner_analyze.py importable")
    ap.add_argument("--log-dir", required=True, help="the run's tuner_logs/<design> directory")
    args = ap.parse_args()
    find_best(args.tuner_dir, args.log_dir)
