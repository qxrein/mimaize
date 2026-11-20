#!/usr/bin/env python3
"""After adding search seeds: invalidate final GPTQ / downstream results whose selected configuration changed.

The final GPTQ evaluation (ext_gptq.py) uses, per method, the configuration with the best *calibration*
score among all seeds of the GPTQ-aware search. When new seeds are added, the selected configuration can
change; this script removes the stale entries from results_ext/gptq/<model>.json and <model>_extra.json
so that re-running ext_gptq.py recomputes exactly those (it skips entries that are still present).
A record of the selected seed per method is kept in results_ext/gptq/<model>_selected.json.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_ext")
for m in ["opt125m", "opt350m", "qwen0.5b", "llama3.2-1b"]:
    rp = os.path.join(RES, "llm_gptq", m, "runs.jsonl")
    if not os.path.exists(rp):
        continue
    runs = [json.loads(l) for l in open(rp) if l.strip()]
    sel_p = os.path.join(RES, "gptq", f"{m}_selected.json")
    old = json.load(open(sel_p)) if os.path.exists(sel_p) else {}
    new = {}
    for meth in ["ga", "cmaes", "abc_v2", "probe_mckp", "fisher_mckp"]:
        rs = [r for r in runs if r["method"] == meth and abs(r["budget"] - 0.75) < 1e-6]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["calib_score"])
        new[meth] = {"seed": best["seed"], "n_seeds": len(rs), "config": best["config"]}
        # before this script existed, the selection was made among seeds 0-2
        prev_cfg = old.get(meth, {}).get("config")
        if prev_cfg is None:
            prev3 = [r for r in rs if r["seed"] <= 2]
            prev_cfg = max(prev3, key=lambda r: r["calib_score"])["config"] if prev3 else None
        if prev_cfg != best["config"]:
            for fn in (f"{m}.json", f"{m}_extra.json"):
                p = os.path.join(RES, "gptq", fn)
                if os.path.exists(p):
                    d = json.load(open(p))
                    if meth in d:
                        del d[meth]
                        json.dump(d, open(p, "w"), indent=2)
            print(f"[refresh] {m} {meth}: selected seed changed -> seed {best['seed']}; final results invalidated")
        else:
            print(f"[refresh] {m} {meth}: unchanged (seed {best['seed']} of {len(rs)})")
    json.dump(new, open(sel_p, "w"), indent=2)
