"""Kuszob alapu metrikak: a fejleszto halmaz EER-kuszobe, valtoztatas nelkul az eval halmazon.

Hasznalat:
    python scripts/threshold_metrics.py --runs out_mel out_lfcc --out results

A --runs mappakban <verzio>_seed<n> almappakat keres (pl. v03_seed0). Egy futashoz kell
egy dev_scores.csv (evaluate.py --split dev) es egy eval_scores.csv (evaluate.py --split eval);
a ket fajl lehet kulon --runs mappaban is, ha az almappa neve egyezik.
Pozitiv osztaly: hamis (spoof). Dontes: pontszam >= kuszob -> hamis.
Kimenet: runs_threshold.csv (futasonkent), summary_threshold.csv (verzionkent atlag, szoras)
es per_attack_miss.csv (atengedett hamis felvetelek aranya tamadasonkent).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RUN_RE = re.compile(r"^v(\d)(\d)_seed(\d+)$")


def eer_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def metrics(df, thr):
    y = (df.label == "spoof").to_numpy()
    p = df.score.to_numpy() >= thr
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    fp, tn = int((p & ~y).sum()), int((~p & ~y).sum())
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    prec_b, rec_b = tn / (tn + fn), tn / (tn + fp)
    f1 = 2 * prec * rec / (prec + rec)
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b)
    _, eer = eer_threshold(df.score.to_numpy(), y.astype(int))
    return {
        "eval_eer": eer * 100, "auc": roc_auc_score(y, df.score) * 100,
        "precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
        "fpr": fp / (fp + tn) * 100, "fnr": fn / (fn + tp) * 100,
        "bal_acc": (rec + rec_b) / 2 * 100, "acc": (tp + tn) / len(df) * 100,
        "f1_bona": f1_b * 100, "macro_f1": (f1 + f1_b) / 2 * 100,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    files = defaultdict(dict)
    for root in args.runs:
        for d in Path(root).iterdir():
            if d.is_dir() and RUN_RE.match(d.name):
                for kind in ("dev", "eval"):
                    f = d / f"{kind}_scores.csv"
                    if f.exists():
                        files[d.name][kind] = f

    rows, miss = [], []
    for name in sorted(files):
        if set(files[name]) != {"dev", "eval"}:
            print(f"kihagyva (hianyzik dev vagy eval): {name}")
            continue
        major, minor, seed = RUN_RE.match(name).groups()
        version = f"{major}.{minor}"
        dev, ev = pd.read_csv(files[name]["dev"]), pd.read_csv(files[name]["eval"])
        thr, dev_eer = eer_threshold(dev.score.to_numpy(), (dev.label == "spoof").astype(int).to_numpy())
        rows.append({"version": version, "seed": int(seed), "threshold": thr, "dev_eer": dev_eer * 100, **metrics(ev, thr)})
        for sysid, grp in ev[ev.label == "spoof"].groupby("system"):
            miss.append({"version": version, "seed": int(seed), "system": sysid, "miss": (grp.score < thr).mean() * 100})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs_threshold.csv", index=False)
    summary = runs.drop(columns=["seed", "threshold"]).groupby("version").agg(["mean", "std"])
    summary.to_csv(out / "summary_threshold.csv")
    m = pd.DataFrame(miss)
    m.to_csv(out / "per_attack_miss.csv", index=False)
    print(summary.xs("mean", axis=1, level=1).round(2).to_string())
    print()
    print(m.groupby(["system", "version"]).miss.mean().unstack().round(1).to_string())


if __name__ == "__main__":
    main()
