#!/usr/bin/env python3
"""Agrège la réévaluation en planification (run_eval_grid.sh) : taux de succès par
configuration x solveur x contexte H, pour une campagne d'entraînement donnée.

  python collect_planning.py --train g9p --tag ev2     # Pendulum
  python collect_planning.py --train g9m --tag ev2     # MountainCar

Sorties dans resultats/<tag>_<train>/ : planning.md, planning.csv."""
import argparse, io, json, math, os, re, subprocess, tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--tag", default="ev2")
p.add_argument("--host", default="ablette.polytechnique.fr")
p.add_argument("--no-fetch", action="store_true")
a = p.parse_args()
out = HERE / "resultats" / f"{a.tag}_{a.train}"
out.mkdir(parents=True, exist_ok=True)

if not a.no_fetch:
    user = os.environ.get("X_USER")
    target = f"{user}@{a.host}" if user else a.host
    cmd = (f"cd ~/psc/runs && tar czf - $(ls -d {a.tag}_{a.train}_*/eval_*.json "
           f"{a.tag}_{a.train}_*/exit_code 2>/dev/null) 2>/dev/null")
    data = subprocess.run(["ssh", "-o", "BatchMode=yes", target, cmd], capture_output=True).stdout
    if data:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(out / "runs", filter="data")

rows, refs = [], {}
for f in sorted((out / "runs").glob(f"{a.tag}_{a.train}_*/eval_*.json")):
    job = f.parent.name[len(a.tag) + 1:]
    d = json.loads(f.read_text())
    tag = f.stem[len("eval_"):]
    if job.endswith(("_oracle", "_random")):
        refs[job.rsplit("_", 1)[1]] = d["success_rate"]
        continue
    m = re.fullmatch(rf"{a.train}_(.+)_s(\d+)", job)
    solver, h = re.fullmatch(r"(\w+)_h(\d)", tag).groups()
    rows.append({"arm": m.group(1), "seed": int(m.group(2)), "solver": solver,
                 "H": int(h), "success": d["success_rate"]})
df = pd.DataFrame(rows)
df.to_csv(out / "planning.csv", index=False)
if df.empty:
    raise SystemExit("aucun résultat")

ORDER = ["base", "phase", "hist", "hist_u0p01", "hist_u0p1", "hist_r0p1"]
arms = [x for x in ORDER if x in set(df.arm)] + sorted(set(df.arm) - set(ORDER))


def welch_p(x, y):
    x, y = x.dropna(), y.dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx, vy = x.var() / len(x), y.var() / len(y)
    if vx + vy == 0:
        return float("nan")
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    from scipy import stats
    dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return 2 * stats.t.sf(abs(t), dof)


cols = sorted({(s, h) for s, h in zip(df.solver, df.H)}, key=lambda c: (c[0] != "gd", c[1]))
lines = [f"# Planification {a.train} ({a.tag})", "",
         f"Oracle (vrai simulateur) : {refs.get('oracle', float('nan')):.1f} % · "
         f"politique aléatoire : {refs.get('random', float('nan')):.1f} %. "
         "Taux de succès moyen ± erreur standard sur les seeds (n entre parenthèses).", "",
         "| bras | " + " | ".join(f"{s} H={h}" for s, h in cols) + " |",
         "|---|" + "---|" * len(cols)]
for arm in arms:
    cells = []
    for s, h in cols:
        v = df[(df.arm == arm) & (df.solver == s) & (df.H == h)].success
        cells.append(f"{v.mean():.1f} ± {v.std() / math.sqrt(len(v)) if len(v) > 1 else 0:.1f} ({len(v)})"
                     if len(v) else "—")
    lines.append(f"| {arm} | " + " | ".join(cells) + " |")
for ref in ("base", "hist"):
    if ref not in arms:
        continue
    lines += ["", f"p (Welch) vs {ref} :", "",
              "| bras | " + " | ".join(f"{s} H={h}" for s, h in cols) + " |", "|---|" + "---|" * len(cols)]
    for arm in arms:
        if arm == ref:
            continue
        cells = []
        for s, h in cols:
            sel = lambda r: df[(df.arm == r) & (df.solver == s) & (df.H == h)].success
            pv = welch_p(sel(arm), sel(ref))
            cells.append("—" if math.isnan(pv) else f"{pv:.2g}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
(out / "planning.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
