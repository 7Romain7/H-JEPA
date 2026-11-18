#!/usr/bin/env python3
"""Jobs de réévaluation des checkpoints g9 (10/10/2026) : solveur {gd, cem} x contexte
H {1, 2, 3}, sur les modèles déjà entraînés (5 époques). Un job par checkpoint,
épinglé à la machine qui le détient (/tmp local) sauf si une copie existe sur le NFS ;
plus oracle et politique aléatoire par environnement.

  python make_eval_jobs_g9.py --num-eval 100 --out eval_g9.txt"""
import argparse, json, os, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = {"g9p": "pendulum", "g9m": "mountaincar"}
p = argparse.ArgumentParser()
p.add_argument("--num-eval", type=int, default=100)
p.add_argument("--tag", default="ev2", help="préfixe des jobs d'évaluation")
p.add_argument("--out", default="eval_g9.txt")
p.add_argument("--host", default="ablette.polytechnique.fr")
a = p.parse_args()

state = json.loads((HERE / "runs_state.json").read_text())
user = os.environ.get("X_USER")
target = f"{user}@{a.host}" if user else a.host
cmd = ("cd ~/psc/runs && for j in g9p_* g9m_*; do echo $j $(cat $j/exit_code 2>/dev/null || echo -) "
       "$(ls $j/weights_epoch_*.pt >/dev/null 2>&1 && echo nfs || echo -); done")
out = subprocess.run(["ssh", "-o", "BatchMode=yes", target, cmd], capture_output=True, text=True).stdout
runs = [l.split() for l in out.splitlines() if l[:4] in ("g9p_", "g9m_") and len(l.split()) == 3]

lines, pinned, skipped = [], 0, []
common = f"eval.num_eval={a.num_eval}"
for prefix, env in ENVS.items():
    for w in ("oracle", "random"):
        lines.append(f"{a.tag}_{prefix}_{w} | bash scripts/cluster/run_eval_grid.sh {env} {w} {common}")
for jid, rc, nfs in sorted(runs):
    if rc != "0" or jid not in state:
        skipped.append(jid)
        continue
    pin = "" if nfs == "nfs" else f"@{state[jid]['name']}"
    pinned += bool(pin)
    lines.append(f"{a.tag}_{jid}{pin} | bash scripts/cluster/run_eval_grid.sh {ENVS[jid[:3]]} {jid} {common}")
Path(a.out).write_text("# réévaluation g9 : solveur x contexte\n" + "\n".join(lines) + "\n")
print(f"{a.out} : {len(lines)} jobs ({pinned} épinglés), ignorés : {skipped}")
