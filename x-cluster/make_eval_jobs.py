#!/usr/bin/env python3
"""Génère les jobs d'évaluation en planification des checkpoints d'une campagne.

  python make_eval_jobs.py --prefix i9 --env pusht --num-eval 100 --out eval_jobs.txt

Un job par entraînement terminé (exit_code 0), plus une politique aléatoire de référence.
Le checkpoint est sur le /tmp local de la machine d'entraînement (cf. run_job.sh) : le job
est donc épinglé à cette machine (`id@machine`), sauf si une copie existe sur le NFS
(~/psc/runs/<job>/weights_epoch_*.pt, jobs lancés en `keep`).
Même --seed pour tous -> mêmes paires départ/but pour tous les modèles (comparaison appariée)."""
import argparse, json, os, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--prefix", required=True)
p.add_argument("--env", required=True, choices=["pusht", "reacher"])
p.add_argument("--num-eval", type=int, default=100)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--extra", default="")
p.add_argument("--out", default="eval_jobs.txt")
p.add_argument("--host", default="ablette.polytechnique.fr")
a = p.parse_args()

state = json.loads((HERE / "runs_state.json").read_text())
user = os.environ.get("X_USER")
target = f"{user}@{a.host}" if user else a.host
# une ligne par run : <job> <exit_code ou -> <nfs|->
cmd = (f"cd ~/psc/runs && for j in {a.prefix}_*; do "
       "echo $j $(cat $j/exit_code 2>/dev/null || echo -) "
       "$(ls $j/weights_epoch_*.pt >/dev/null 2>&1 && echo nfs || echo -); done")
out = subprocess.run(["ssh", "-o", "BatchMode=yes", target, cmd], capture_output=True, text=True).stdout
runs = [l.split() for l in out.splitlines() if l.startswith(a.prefix + "_") and len(l.split()) == 3]

lines, pinned, skipped = [], 0, []
common = f"eval.num_eval={a.num_eval} seed={a.seed} {a.extra}".strip()
for jid, rc, nfs in sorted(runs):
    if rc != "0" or jid not in state:
        skipped.append(jid)
        continue
    host = state[jid]["name"]
    pin = "" if nfs == "nfs" else f"@{host}"
    pinned += bool(pin)
    lines.append(f"ev_{jid}{pin} | bash scripts/cluster/run_eval.sh {a.env} {jid} {common}")
lines.append(f"ev_{a.prefix}_random | bash scripts/cluster/run_eval.sh {a.env} random {common}")
Path(a.out).write_text("# évaluation en planification\n" + "\n".join(lines) + "\n")
print(f"{a.out} : {len(lines)} jobs ({pinned} épinglés à leur machine), ignorés : {skipped}")
