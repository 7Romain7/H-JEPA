#!/usr/bin/env python3
"""Génère les jobs de la campagne g9 (07/10/2026) : Idée 9 avec historique et sonde
le long de u_t, sur Pendulum et MountainCar (environnements Gymnasium, inertie).

  python make_jobs_g9.py --machines 155 --epochs 6 --out jobs_g9.txt

Un préfixe par environnement (g9p = Pendulum, g9m = MountainCar) pour que
collect_results.py --prefix g9p / g9m compare les bras d'un même environnement.
Chaque job entraîne puis évalue en planification sur la même machine (run_job.sh
--eval). Boucle externe sur les seeds : si on manque de machines, chaque bras et
chaque environnement ont déjà leurs premières seeds."""
import argparse, math

ENVS = [("g9p", "pendulum"), ("g9m", "mountaincar")]
PHASE = "model._target_=phase_space.PhaseSpaceLeWM wm.phase_include_first=true"
HIST = f"{PHASE} +model.phase_history=true"
SCAL = "loss.scal.norm=inv"
ARMS = [
    ("base", ""),                                                     # LeWM, même budget
    ("phase", PHASE),                                                 # Markov strict (2 frames)
    ("hist", HIST),                                                   # 3 frames + 3 actions
    ("hist_u0p01", f"{HIST} {SCAL} loss.scal.probe=u loss.scal.weight=0.01"),
    ("hist_u0p1", f"{HIST} {SCAL} loss.scal.probe=u loss.scal.weight=0.1"),
    ("hist_r0p1", f"{HIST} {SCAL} loss.scal.probe=random loss.scal.weight=0.1"),
]

p = argparse.ArgumentParser()
p.add_argument("--machines", type=int, required=True)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--num-eval", type=int, default=100)
p.add_argument("--keep-seeds", type=int, default=1, help="poids recopiés sur le NFS pour les seeds < N")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--out", default="jobs_g9.txt")
a = p.parse_args()

per_seed = len(ENVS) * len(ARMS)
n_seeds = max(3, math.ceil(a.machines / per_seed))
lines = []
for s in range(n_seeds):
    for prefix, env in ENVS:
        for name, ov in ARMS:
            keep = "keep" if s < a.keep_seeds else "local"
            lines.append(f"{prefix}_{name}_s{s} | bash scripts/cluster/run_job.sh {keep} "
                         f"--eval {env},eval.num_eval={a.num_eval} data={env} seed={s} "
                         f"trainer.max_epochs={a.epochs} num_workers={a.workers} {ov}".rstrip())
lines = lines[:a.machines]
with open(a.out, "w") as f:
    f.write("# job_id | commande (exécutée dans ~/psc/code, venv activé, $PSC_RUN_DIR = dossier du run)\n")
    f.write("\n".join(lines) + "\n")
print(f"{a.out} : {len(lines)} jobs ({len(ENVS)} env × {len(ARMS)} bras, seeds 0..{n_seeds - 1})")
