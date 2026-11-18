#!/usr/bin/env python3
"""Jobs de la campagne c9 (14/10/2026) : grille alpha (poids de L_curv, Temporal
Straightening) x lambda (poids de L_scal), avec la loss multi-blocs (k = 5) et 4 époques,
sur Pendulum (c9p) et MountainCar (c9m).

  Idée 9 (mémoire 3) : alpha x lambda complet ; LeWM : alpha seul (L_scal n'existe pas
  pour LeWM : elle porte sur la vitesse u du prédicteur Idée 9).

  python make_jobs_c9.py --machines 150 --out jobs_c9.txt"""
import argparse, math

ENVS = [("c9p", "pendulum"), ("c9m", "mountaincar")]
ALPHAS = [0.0, 0.1, 1.0]
LAMBDAS = [0.0, 0.1, 1.0]
COMMON = ("data.dataset.num_steps=8 loss.multi.k=5 loss.multi.weight=1.0 "
          "+grad_ckpt=true loader.prefetch_factor=1 loader.pin_memory=false")
HIST = ("model._target_=phase_space.PhaseSpaceLeWM wm.phase_include_first=true "
        "+model.phase_history=true loss.scal.norm=inv loss.scal.probe=u")
tag = lambda x: f"{x:g}".replace(".", "p")

p = argparse.ArgumentParser()
p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
p.add_argument("--prefix", default="c9", help="préfixe des jobs (c9 -> c9p_, c9m_)")
p.add_argument("--machines", type=int, required=True)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--keep-seeds", type=int, default=1, help="poids recopiés sur le NFS pour les seeds < N")
p.add_argument("--out", default="jobs_c9.txt")
a = p.parse_args()
ENVS = [(f"{a.prefix}p", "pendulum"), (f"{a.prefix}m", "mountaincar")]
ARMS = [(f"base_a{tag(x)}", f"loss.curv.weight={x}") for x in a.alphas]
ARMS += [(f"hist_a{tag(x)}_l{tag(l)}", f"{HIST} loss.curv.weight={x} loss.scal.weight={l}")
         for x in a.alphas for l in a.lambdas]

n_seeds = max(2, math.ceil(a.machines / (len(ENVS) * len(ARMS))))
lines = []
for s in range(n_seeds):
    for prefix, env in ENVS:
        for name, ov in ARMS:
            keep = "keep" if s < a.keep_seeds else "local"
            lines.append(f"{prefix}_{name}_s{s} | bash scripts/cluster/run_job.sh {keep} --eval-grid {env} "
                         f"data={env} seed={s} trainer.max_epochs={a.epochs} num_workers=6 {COMMON} {ov}")
lines = lines[:a.machines]
with open(a.out, "w", encoding="utf-8") as f:
    f.write("# job_id | commande (exécutée dans ~/psc/code, venv activé)\n" + "\n".join(lines) + "\n")
print(f"{a.out} : {len(lines)} jobs ({len(ENVS)} env x {len(ARMS)} bras, seeds 0..{n_seeds - 1})")
