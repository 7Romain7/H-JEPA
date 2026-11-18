#!/usr/bin/env python3
"""Jobs de la campagne s9 (18/10/2026) : latent SPATIAL (64 patchs, code ~/psc/code_patch,
branche git `patch`), comme Temporal Straightening. Loss multi-blocs (k = 5), 4 époques,
Pendulum (s9p) et MountainCar (s9m), 3 seeds.

  LeWM spatial      : alpha (L_curv sur la tête d'agrégation) in {0, 0.1}
  Idée 9 spatiale   : alpha in {0, 0.1} x lambda (L_scal à travers la tête) in {0, 0.1}
  Idée 9, tête mean : alpha = 0, lambda = 0.1 (moyenne des patchs au lieu du MLP appris)

  python make_jobs_s9.py --machines 42 --out jobs_s9.txt
  PSC_REMOTE_CODE='~/psc/code_patch' python cluster.py run jobs_s9.txt ..."""
import argparse

COMMON = ("data.dataset.num_steps=8 loss.multi.k=5 loss.multi.weight=1.0 "
          "+grad_ckpt=true loader.prefetch_factor=1 loader.pin_memory=false")
HIST = "model.phase=true wm.phase_include_first=true loss.scal.norm=inv loss.scal.probe=u"
ARMS = [("base_a0", "loss.curv.weight=0.0"), ("base_a0p1", "loss.curv.weight=0.1")]
ARMS += [(f"hist_a{a.replace('.', 'p')}_l{l.replace('.', 'p')}",
          f"{HIST} loss.curv.weight={a} loss.scal.weight={l}")
         for a in ("0", "0.1") for l in ("0", "0.1")]
ARMS += [("histmean_a0_l0p1", f"{HIST} loss.curv.weight=0.0 loss.scal.weight=0.1 model.agg.kind=mean")]
ENVS = [("s9p", "pendulum"), ("s9m", "mountaincar")]

p = argparse.ArgumentParser()
p.add_argument("--machines", type=int, required=True)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--out", default="jobs_s9.txt")
a = p.parse_args()
lines = []
for s in range(a.seeds):
    for prefix, env in ENVS:
        for name, ov in ARMS:
            keep = "keep" if s == 0 else "local"
            lines.append(f"{prefix}_{name}_s{s} | env TRAIN_CONFIG=lewm_spatial bash scripts/cluster/run_job.sh {keep} "
                         f"--eval-grid {env} data={env} seed={s} trainer.max_epochs={a.epochs} num_workers=6 {COMMON} {ov}")
lines = lines[:a.machines]
with open(a.out, "w", encoding="utf-8") as f:
    f.write("# job_id | commande (exécutée dans ~/psc/code_patch, venv activé)\n" + "\n".join(lines) + "\n")
print(f"{a.out} : {len(lines)} jobs ({len(ENVS)} env x {len(ARMS)} bras x {a.seeds} seeds)")
