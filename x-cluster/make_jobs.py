#!/usr/bin/env python3
"""Génère jobs.txt pour la campagne Idée 9 (espace des phases + L_scal), 28/09/2026.

  python make_jobs.py --machines 150 --epochs 1      # 8 bras, seeds jusqu'à ≥ 1 job par machine
  python make_jobs.py --auto --epochs 1              # compte les GPU libres maintenant

Bras (cf. « Idée 9 — Mathématiques : espace des phases et L_scal ») :
  base         LeWM d'origine (prédicteur f_θ sur l'historique, K = 3)
  phase        Idée 9 : û = h_θ(z_t, u_t, a_t) Markov strict, ẑ = z_t + û, λ_scal = 0
  pdf<λ>       Idée 9 + L_scal version PDF   ‖J_u w − γ̂w‖²
  inv<λ>       Idée 9 + L_scal invariante    ‖J_u w − γ̂w‖² / ‖J_u w‖²

Chaque job passe par scripts/cluster/run_job.sh (checkpoints sur /tmp local, quota NFS) ;
`keep` = recopier les derniers poids dans ~/psc/runs/<job>/ (seulement les premières seeds,
~70 Mo chacun). trainer.max_epochs fixe la longueur du planning de LR (warmup 1 % + cosinus) :
le choisir pour que les jobs finissent avant --deadline, sinon ils sont coupés en plein cosinus."""
import argparse, math

PHASE = "model._target_=phase_space.PhaseSpaceLeWM"
LAMBDAS = [0.01, 0.1, 1.0]
ARMS = [("base", ""), ("phase", PHASE)]
ARMS += [(f"{n}{l:g}".replace(".", "p"), f"{PHASE} loss.scal.norm={n} loss.scal.weight={l}")
         for n in ("pdf", "inv") for l in LAMBDAS]
MIN_SEEDS = 3

p = argparse.ArgumentParser()
p.add_argument("--auto", action="store_true", help="compter les GPU libres maintenant")
p.add_argument("--machines", type=int, default=0)
p.add_argument("--min-vram", type=int, default=0)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--workers", type=int, default=6, help="num_workers du DataLoader")
p.add_argument("--prefix", default="i9", help="préfixe des job ids (nouvelle campagne = nouveau préfixe)")
p.add_argument("--extra", default="", help="overrides Hydra ajoutés à tous les jobs")
p.add_argument("--data", default="pusht", help="groupe Hydra data= (pusht, reacher_sub, ...)")
p.add_argument("--eval", default="", help="évaluer en planification après l'entraînement (pusht|reacher)")
p.add_argument("--keep-seeds", type=int, default=2, help="poids recopiés sur le NFS pour les seeds < N")
p.add_argument("--out", default="jobs.txt")
a = p.parse_args()

n_machines = a.machines
if a.auto:
    import cluster
    ns = argparse.Namespace(rooms=None, hosts=None, min_vram=a.min_vram, allow_users=False)
    n_machines = len(cluster.free_machines(ns))
    print(f"{n_machines} GPU libres détectés")

n_seeds = max(MIN_SEEDS, math.ceil(n_machines / len(ARMS)))

ev = f"--eval {a.eval} " if a.eval else ""

with open(a.out, "w") as f:
    f.write("# job_id | commande (exécutée dans ~/psc/code, venv activé, $PSC_RUN_DIR = dossier du run)\n")
    # seed en boucle externe : si on manque de machines, chaque bras a déjà ses premières seeds
    for s in range(n_seeds):
        for name, ov in ARMS:
            keep = "keep" if s < a.keep_seeds else "local"
            f.write(f"{a.prefix}_{name}_s{s} | bash scripts/cluster/run_job.sh {keep} {ev}"
                    f"data={a.data} seed={s} trainer.max_epochs={a.epochs} num_workers={a.workers} {ov} {a.extra}".rstrip() + "\n")
print(f"{a.out} : {len(ARMS)} bras × {n_seeds} seeds = {len(ARMS) * n_seeds} jobs")
