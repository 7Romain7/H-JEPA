# PSC — LeWM-Phase : un prédicteur en espace des phases pour un world model JEPA

Branche de Joachim, indépendante de `main` (code H-JEPA). Elle contient le code du projet
PSC construit sur LeWM (`stable-worldmodel` 0.1.1) : le modèle LeWM-Phase (« Idée 9 »), la loss
`L_scal`, la loss multi-blocs, la loss de courbure du papier *Temporal Straightening*, et une
version à latent par patchs.

Le déroulé des expériences et les résultats (28 septembre – 14 octobre 2026) sont dans
[`rapport/rapport.pdf`](rapport/rapport.pdf).

## Contenu

| Dossier | Contenu |
|---|---|
| [`code/`](code) | Version principale (latent = un vecteur par image). Entraînement : `scripts/train/lewm.py` (toutes les losses), `scripts/train/phase_space.py` (LeWM-Phase et `L_scal`), `scripts/train/gym_goal.py` (Pendulum et MountainCar avec critère de succès). Collecte des données : `scripts/data/collect_gym.py`. Planification et mesures : `scripts/plan/` (`eval_wm.py`, `psc_plan.py`, `rollout_error.py`, `landscape.py`). Tests : `tests/`. |
| [`code_patchs/`](code_patchs) | Même code, plus la version à latent par patchs (64 patchs, comme *Temporal Straightening*) : `scripts/train/spatial.py`, config `scripts/train/config/lewm_spatial.yaml`. |
| [`x-cluster/`](x-cluster) | Lancement des campagnes sur les machines des salles informatiques de l'X : `cluster.py` (lancement, suivi, arrêt), `make_jobs_*.py` et `launch_*.sh` (une paire par campagne), `collect_*.py` (agrégation des résultats). |
| [`rapport/`](rapport) | Rapport LaTeX (`rapport.tex`, `rapport.pdf`) et script des figures. |

## Notes

- `x-cluster/cluster.py` a besoin d'un fichier `machines.json` (liste des machines de l'X) qui
  n'est pas publié ici, et des variables d'environnement `X_USER` (login X) et, hors campus,
  `X_JUMP`. Les résultats bruts des campagnes ne sont pas publiés non plus.
- Règle de la DSI : calcul intensif uniquement la nuit (18 h – 8 h) et le week-end.
- Dans `code_patchs/`, la tête d'agrégation de `L_curv` est appliquée aux vitesses ; le code du
  papier l'applique aux états avant de faire les différences. Voir la discussion à la fin du
  projet avant de réutiliser cette partie.
