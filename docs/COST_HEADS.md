# Têtes de coût pour la planification latente (Push-T)

Branche `cost-heads`. On garde un world model H-JEPA **gelé** (2 niveaux, checkpoints publiés
`pusht_hjepa_l2`, seeds 42/43/44) et on ne change que le **coût de planification**
c(ẑ, z_goal) que le planificateur minimise. Toutes les têtes passent par le même planificateur
plat de niveau 1 (rollout du prédicteur de niveau 1, horizon 15 pas modèle, `cost_last_n` 2,
replanification toutes les 10 actions, budget 90 actions, 50 tâches fixes
`goal_offset_75_val`, but à 75 pas d'env).

## Têtes

| nom | coût | entraînement |
|---|---|---|
| L2 | ‖z − g‖² dans le latent de niveau 1 | aucun (coût natif) |
| straight_c1_s{λ} | ‖h(z) − h(g)‖², h MLP 192→512→512→128 | 1 − cos(v_t, v_{t+1}) sur h + λ·SIGReg(h(z)), 30k pas |
| sigreg_only | idem | SIGReg seul (contrôle) |
| qrl (b) | IQE d(φ(z) → φ(g)), 64 composantes × 8 | QRL (Wang et al. 2023), contrainte d(s, s') ≤ 1, 100k pas |
| gcivl (c) | −V(z, g) | GCIVL (OGBench), récompense sparse « but atteint », γ 0,99, expectile 0,9, 100k pas |
| level2 (d) | ‖E₂(z) − E₂(g)‖², E₂ encodeur de niveau 2 d'H-JEPA | aucun (coût de `pusht_l2_project`) |

Un pas = un pas modèle = 5 pas d'env. Les têtes s'entraînent sur les latents gelés de
4 000 épisodes de `pusht_expert_train` (498k frames, `encode_latents.py`).

## Diagnostic offline (étape 2)

`cost_diagnostic.py`, sur `pusht_expert_val` (21 épisodes experts, held-out) : 939 segments
de 15 pas (but = dernière frame, 75 pas d'env comme les tâches). Spearman entre c et le nombre
de pas restants (moyenne par segment), monotonie (fraction des pas où c baisse), sur les latents
encodés et sur les latents prédits par le prédicteur en boucle ouverte avec les actions expertes
(historique 1, comme en planification). Effective rank (RankMe) de l'espace du coût sur 10k
états d'entraînement.

## Planification (étapes 1 et 4)

- **GD** : `pusht_flat` inchangé (AdamW, 800 échantillons × 90 pas), le planificateur de l'article.
- **CEM-300** : `pusht_flat_cem`, réglage par défaut de LeWM (300 échantillons, 30 élites, 30 itérations).
- **iCEM-3000** : `pusht_flat_icem`, CEM avec momentum 0,5 et 30 élites conservées,
  3000 échantillons × 60 itérations (calcul comparable au GD).

## Résultats

Push-T, 3 seeds de modèle, 939 segments de 15 pas (but à 75 pas d'env).

| coût | Spearman (encodé) | Spearman (prédit) | monotonicité (encodé / prédit) | effective rank | success rate CEM-300 | success rate iCEM-3000 | success rate GD |
|---|---|---|---|---|---|---|---|
| L2 (latent niveau 1) | 0.64 ± 0.01 | 0.45 ± 0.02 | 0.67 ± 0.00 / 0.62 ± 0.01 | 103.4 ± 0.9 (/192) | 10.0 ± 1.2 | 16.7 ± 1.3 | 40.7 ± 1.8 |
| latent niveau 2 H-JEPA (d) | 0.69 ± 0.01 | 0.55 ± 0.01 | 0.70 ± 0.00 / 0.65 ± 0.00 | 86.4 ± 0.3 (/192) | 12.7 ± 3.5 | 23.3 ± 4.1 | 43.3 ± 2.7 |
| straight_c1_s0 | 0.67 ± 0.06 | 0.57 ± 0.03 | 0.71 ± 0.03 / 0.68 ± 0.01 | 2.0 ± 0.0 (/128) | – | – | 0.7 ± 0.7 |
| straight_c1_s0.001 | 0.74 ± 0.01 | 0.64 ± 0.02 | 0.75 ± 0.01 / 0.70 ± 0.01 | 16.6 ± 0.3 (/128) | 12.0 ± 2.3 | 16.0 ± 2.3 | 22.7 ± 2.4 |
| straight_c1_s0.01 | 0.67 ± 0.00 | 0.46 ± 0.01 | 0.67 ± 0.01 / 0.61 ± 0.01 | 66.9 ± 0.5 (/128) | – | – | 28.7 ± 2.7 |
| straight_c1_s0.1 | 0.56 ± 0.02 | 0.29 ± 0.01 | 0.66 ± 0.00 / 0.58 ± 0.00 | 125.9 ± 0.1 (/128) | – | – | 21.3 ± 1.8 |
| straight_c1_s1 | 0.44 ± 0.02 | 0.22 ± 0.02 | 0.61 ± 0.01 / 0.55 ± 0.00 | 127.2 ± 0.0 (/128) | – | – | – |
| SIGReg seul (contrôle) | 0.36 ± 0.01 | 0.20 ± 0.02 | 0.59 ± 0.00 / 0.55 ± 0.00 | 127.3 ± 0.0 (/128) | – | – | – |
| QRL (b) | 0.83 ± 0.00 | 0.46 ± 0.02 | 0.76 ± 0.00 / 0.63 ± 0.01 | 9.0 ± 0.5 (/512) | – | – | – |
| GCIVL (c) | 0.88 ± 0.00 | 0.33 ± 0.03 | 0.79 ± 0.00 / 0.62 ± 0.01 | – | – | – | – |

Lignes `straight_c1_s{λ}` : straightening (poids 1) + SIGReg de poids λ. « – » = non évalué :
seules les 2 meilleures têtes au diagnostic non effondrées (straight_c1_s0.001, level2) passent en
CEM (étape 4) ; les GD des autres têtes straightening sont des contrôles ajoutés.

### Lecture

- **Collapse.** Sans SIGReg, la tête straightening tombe à un effective rank de 2/128 ; en
  planification GD elle s'effondre aussi (0,7 %). SIGReg l'évite ; SIGReg seul est le pire coût
  au diagnostic, le gain de classement vient donc du straightening.
- **Diagnostic ≠ planification.** straight_c1_s0.001 a le meilleur Spearman sur latents prédits
  (0,64) mais planifie moins bien que L2 en GD (22,7 % contre 40,7 %). Les variantes de rang plus
  élevé (λ 0,01 : 28,7 % ; λ 0,1 : 21,3 %) ne rattrapent pas L2. Le Spearman le long de
  trajectoires expertes ne mesure pas si le coût sépare des états hors de ces trajectoires, que
  l'optimiseur visite.
- **Latent de niveau 2 (d)** : meilleur coût en planification avec les trois planificateurs
  (43,3 % en GD, 23,3 % en iCEM), et 2ᵉ au diagnostic sur latents prédits parmi les têtes non
  effondrées (0,55).
- **QRL / GCIVL** : excellents sur latents encodés (0,83 / 0,88), beaucoup moins sur latents
  prédits (0,46 / 0,33) : ils ne se transfèrent pas aux états issus du prédicteur.
- **CEM.** Sur ce problème (horizon 15 × 10 dimensions d'action), le CEM optimise bien moins que
  le GD : coût prédit médian atteint sur 16 segments val 380 (CEM-300) et 274 (iCEM-3000), contre
  47 (GD) et 258 (actions expertes) (`debug_solvers.py`). iCEM et GD classent les trois coûts
  évalués dans le même ordre (niveau 2 > L2 ≥ straightening) ; en CEM-300 les écarts sont dans
  le bruit.


## Checkpoints

- **World model gelé** : checkpoints publiés `pusht_hjepa_l2` (seeds 42/43/44), téléchargés par
  `hjepa_setup.sh` (non versionnés ici).
- **Têtes entraînées** : `checkpoints/cost_heads/pusht/<tête>/seed<s>/` (24 têtes, 54 Mo), avec
  `head.pt` (module complet, `torch.save`), `args.json` (hyperparamètres) et `metrics.jsonl`
  (pertes et effective rank pendant l'entraînement). La seed de la tête = la seed du world model
  dont elle utilise les latents. Chargement : `build_head('<chemin>/head.pt')` (`h_jepa/cost_heads.py`,
  nécessite `h_jepa` dans le `PYTHONPATH`). Pour planifier avec une tête :
  `bash h_jepa/scripts/cost_heads/job.sh plan pusht_flat <seed> <chemin>/head.pt <nom>`.

## Reproduire

```bash
bash h_jepa/scripts/cost_heads/hjepa_setup.sh          # env + checkpoints + tâches (cluster)
python h_jepa/scripts/cost_heads/make_jobs_ch.py A|B|C|D  # listes de jobs (x-cluster/cluster.py)
python h_jepa/scripts/cost_heads/aggregate_ch.py $HJEPA_HOME/results
```
