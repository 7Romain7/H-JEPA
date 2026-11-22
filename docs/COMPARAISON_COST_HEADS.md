# Têtes de coût : comparaison des deux implémentations

Branche `cost-heads-romain`, construite sur `cost-heads`.

Deux d'entre nous ont suivi le même protocole du tuteur sans se concerter. Ce
document met les deux travaux côte à côte pour repérer ce qui se recoupe et ce
que chacun apporte que l'autre n'a pas.

| | Implémentation A | Implémentation B |
|---|---|---|
| Auteur | Reddimolk | Romain |
| Code | `h_jepa/cost_heads.py`, `train_cost_head.py`, `cost_diagnostic.py` | dossier `psc/` |
| Compte rendu | `docs/COST_HEADS.md` | ce document, `psc/README.md` |
| Résultats bruts | `checkpoints/cost_heads/` | `psc/results/` |

Les chiffres de A sont recopiés de `docs/COST_HEADS.md`. Ceux de B viennent de
`psc/results/`. Tous sont des moyennes sur les seeds 42, 43 et 44 du world model,
suivies de l'erreur standard quand elle est donnée.

## Ce qui diffère dans le montage

| Point | A | B |
|---|---|---|
| World model gelé | niveau 1 du H-JEPA à deux niveaux (`pusht_hjepa_l2`) | LeWM à plat (`pusht_lewm`, `cube_lewm`) |
| Environnements | Push-T | Push-T et OGBench Cube |
| Planner | GD du dépôt, CEM-300 et iCEM-3000 ajoutés | GD du dépôt seulement |
| Données d'entraînement des têtes | 4 000 épisodes | 3 000 épisodes sur Push-T, 1 500 sur Cube |
| Projector du straightening | MLP avec LayerNorm, sortie 128, 30 000 pas | MLP sans LayerNorm, sortie 192, 4 000 pas |
| Loss de straightening | 1 − cos entre déplacements consécutifs | la même, plus une variante « corde » |
| QRL | quasimétrique IQE, 100 000 pas | quasimétrique MRN, 20 000 pas |
| GCIVL | 100 000 pas | 30 000 pas |
| Coût en planning | coût brut de la tête | coût remis à l'échelle du L2 par égalité des médianes |
| Diagnostic | 939 segments de 15 pas | 179 fenêtres de 15 pas, plus des fenêtres de 8 et 4 pas |
| Tête ConfigWM (3e) et linear probe | non | oui |
| Comparaison tâche par tâche avec la baseline | non | oui |
| Têtes entraînées sur le latent de niveau 2 | oui | non |

La variante « corde » de B remplace le déplacement suivant par le vecteur qui va
de l'état courant au dernier état d'un segment de 8 pas.

Le diagnostic de B ne prend qu'une frame sur cinq comme point de départ alors
que celui de A part de chaque frame (`--start-stride 1`), d'où l'écart entre 179
et 939 segments sur les mêmes 21 épisodes.

## Push-T : diagnostic offline

Spearman calculé segment par segment puis moyenné, à 15 pas latents du goal.

| Coût | Spearman encodé, A | B | Spearman prédit, A | B | Monotonicité prédit, A | B | Effective rank, A | B |
|---|---|---|---|---|---|---|---|---|
| L2 | 0,64 | 0,74 | 0,45 | 0,38 | 0,62 | 0,61 | 103 / 192 | 125 / 192 |
| straightening, SIGReg 0 | 0,67 | 0,66 | 0,57 | 0,48 | 0,68 | 0,64 | 2 / 128 | 5 / 192 |
| straightening, SIGReg 0,001 | 0,74 | 0,73 | 0,64 | 0,50 | 0,70 | 0,66 | 17 / 128 | 9 / 192 |
| straightening, SIGReg 0,01 | 0,67 | 0,73 | 0,46 | 0,43 | 0,61 | 0,62 | 67 / 128 | 38 / 192 |
| straightening, SIGReg 0,1 | 0,56 | 0,61 | 0,29 | 0,32 | 0,58 | 0,58 | 126 / 128 | 93 / 192 |
| corde, SIGReg 0,01 | | 0,76 | | 0,49 | | 0,67 | | 30 / 192 |
| SIGReg seul | 0,36 | 0,50 | 0,20 | 0,14 | 0,55 | 0,54 | 127 / 128 | 112 / 192 |
| QRL | 0,83 | 0,77 | 0,46 | 0,39 | 0,63 | 0,62 | 9 / 512 | |
| GCIVL | 0,88 | 0,85 | 0,33 | 0,23 | 0,62 | 0,62 | | |

Les deux implémentations donnent le même classement. Sur les latents prédits le
straightening avec un SIGReg faible arrive en tête et le contrôle « SIGReg seul »
en dernier. QRL et GCIVL, excellents sur les latents encodés, y perdent leur
avance. L'effondrement du projector sans SIGReg est observé des deux côtés.

Les valeurs absolues de A sont plus élevées sur les latents prédits, de 0,05 à
0,14. Les world models ne sont pas les mêmes et les têtes de A sont entraînées
sept fois plus longtemps, si bien qu'on ne peut pas dire laquelle des deux causes
domine.

## Push-T : planning avec le planner GD

Success rate en pourcentage, 50 tâches par seed.

| Coût | A | B |
|---|---|---|
| L2 | 40,7 ± 1,8 | 38,0 ± 5,3 |
| L2 dans le latent de niveau 2 (3d) | 43,3 ± 2,7 | 42,0 ± 4,2 |
| straightening, SIGReg 0 | 0,7 ± 0,7 | |
| straightening, SIGReg 0,001 | 22,7 ± 2,4 | |
| straightening, SIGReg 0,01 | 28,7 ± 2,7 | |
| straightening, SIGReg 0,1 | 21,3 ± 1,8 | |
| corde, SIGReg 0,01 | | 33,3 ± 4,8 |
| QRL | | 8,7 ± 1,8 |

Les deux lignes L2 ne mesurent pas le même world model, puisque A gèle le
niveau 1 d'un H-JEPA à deux niveaux quand B gèle LeWM. La ligne 3d est en
revanche la même expérience lancée par deux chemins différents, dont les
résultats s'accordent.

Aucune tête entraînée ne bat le L2 dans l'une ou l'autre implémentation. La
variante « corde » de B est celle qui s'en approche le plus. Elle gagne 30 couples
(tâche, seed) sur 150 face au L2 et en perd 37, ce qui montre qu'elle ne réussit
pas les mêmes tâches.

A a aussi évalué CEM-300 et iCEM-3000, ce que B n'a pas fait. Le classement des
coûts y reste celui du GD alors que le CEM optimise beaucoup moins bien le coût
prédit.

## Cube : ce que seule B a mesuré

Diagnostic à 4 pas latents du goal, qui est la distance des tâches sur Cube.

| Coût | Spearman prédit | Monotonicité prédit | Success rate GD |
|---|---|---|---|
| L2 | 0,83 | 0,90 | 26,7 ± 0,7 |
| straightening consécutif, SIGReg 0,01 | 0,86 | 0,92 | |
| corde, SIGReg 0,01 | 0,85 | 0,92 | 8,0 ± 2,0 |
| SIGReg seul | 0,82 | 0,89 | |
| QRL | 0,83 | 0,91 | 12,7 ± 2,7 |
| GCIVL | 0,76 | 0,86 | |
| L2 dans le latent de niveau 2 (3d) | | | 14,0 ± 1,2 |
| ConfigWM, query cube | 0,67 | 0,80 | |
| ConfigWM, query bras | 0,51 | 0,70 | |
| ConfigWM, contrôle sans query | 0,57 | 0,74 | |

Sur Cube l'écart entre diagnostic et planning est encore plus net que sur Push-T.
Les deux têtes passées en planning égalaient ou dépassaient le L2 au diagnostic
et tombent à moins de la moitié de son success rate.

## Linear probe : ce que seule B a mesuré

R² d'une régression ridge des latents vers l'état du simulateur, ajustée sur les
latents encodés d'entraînement puis évaluée sur la validation.

| Latents | Push-T, bloc | Push-T, agent | Cube, cube | Cube, bras |
|---|---|---|---|---|
| encodés | 0,93 | 0,94 | 0,58 | 0,96 |
| prédits à 4 pas | 0,91 | 0,94 | 0,58 | 0,97 |
| prédits à 8 pas | 0,56 | 0,92 | 0,58 | 0,96 |
| prédits à 15 pas | négatif | 0,90 | 0,58 | 0,95 |

L'état du cube se décode mal dès les latents encodés. Sur Push-T le bloc se
décode bien à l'encodage puis l'information se perd dans le rollout au-delà de
8 pas, alors que la position de l'agent se conserve. Le chiffre de Cube mélange
la position et l'angle du cube, il reste à le détailler par composante.

## Ce que les deux travaux établissent ensemble

Le diagnostic offline de l'étape 2 ne prédit pas le succès en planning. Ce constat
tient avec deux world models, deux réglages de têtes, deux environnements et
trois planners. `docs/COST_HEADS.md` l'explique par le fait que le Spearman le
long de trajectoires expertes ne dit rien des états que l'optimiseur visite hors
de ces trajectoires. Le linear probe suggère une seconde cause, l'information sur
l'objet étant mal portée par les latents prédits.

Le coût dans le latent de niveau 2 est le seul à égaler ou dépasser le L2 sur
Push-T dans les deux implémentations. Sur Cube il fait nettement moins bien.

## Ce qu'il reste à trancher

1. Choisir un seul world model gelé pour la suite. LeWM à plat et le niveau 1 de
   H-JEPA donnent des baselines proches sur Push-T. Les têtes ne sont en revanche
   pas transférables de l'un à l'autre.
2. Fusionner le code. `cost_heads.py` est plus général, avec une interface
   `pair_cost` commune et une intégration dans `planning_eval.py`. Le dossier
   `psc/` contient ce qui lui manque, à savoir le support de Cube, la tête 3e, le
   probe et la comparaison tâche par tâche.
3. Décider si le coût d'une tête doit être remis à l'échelle avant le planning.
   B le fait et A ne le fait pas, sans qu'aucun des deux ait mesuré l'effet de ce
   choix.
4. Refaire le planning de la variante « corde » et de QRL dans le montage de A
   pour savoir si les écarts viennent des têtes ou du world model.
