# Cadre commun des têtes de coût

Version 1, proposition du 24 novembre 2026. Tant que l'équipe ne l'a pas validée,
chaque ligne du tableau de la section 9 reste ouverte.

Le premier tour a été fait deux fois, dans deux montages différents (voir
`docs/COST_HEADS.md` et `docs/COMPARAISON_COST_HEADS.md` sur les branches
`cost-heads` et `cost-heads-romain`). Ce mémo fixe ce qui doit être identique
pour tout le monde afin que chaque binôme travaille sur un arm sans que les
résultats cessent d'être comparables.

## 1. La question

La question reste celle du tuteur, à savoir quel coût planifie le mieux à world
model gelé et à planner fixé. Un arm est une famille de coûts qui partagent le
même signal d'entraînement. Le coût est la seule chose qu'un binôme modifie.

## 2. Ce que le premier tour a montré

Success rate sur Push-T avec le planner à gradient, 50 tâches et 3 seeds, dans
le montage de `cost-heads`.

| Coût | Spearman sur latents prédits | Success rate |
|---|---|---|
| L2 | 0,45 | 40,7 ± 1,8 |
| L2 dans le latent de niveau 2 | 0,55 | 43,3 ± 2,7 |
| GCIVL | 0,33 | 33,3 ± 2,7 |
| straightening, SIGReg 0,001 | 0,64 | 22,7 ± 2,4 |
| QRL | 0,46 | 6,7 ± 0,7 |

Aucune tête entraînée ne bat le L2. Le diagnostic offline classe les têtes dans
un ordre que le planning ne confirme pas. Le straightening, premier au
diagnostic, perd 18 points sur le L2 alors que GCIVL, classée sous le L2, est la
meilleure des têtes entraînées. Le second montage (LeWM à plat, Push-T et Cube)
aboutit aux mêmes constats. Son linear probe ajoute que l'état du cube se décode
mal des latents (R² de 0,58) et que la position du bloc de Push-T se perd dans le
rollout au-delà de 8 pas.

## 3. Ce qui est gelé

| Élément | Valeur proposée | Raison |
|---|---|---|
| World model | niveau 1 de `<env>_hjepa_l2`, checkpoints publiés, seeds 42, 43 et 44 | la ligne « latent de niveau 2 » devient un simple changement de coût dans le même world model ; le code de `cost-heads` est déjà construit dessus |
| Environnements | Push-T et OGBench Cube | ceux du tuteur |
| Planner | configs `<env>_flat` du dépôt, sans retouche | seul planner qui a une valeur publiée ; d'après `cost-heads` le CEM optimise moins bien le coût prédit |
| Tâches de test | les 50 tâches publiées de chaque environnement | références de l'article |
| Données des têtes | latents de 4 000 épisodes d'entraînement, la tête de la seed *s* ne voyant que le world model de la seed *s* | déjà en place sur Push-T |
| Validation du diagnostic | `pusht_expert_val` (21 épisodes) et `cube_single_expert_val` (200 épisodes) | jeux held-out du dépôt |

LeWM à plat reste disponible comme contrôle de robustesse pour les têtes qui
entrent dans le tableau final.

Toute modification d'une ligne de ce tableau change la version du cadre et se
décide à plusieurs. Les résultats obtenus avant et après ne se mélangent pas
dans un même tableau.

Un world model réentraîné, par exemple avec un terme de straightening dans sa
loss, sort de ce cadre. Ses résultats se présentent dans un tableau séparé.

## 4. Ce qu'une tête a le droit de faire

1. Elle hérite de `CostHead` dans `h_jepa/cost_heads.py`, définit
   `pair_cost(z, g)` et se branche par l'option `++cost_head=`. Le rollout, le
   solver et le reste de `planning_eval.py` ne sont pas touchés.
2. Son coût est différentiable par rapport au latent prédit.
3. Elle ne s'entraîne que sur le jeu d'entraînement de l'environnement. Elle
   peut utiliser l'encodeur gelé, les actions et les latents prédits par le
   prédicteur gelé.
4. Elle déclare son signal parmi quatre catégories, qui sont aucun signal de
   tâche, le temps écoulé, une étiquette de succès ou l'état du simulateur. Cette
   catégorie figure dans le tableau final, parce qu'elle est l'axe de comparaison
   du projet.
5. Elle reste un petit MLP dont la taille est écrite dans `args.json`.
6. Elle ne voit jamais les tâches de test.

Il n'y a pas de remise à l'échelle du coût à prévoir. Le planner utilise AdamW
puis garde le candidat de coût minimal, si bien que multiplier un coût par une
constante positive ne change pas le plan obtenu, au terme epsilon d'Adam près.

## 5. L'échelle d'évaluation

| Barreau | Ce qu'on fait | Sur quoi | Nombre d'essais |
|---|---|---|---|
| 1 | courbes de loss et effective rank de la sortie | latents d'entraînement | libre |
| 2 | diagnostic offline avec `cost_diagnostic.py` | jeu de validation | libre |
| 3 | planning à gradient | tâches de développement | libre |
| 4 | planning à gradient, 3 seeds | 50 tâches de test | 2 variantes par arm et par tour |
| 5 | planning iCEM-3000, 3 seeds | 50 tâches de test | les variantes du barreau 4 |

Le diagnostic se mesure à la distance des tâches, soit 15 pas latents sur Push-T
et 4 sur Cube, avec un segment par frame de départ. Sur Push-T il se mesure aussi
à 4 et 8 pas, le planner replanifiant bien avant d'avoir parcouru 15 pas.

Les tâches de développement sont un second jeu de 50 tâches par environnement,
tiré d'épisodes d'entraînement que les têtes n'ont pas vus. Il n'existe pas
encore. Une évaluation sur Cube dure 3 minutes, ce qui rend très facile de
régler une tête sur les tâches de test sans s'en rendre compte. Le second jeu
sert à absorber ces réglages.

Les deux variantes qui montent au barreau 4 sont annoncées dans un commit avant
d'être lancées.

Le tuteur demandait que seules les deux meilleures têtes au diagnostic passent
en planning. Au premier tour ce filtre a écarté GCIVL, qui s'est révélée la
meilleure tête entraînée une fois passée en planning. Le barreau 3 le remplace,
sous réserve de son accord.

## 6. Ce qu'on rapporte

Un tableau par environnement, produit par `aggregate_ch.py` à partir des
fichiers de résultats et jamais rempli à la main.

| Colonne | Définition |
|---|---|
| Signal | la catégorie de la règle 4 |
| Spearman, encodé et prédit | corrélation de rang entre coût et pas restants, calculée par segment puis moyennée |
| Monotonicité, encodé et prédit | part des pas consécutifs où le coût baisse |
| Effective rank | celui de la sortie de la tête sur 10 000 états d'entraînement |
| Success rate GD et iCEM | moyenne ± erreur standard sur les 3 seeds |
| Gagnés et perdus | couples (tâche, seed) que la tête réussit quand le L2 échoue et l'inverse, sur 150 |

Une tête est dite meilleure ou moins bonne que le L2 quand le test des signes sur
les couples gagnés et perdus donne p < 0,05. À titre d'exemple, 30 gagnés pour 37
perdus donnent p = 0,46 et 25 gagnés pour 10 perdus donnent p = 0,017. L'écart
des moyennes ne suffit pas, l'erreur standard de la baseline ayant atteint
5 points sur Push-T dans le second montage.

## 7. Dépôt et résultats

| Chemin | Contenu |
|---|---|
| `checkpoints/cost_heads/<env>/<tête>/seed<s>/` | `head.pt`, `args.json`, `metrics.jsonl` |
| `results/cost_heads/<env>/diag/` | sorties du diagnostic |
| `results/cost_heads/<env>/plan_dev/<planner>_<tête>/seed<s>/` | planning sur les tâches de développement |
| `results/cost_heads/<env>/plan/<planner>_<tête>/seed<s>/` | planning sur les tâches de test |

La branche `main` porte le socle commun. Chaque binôme travaille sur une branche
`arm/<nom>` tirée de ce socle et y revient par une pull request interne. Avant
de créer une pull request il faut vérifier que « base repository » désigne notre
dépôt, GitHub proposant par défaut celui des auteurs de H-JEPA. Aucune pull
request ni issue ne doit partir vers leur dépôt.

Seules les têtes du tableau final sont versionnées, le dépôt pesant déjà 118 Mo
sur `cost-heads`.

## 8. Arms et binômes

| Arm | Signal | Point de départ | Première question | Binôme |
|---|---|---|---|---|
| A, straightening (3a) | aucun | premier au diagnostic, 21 à 29 % en planning, collapse sans SIGReg | pourquoi un espace plus droit le long des trajectoires expertes planifie moins bien | |
| A′, distance temporelle (3b) | temps | excellent sur latents encodés, 6,7 % en planning | la chute vient-elle du passage aux latents prédits | |
| B, value supervisée (3c) | succès | meilleure tête entraînée en planning, 33,3 % | l'entraîner aussi sur des latents prédits comble-t-il l'écart | |
| C, coût configurable (3e) | état du simulateur | sous le L2 au diagnostic, jamais passé en planning | l'information sur l'objet est-elle assez présente dans les latents prédits | |
| T, banc et diagnostic | | diagnostic qui ne prédit pas le planning | quel diagnostic calculé hors des trajectoires expertes le prédit | |

L'arm T ne produit aucune tête. Il tient le socle et les tableaux, puis cherche
un diagnostic fiable, sujet devenu le résultat principal du premier tour.

## 9. Décisions à valider

| Décision | Proposition | Qui tranche |
|---|---|---|
| World model gelé | niveau 1 de `<env>_hjepa_l2` | équipe |
| Remplacer le filtre du diagnostic par des tâches de développement | oui | tuteur |
| Spearman du tableau | par segment puis moyenné | tuteur |
| Ce que le tuteur appelle straightening d'ordre 1 et d'ordre 2 | à lui demander | tuteur |
| Dépôt partagé public ou privé | privé, détaché du fork | équipe |
| Répartition des binômes | colonne de la section 8 | équipe |

## 10. À faire avant de se séparer

1. Fusionner `cost-heads` dans `main` et poser l'étiquette `cadre-v1`.
2. Rendre `job.sh`, `encode_latents.py` et `cost_diagnostic.py` indépendants de
   Push-T, puis refaire la baseline et le latent de niveau 2 sur Cube.
3. Produire les tâches de développement des deux environnements.
4. Ajouter à `aggregate_ch.py` les colonnes « Signal » et « Gagnés et perdus ».
5. Porter depuis `psc/` la tête à query, le linear probe et la variante
   « corde » du straightening.
