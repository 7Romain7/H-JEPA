# psc : têtes de coût sur LeWM à plat, Push-T et Cube

Seconde implémentation du protocole du tuteur, écrite sans connaître celle de la
branche `cost-heads`. La comparaison des deux se trouve dans
[`docs/COMPARAISON_COST_HEADS.md`](../docs/COMPARAISON_COST_HEADS.md).

## Ce qu'on a repris du dépôt

Les checkpoints LeWM publiés (`<env>_lewm`, seeds 42, 43 et 44), les 50 tâches
fixes de chaque environnement, le planner à gradient des configs `<env>_flat` et
la classe `SIGReg` de `h_jepa/loss.py`. Aucun fichier du dépôt n'est modifié.

## Ce qu'on a ajouté

| Fichier | Rôle |
|---|---|
| `diagnostic.py` | encode des trajectoires, déroule le prédicteur, mesure Spearman et monotonicité d'un coût sur latents encodés et prédits ; sert aussi à écrire le cache de latents |
| `heads.py` | les têtes : `Projector` (3a), `QuasiMetric` (3b), `ValueHead` (3c), `QueryHead` (3e) ainsi que l'effective rank |
| `train_3a.py` | straightening, formes « consécutif » et « corde », avec SIGReg |
| `train_3b.py` | objectif de QRL sur une quasimétrique MRN |
| `train_3c.py` | GCIVL avec goals relabellisés |
| `factors.py` | facteurs lus dans l'état simulateur (objet, agent ou bras) |
| `probe.py` | linear probe des facteurs sur latents encodés et prédits |
| `train_3e.py` | tête conditionnée par une query, d'après la description de ConfigWM donnée par le tuteur |
| `eval_head.py` | évaluation de planning avec une tête à la place du L2 |

## Lancer

Depuis `h_jepa/`, avec `$CK` le checkpoint et `$HJEPA_HOME` le dossier des données :

```bash
# cache des latents d'entraînement
python ../psc/diagnostic.py --ckpt $CK --h5 $HJEPA_HOME/pusht_expert_train.h5 \
  --offsets 15 --max-episodes 3000 --save-cache cache.pt --only-cache --out /dev/null
# une tête
python ../psc/train_3a.py --cache cache.pt --out-dir heads --name corde_sig0.01 \
  --straight corde --w-sigreg 0.01
# diagnostic du L2 et des têtes
python ../psc/diagnostic.py --ckpt $CK --h5 $HJEPA_HOME/pusht_expert_val.h5 \
  --offsets 15 8 4 --heads heads/*.pt --out diag.json
# planning
PSC_HEAD=heads/corde_sig0.01.pt python ../psc/eval_head.py --config-name pusht_flat \
  seed=42 policy=$CK output.dir=eval_corde
```

## Résultats

Les fichiers bruts sont dans `results/`, un dossier par étape. Les chemins de
machine y sont remplacés par `$PSC_ROOT`.

| Dossier | Contenu |
|---|---|
| `baseline-pusht`, `baseline-cube`, `baseline-fourroom` | `metrics.yaml` de la baseline L2 |
| `diagnostic` | diagnostic du L2 seul sur Push-T |
| `tete3a` à `tete3e` | diagnostics des têtes, historiques d'entraînement, linear probe |
| `planning` | `metrics.yaml` des têtes passées en planning |

Les tableaux sont dans le document de comparaison.

## Limites

Un seul réglage par tête, sans recherche d'hyperparamètres. QRL et la tête 3e
sont réécrits d'après le mail du tuteur, sans partir des codes d'origine. Les
têtes entraînées ne sont pas versionnées. Le coût d'une tête est remis à
l'échelle du L2 par égalité des médianes sur le premier lot de candidats. Cette
étape est inutile, le planner AdamW étant insensible à une constante positive
devant le coût.
