# x-cluster : lancer les expériences du PSC sur les GPU des salles info de l'X

> **Pour un·e Claude qui reprend ce dossier** : lis tout ce fichier avant d'agir. La partie « État actuel » en bas dit ce qui a été testé et ce qui ne l'a pas été.

---

## 0. Règle DSI (24/09/2026) : nuit et week-end uniquement

**Incident du 24/09/2026** : la DSI a coupé de force `test0`/`test1` (lancés le 23/09, laissés tourner par erreur pendant les heures de cours) et a envoyé un avertissement : *"l'utilisation des ressources de calcul intensif est autorisée exclusivement en dehors des plages horaires pédagogiques, c'est-à-dire la nuit et le week-end [...] En cas de récidive, des mesures restrictives pourront être appliquées, incluant un blocage de votre compte."* La coupure de force s'est aussi traduite, pendant quelques heures, par un accès refusé au home (`Permission denied` au login, y compris par mot de passe) sur toutes les machines — pas un bug de notre côté, l'infra elle-même était temporairement bloquée.

**Horaires confirmés par la DSI (réponse du 24/09/2026)** : *« arrangez-vous pour que vos calculs n'aient lieu qu'en heures non ouvrées (18h-8h) »* en semaine ; le week-end est intégralement en heures non ouvrées. Piège signalé en plus par la DSI, distinct de cette fenêtre : *« chaque jour, du lundi au vendredi, toutes les tâches utilisateur sont supprimées si elles s'exécutent sans la présence physique de l'utilisateur ou depuis trop longtemps »* — nos jobs `nohup` sont exactement ce cas de figure. Le `--deadline` ci-dessous (SIGTERM propre avant la fin de la fenêtre) est notre protection principale contre ça : un job terminé avant 8h ne devrait jamais être rattrapé par ce nettoyage. Pour un run qui devrait déborder largement (plusieurs jours, ou rester up en heures ouvrées), demander l'exception motivée mentionnée par la DSI plutôt que de compter sur `--deadline` seul.

**Conséquence directe, deux garde-fous ajoutés dans `cluster.py` (indépendants de `lewm.py`, donc rien touché côté code d'entraînement) :**
1. `cluster.py run` refuse de lancer quoi que ce soit si l'heure actuelle tombe dans la fenêtre pédagogique (semaine, lundi-vendredi 8h-18h, valeurs confirmées par la DSI). Contournable avec `--force` si besoin (exception accordée, etc.).
2. `run` prend un argument **obligatoire** `--deadline "YYYY-MM-DD HH:MM"` (heure locale) : chaque job est lancé sous `timeout --signal=TERM`, calculé pour s'arrêter proprement 5 minutes avant cette échéance absolue, quelle que soit la vitesse de la machine sur laquelle il tombe. Ça remplace le besoin de calibrer un nombre de pas précis : on choisit directement une heure de fin, et Lightning reçoit un SIGTERM propre (pas un SIGKILL brutal) avant.

**En pratique** : la fenêtre autorisée est plus large qu'on ne le pensait — 18h-8h **tous les soirs** (pas seulement le week-end), plus le week-end intégral. Un run de ~8h tient largement dans une seule nuit de semaine (ex. lancer un soir à 19h, `--deadline` le lendemain avant 8h), pas besoin d'attendre le week-end.

## 1. Contexte

- **PSC** (projet scientifique de 2A à l'X) sur les JEPA / world models. On entraîne des modèles avec **`stable-worldmodel`** (package pip `stable-worldmodel`, repo GitHub `galilai-group/stable-worldmodel`, écosystème LeWM/LeJEPA). L'expérience principale : ajouter une régularisation du jacobien local de la prédiction par rapport à l'action (**L_jac**) dans `lejepa_forward`, puis comparer baseline et plusieurs valeurs de λ_jac sur plusieurs seeds.
- **Calcul** : les 169 PC des salles info 30 à 36 de l'X, un GPU chacun. Accès SSH **autorisé par la DSI**.
- **Repo du projet** : `https://github.com/Reddimolk/psc-jepa-straightening` (public ou privé, à confirmer). Copie locale : `C:\Users\User\Documents\PSC\psc-jepa-straightening`.
- **Ce dossier** : `C:\Users\User\Documents\PSC\x-cluster`.

### Ce que Claude (Cowork) peut et ne peut pas faire
- Claude **ne peut pas** se connecter aux machines de l'X : la VM Cowork sur le PC n'a pas de réseau sortant, et le cloud bloque SSH. **C'est Joachim qui tape les commandes `ssh` / `python cluster.py ...`** dans un terminal Windows (PowerShell), puis colle les sorties à Claude.
- Claude peut lire et modifier les fichiers de ce dossier et du repo, écrire du code et interpréter les sorties.

---

## 2. Les fichiers

| Fichier | Rôle |
|---|---|
| `cluster.py` | Le lanceur (tourne **sur le PC**, stdlib Python + `ssh`). Sous-commandes : `probe`, `free`, `exec`, `run`, `status`, `kill`. |
| `remote_setup.sh` | Installation sur le cluster (à lancer une fois, puis à chaque changement de dépendances) : clone le repo, crée le venv, installe `requirements.txt`, vérifie CUDA. |
| `make_jobs.py` | Génère `jobs.txt`, la liste des entraînements (grille λ_jac × seeds). |
| `jobs.txt` | Généré. Une ligne = `id_du_job \| commande`. |
| `runs_state.json` | Généré. Suivi local des jobs lancés (machine, statut). **Ne pas supprimer pendant une campagne.** |
| `machines.json` | Les 169 machines (ip, url, nom, salle). |

### Comment ça marche
1. `run` sonde les 169 machines en parallèle via SSH : allumée ? quelqu'un de connecté ? GPU occupé ?
2. Sur **chaque machine libre, il lance un job** (1 job = 1 entraînement = 1 GPU), tous en parallèle. Le job tourne en `nohup` côté X : **le PC peut se mettre en veille ou être éteint**, les entraînements continuent.
3. S'il reste des jobs en attente, il re-sonde toutes les 120 s et remplit les machines qui se libèrent.
4. Plus tard, `status` dit quels jobs ont fini (OK / code d'erreur / DEAD si la machine a été redémarrée).

Un job n'utilise qu'une machine. On ne répartit pas un même entraînement sur plusieurs machines : c'est inutile pour une grille d'expériences, et le réseau des salles serait trop lent pour ça.

### Organisation sur le cluster (home Linux de l'X)
```
~/psc/
├── code/        clone du repo PSC (git pull pour mettre à jour)
├── venv/        environnement Python (torch, stable-worldmodel…)
├── datasets/    datasets  (variable LOCAL_DATASET_DIR)
├── swm_home/    cache stable-worldmodel, dont swm_home/checkpoints/  (variable STABLEWM_HOME)
└── runs/<job>/  un dossier par job :
       log.txt       sortie complète (stdout+stderr)
       cmd.txt       commande lancée
       commit.txt    commit git du code utilisé
       host.txt      machine
       pid           process
       exit_code     n'apparaît qu'à la fin (0 = succès)
```
Chaque job est lancé dans `~/psc/code`, venv activé, avec ces variables : `PSC_RUN_DIR=~/psc/runs/<job>`, `STABLEWM_HOME=~/psc/swm_home`, `LOCAL_DATASET_DIR=~/psc/datasets`, `WANDB_MODE=offline` (le script d'entraînement utilise un logger wandb), `HYDRA_FULL_ERROR=1`, et `nice -n 5`.

---

## 3. Mise en route — à faire dans l'ordre

Toutes les commandes « sur le PC » se tapent dans **PowerShell**, dans le dossier `x-cluster` (`cd C:\Users\User\Documents\PSC\x-cluster`). Remplace `TON_LOGIN` par ton login X.

### Étape A — Préparer le repo PSC (sur le PC)

Le code qui entraîne le modèle doit être **dans ton repo**, parce que c'est lui qui sera cloné sur les machines de l'X.

**A1. Copier le script d'entraînement de stable-worldmodel dans ton repo.**
`pip install stable-worldmodel` n'installe que la bibliothèque. Le script d'entraînement `scripts/train/lewm.py` (qui contient `lejepa_forward`, la fonction où on ajoutera L_jac) et sa config Hydra `scripts/train/config/` sont seulement dans le repo GitHub. Il faut donc les copier chez toi :
```powershell
cd C:\Users\User\Documents\PSC
git clone https://github.com/galilai-group/stable-worldmodel.git swm-upstream
mkdir psc-jepa-straightening\scripts\train
Copy-Item swm-upstream\scripts\train\lewm.py   psc-jepa-straightening\scripts\train\
Copy-Item swm-upstream\scripts\train\config    psc-jepa-straightening\scripts\train\ -Recurse
```
(`lewm.py` a `@hydra.main(config_path='./config', config_name='lewm')` : le dossier `config` doit rester à côté de lui.)
Regarder aussi s'il y a d'autres fichiers dans `swm-upstream\scripts\train\` que `lewm.py` importe, et les copier aussi.

**A2. Commit + push :**
```powershell
cd psc-jepa-straightening
git add scripts
git commit -m "Ajout script d'entraînement LeWM (copié de stable-worldmodel)"
git push
```
C'est tout : les machines de l'X récupèrent le code par `git clone` / `git pull` depuis GitHub. **Tout ce qui n'est pas poussé sur GitHub n'existe pas pour le cluster.**

**A3. Le repo est-il public ou privé ?** (Sur GitHub : page du repo, badge « Public » ou « Private ».)
- **Public** : rien à faire.
- **Privé** : les machines de l'X doivent s'authentifier auprès de GitHub. Le plus simple : se connecter à une machine de l'X (`ssh TON_LOGIN@ablette.polytechnique.fr`), y taper `ssh-keygen -t ed25519` (Entrée à chaque question), puis `cat ~/.ssh/id_ed25519.pub`, et coller cette ligne dans GitHub → Settings → SSH and GPG keys → New SSH key. Ensuite, dans `remote_setup.sh`, mettre `REPO_URL` en version SSH : `git@github.com:Reddimolk/psc-jepa-straightening.git`.

### Étape B — Clé SSH du PC vers l'X (pour ne plus taper de mot de passe)
```powershell
ssh-keygen -t ed25519            # Entrée à chaque question (si une clé existe déjà, répondre n pour ne pas l'écraser)
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh TON_LOGIN@ablette.polytechnique.fr "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
setx X_USER TON_LOGIN            # permanent ; fermer et rouvrir PowerShell ensuite
```
Test : `ssh TON_LOGIN@anchois.polytechnique.fr hostname` doit répondre **sans demander de mot de passe**. Si ça marche aussi sur une autre machine que celle où la clé a été copiée, le home est partagé (NFS) : une installation suffira pour toutes les machines.

**Hors campus** : les machines ne sont joignables que depuis le réseau de l'X. De l'extérieur, il faut le VPN ou la passerelle SSH de l'école (nom à demander à la DSI), puis `setx X_JUMP TON_LOGIN@<passerelle>`.

### Étape C — Vérifications
```powershell
python cluster.py probe --rooms 34                                   # état des 25 machines de la salle 34
python cluster.py exec "quota -s; df -h ~; python3 --version" --hosts ablette
python cluster.py free                                               # nb de GPU libres, toutes salles
```
À noter et à donner à Claude :
- **Quota du home** : le venv (torch + CUDA) fait ~5-6 Go, plus les datasets et les checkpoints. Si le quota est trop petit, demander de l'espace à la DSI.
- **Version de Python** sur les machines : stable-worldmodel demande un Python récent. Si `python3` est trop vieux, voir s'il existe `python3.11` / `python3.12`, ou un `module load`.

### Étape D — Installation sur le cluster
```powershell
Get-Content remote_setup.sh -Raw | ssh TON_LOGIN@ablette.polytechnique.fr "tr -d '\r' | bash -s"
```
(En PowerShell, `< fichier` ne marche pas : d'où le `Get-Content | ssh`. Le `tr -d '\r'` enlève les fins de ligne Windows si le fichier a été édité sous Windows.)

Le script clone ton repo dans `~/psc/code`, crée `~/psc/venv` et y installe `requirements.txt`, qui contient `stable-worldmodel[all]`. Ton repo n'est pas un « package Python » (pas de `pyproject.toml`) : c'est normal, et ça ne demande rien de ta part. Il affiche à la fin `cuda dispo: True` et `stable_worldmodel OK` si tout va bien.
**À relancer à chaque fois que `requirements.txt` change** (il fait aussi un `git pull`).

### Étape E — Un seul job de test avant la grande campagne
Écrire un `test.txt` d'une ligne, avec un entraînement très court (nombre d'epochs/steps réduit, override Hydra à trouver dans `scripts/train/config/lewm.yaml`) :
```
test0 | python scripts/train/lewm.py <overrides pour un run court>
```
```powershell
python cluster.py run test.txt --rooms 34
python cluster.py status
python cluster.py exec "tail -50 ~/psc/runs/test0/log.txt" --hosts <machine affichée par status>
```
Si ça plante, donner le log à Claude.

### Étape F — La campagne complète, sur toutes les machines libres
D'abord, dans `make_jobs.py`, régler `TRAIN`, `ENV` et les noms des overrides Hydra (`seed=`, `+loss.jac.weight=`) pour qu'ils correspondent au vrai `lewm.py` et à sa config. Pour l'instant ce sont des suppositions.
```powershell
python cluster.py free              # combien de GPU libres
python make_jobs.py --auto          # ajoute des seeds jusqu'à ≥ 1 job par GPU libre
python cluster.py run jobs.txt --deadline "2026-09-25 07:30"  # voir §0 : --deadline obligatoire (ex. lancé un soir de semaine ~19h)
python cluster.py status            # à n'importe quel moment, même le lendemain
```
`--deadline` est **obligatoire** depuis le 24/09/2026 (règle DSI, §0) : chaque job s'auto-arrête proprement (SIGTERM) 5 min avant cette heure, même sur la machine la plus lente. `run` refuse aussi de lancer quoi que ce soit si l'heure actuelle tombe dans la fenêtre pédagogique confirmée (semaine 8h-18h) — utiliser `--force` seulement si une exception a été accordée par la DSI.

Options de `run` : `--rooms 31 34` (seulement ces salles), `--min-vram 20000` (exclut les cartes 16 Go des salles 30 et 32 si le modèle n'y tient pas), `--retry-failed` (relance les jobs en échec ou DEAD), `--poll 60`, `--allow-users` (déconseillé).
Si `run` est interrompu (Ctrl+C, PC éteint), relancer la même commande : les jobs déjà lancés sont sautés.

---

## 4. Au quotidien : modifier le code et relancer
1. Modifier le code sur le PC (dans `psc-jepa-straightening`), puis `git commit` et `git push`.
2. Mettre à jour le cluster : `python cluster.py exec "cd ~/psc/code && git pull" --hosts ablette` (une machine suffit si le home est partagé).
3. Si `requirements.txt` a changé, relancer l'étape D.
4. `python make_jobs.py ...`, puis `python cluster.py run jobs.txt`.

⚠️ Avec le home partagé, un `git pull` pendant une campagne change le code des jobs lancés **ensuite**. Chaque job enregistre son commit dans `~/psc/runs/<job>/commit.txt` pour garder la trace. Pour une nouvelle campagne, utiliser de nouveaux ids de job (sinon `run` les considère déjà faits) ou supprimer `runs_state.json`.

## 5. Récupérer les résultats
Les logs sont dans `~/psc/runs/<job>/log.txt`, les checkpoints dans `~/psc/swm_home/checkpoints/`. Pour rapatrier sur le PC :
```powershell
scp -r TON_LOGIN@ablette.polytechnique.fr:~/psc/runs .\resultats\
```
(Pour les courbes wandb hors ligne : `wandb sync` sur les dossiers `wandb/offline-run-*` si on veut les voir sur wandb.ai.)

## 6. Règles et pièges
- **Bon voisinage** (codé dans `cluster.py`) : jamais de job sur une machine où quelqu'un est connecté (souvent un élève devant l'écran) ou dont le GPU est déjà utilisé (> 1 Go de VRAM ou > 10 % d'utilisation). Un seul de nos jobs par machine, en `nice`.
- **Machines redémarrées** (TP, maintenance, quelqu'un qui éteint) : le job meurt et `status` affiche `DEAD`. Sauvegarder des checkpoints pour pouvoir reprendre, puis `run ... --retry-failed`.
- **À grande échelle (100+ jobs)** : pré-télécharger les datasets **une fois** (TODO dans `remote_setup.sh`), sinon chaque job les télécharge en même temps. Garder des checkpoints petits et peu fréquents, car tout le monde écrit dans le même home et le même quota.
- **GPU différents selon les salles** : les temps d'entraînement ne sont pas comparables entre salles, mais les métriques finales le sont.
- Jamais de mot de passe dans le code (l'ancien `machines.js` en utilisait un via une variable d'environnement) : clés SSH uniquement.

| Salle | GPU | VRAM |
|---|---|---|
| 30 (19 PC) | RTX A4000 | 16 Go |
| 31 (25 PC) | RTX A5000 | 24 Go |
| 32 (25 PC) | RTX 2000 Ada | 16 Go |
| 33, 35, 36 (25 PC chacune) | RTX 4000 Ada | 20 Go |
| 34 (25 PC) | RTX 3090 | 24 Go |

---

## 7bis. État actuel (24/09/2026)
**Fait le 23/09/2026** : L_jac implémenté et smoke-testé avec succès sur le cluster (étape E validée, voir le doc projet `plan-action-couplage-jacobien-action.md`). `make_jobs.py` corrigé (clé `env=` hallucinée supprimée, `+loss.jac.weight=` → `loss.jac.weight=` sans `+`, ces deux clés existant déjà dans `lewm.yaml`).

**Incident DSI du 24/09/2026** : voir §0. `cluster.py` a reçu les deux garde-fous horaires décrits ci-dessus (`--deadline` obligatoire + refus de lancer en heures pédagogiques). Pas encore testé en conditions réelles (l'accès au home était encore coupé au moment du patch) — à vérifier dès que l'accès revient, avant de lancer la vraie campagne.

**Prochaine étape** : lancer la grille complète (étape F) le week-end du 26-27/09/2026 (ou une nuit de semaine), une fois l'accès au home confirmé restauré.

## 7. État actuel (22/09/2026)
**Fait**
- `cluster.py`, `make_jobs.py`, `remote_setup.sh` écrits et **testés avec un faux `ssh`** qui simule 169 machines : 135 jobs lancés en ~1 s, machines occupées sautées, les jobs en attente partent quand une machine se libère, `status` et `--min-vram` fonctionnent.
- `REPO_URL` pointe vers `Reddimolk/psc-jepa-straightening` (contient pour l'instant `.gitignore`, `README.md`, `requirements.txt`).

**Jamais testé sur les vraies machines** — premières choses à vérifier :
- [ ] Étape B : SSH sans mot de passe, home partagé entre machines ?
- [ ] Étape C : quota, version de Python, format réel de la sortie de `nvidia-smi` / `who`
- [ ] Étape A : `lewm.py` + `config/` copiés dans le repo, et poussés
- [ ] Étape D : installation OK (`cuda dispo: True`). `lewm.py` importe aussi `stable_pretraining`, `lightning` et `wandb` : vérifier que `stable-worldmodel[all]` les installe bien
- [ ] Vraies commandes et overrides Hydra dans `make_jobs.py` (nom de l'env, epochs, seed, poids de L_jac)
- [ ] Commande de pré-téléchargement des datasets dans `remote_setup.sh`
- [ ] Implémenter L_jac dans `lejepa_forward` (sketch dans le doc projet `plan-action-couplage-jacobien-action.md`), avec le poids lu dans la config (`cfg.loss.jac.weight`)

**Infos vérifiées dans le code de stable-worldmodel (lecture GitHub, 22/09/2026)**
- `scripts/train/lewm.py` : `@hydra.main(config_path='./config', config_name='lewm')`, logger `WandbLogger`, datasets via `swm.data.load_dataset()` avec cache `LOCAL_DATASET_DIR`, checkpoints dans `get_cache_dir(sub_folder='checkpoints')/<run_id>`, qui dépend de la variable d'env `STABLEWM_HOME`.
