#!/usr/bin/env python3
"""
Lanceur d'expériences PSC sur les machines des salles info de l'X.

Principe (différent de compute.js) : chaque job est lancé en `nohup` sur une
machine libre, avec ses logs dans ~/psc/runs/<job>/ . Le PC local peut donc se
mettre en veille / perdre le wifi sans tuer l'entraînement. On revient ensuite
avec `status` pour voir ce qui a fini.

Uniquement la stdlib Python + le binaire `ssh` (OpenSSH, présent sous Windows 10+).
Authentification par clé SSH (voir README) — pas de mot de passe dans le code.

Usage :
  python cluster.py probe  [--rooms 31 34]            # qui est allumé, GPU libre ?
  python cluster.py exec   "nvidia-smi" --host ablette
  python cluster.py free   [--min-vram 20000]          # combien de GPU libres maintenant
  python cluster.py run    jobs.txt [--rooms 31 34] [--min-vram 20000]
  python cluster.py status
  python cluster.py kill   <job_id>
Variables d'env : X_USER (login X), X_JUMP (passerelle si hors campus, ex. login@ssh.polytechnique.fr)
"""
import argparse, datetime, json, os, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHINES = json.loads((HERE / "machines.json").read_text())
STATE = HERE / "runs_state.json"          # suivi local des jobs lancés
REMOTE_ROOT = "~/psc"                      # sur le home partagé (à vérifier : cf. README)
REMOTE_CODE = os.environ.get("PSC_REMOTE_CODE", f"{REMOTE_ROOT}/code")  # clone du repo (code_patch : latent spatial)
REMOTE_VENV = f"{REMOTE_ROOT}/venv"
# Variables d'env de chaque job : caches stable-worldmodel sur le home partagé, wandb hors ligne
REMOTE_ENV = (f"export STABLEWM_HOME={REMOTE_ROOT}/swm_home "
              # LOCAL_DATASET_DIR retire le 28/09/2026 : confirme via grep sur le vrai
              # package installe (~/psc/venv/lib/python3.11/site-packages/stable_worldmodel/)
              # que cette variable n'est lue nulle part -> elle ne faisait que creer un
              # second cache dataset orphelin sous {REMOTE_ROOT}/datasets/datasets/, jamais
              # utilise par le vrai pipeline (qui ne lit que STABLEWM_HOME).
              f"WANDB_MODE=${{WANDB_MODE:-offline}} HYDRA_FULL_ERROR=1")
WORKERS = 64                               # connexions SSH simultanées (169 machines sondées en ~5 s)

GPU_BY_ROOM = {30: "RTX A4000 16GB", 31: "RTX A5000 24GB", 32: "RTX 2000 Ada 16GB",
               33: "RTX 4000 Ada 20GB", 34: "RTX 3090 24GB", 35: "RTX 4000 Ada 20GB",
               36: "RTX 4000 Ada 20GB"}

# ------------------------------------------------------------ regle DSI (24/09/2026)
# Mail de la DSI (24/09/2026) : "l'utilisation des ressources de calcul intensif est
# autorisee exclusivement en dehors des plages horaires pedagogiques, c'est a dire la
# nuit et le week-end". En cas de recidive, le compte peut etre bloque -> deux gardes
# independants, cote outil (pas touche a lewm.py) :
#  1) cmd_run() refuse de LANCER de nouveaux jobs pendant la fenetre interdite
#     (sauf --force, pour le jour ou la DSI donne des horaires exacts differents).
#  2) chaque job est lance sous `timeout` avec une echeance ABSOLUE (--deadline) :
#     meme lance juste avant la fenetre interdite, il s'auto-tue (SIGTERM) avant.
# Horaires CONFIRMES par la DSI (reponse mail du 24/09/2026, page salles info) :
# "arrangez-vous pour que vos calculs n'aient lieu qu'en heures non ouvrees
# (18h-8h)" en semaine ; le week-end est integralement en heures non ouvrees.
# Piege signale par la DSI, distinct de cette fenetre : "chaque jour, du lundi
# au vendredi, toutes les taches utilisateur sont supprimees si elles s'executent
# sans la presence physique de l'utilisateur ou depuis trop longtemps" -> nos
# jobs nohup sont exactement ce cas de figure. Le --deadline (SIGTERM 5 min avant
# la fin de la fenetre, cf. launch()) est notre protection principale : un job
# qui s'arrete proprement avant 8h ne devrait jamais se faire rattraper par ce
# nettoyage. Si un jour on veut un run qui deborde largement (plusieurs jours,
# ou qui doit rester up pendant des heures ouvrees), demander l'exception
# motivee mentionnee par la DSI plutot que de compter sur le --deadline seul.
PEDAGOGICAL_WEEKDAYS = {0, 1, 2, 3, 4}
PEDAGOGICAL_START_HOUR = 8
PEDAGOGICAL_END_HOUR = 18


def is_pedagogical_time(dt=None):
    dt = dt or datetime.datetime.now()
    if dt.weekday() not in PEDAGOGICAL_WEEKDAYS:
        return False
    return PEDAGOGICAL_START_HOUR <= dt.hour < PEDAGOGICAL_END_HOUR


# ---------------------------------------------------------------- ssh
def ssh(host, cmd, timeout=20):
    """Exécute `cmd` sur `host`. Retourne (code, stdout, stderr)."""
    user = os.environ.get("X_USER")
    target = f"{user}@{host}" if user else host
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=accept-new"]
    if os.environ.get("X_JUMP"):
        args += ["-J", os.environ["X_JUMP"]]
    args += [target, cmd]
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )  # utf-8 explicite : la sortie SSH (ex. barres tqdm) peut contenir
           # des octets non decodables en cp1252, le codec par defaut de
           # subprocess sous Windows, ce qui plantait le thread lecteur.
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 255, "", "timeout"


def select(rooms=None, hosts=None):
    ms = MACHINES
    if rooms:
        ms = [m for m in ms if m["room"] in rooms]
    if hosts:
        hs = {h.lower() for h in hosts}
        ms = [m for m in ms if m["name"].lower() in hs or m["url"] in hs]
    return ms


# ---------------------------------------------------------------- probe
PROBE_CMD = (
    "echo USERS=$(who | awk '{print $1}' | sort -u | tr '\\n' ',');"
    "echo LOAD=$(cut -d' ' -f1 /proc/loadavg);"
    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^/GPU=/'"
)


def probe_one(m):
    code, out, err = ssh(m["url"], PROBE_CMD)
    r = {"name": m["name"], "url": m["url"], "room": m["room"], "up": code == 0}
    if code != 0:
        r["err"] = (err or out)[:60]
        return r
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "USERS":
            r["users"] = [u for u in v.split(",") if u]
        elif k == "LOAD":
            r["load"] = float(v)
        elif k == "GPU":
            name, used, total, util = [x.strip() for x in v.split(",")]
            r.update(gpu=name, mem_used=int(used), mem_total=int(total), util=int(util))
    return r


def is_free(r, me, max_mem_used=1000, max_util=10, allow_users=False):
    if not r.get("up") or "gpu" not in r:
        return False
    others = [u for u in r.get("users", []) if u != me]
    if others and not allow_users:
        return False          # quelqu'un est connecté (souvent physiquement en salle) : on ne squatte pas
    return r["mem_used"] <= max_mem_used and r["util"] <= max_util


def probe(machines, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(probe_one, machines))


def cmd_probe(a):
    res = probe(select(a.rooms, a.hosts))
    me = os.environ.get("X_USER", "")
    print(f"{'machine':<12}{'salle':>6}  {'état':<6}{'GPU':<28}{'mem':>12}{'util':>6}  users")
    for r in sorted(res, key=lambda r: (r["room"], r["name"])):
        if not r["up"]:
            print(f"{r['name']:<12}{r['room']:>6}  {'DOWN':<6}{r.get('err','')}")
            continue
        free = "LIBRE" if is_free(r, me) else "occupé"
        print(f"{r['name']:<12}{r['room']:>6}  {free:<6}{r.get('gpu','?'):<28}"
              f"{r.get('mem_used','?'):>5}/{r.get('mem_total','?'):<6}{r.get('util','?'):>5}%  "
              f"{','.join(r.get('users', []))}")
    up = [r for r in res if r["up"]]
    print(f"\n{len(up)}/{len(res)} joignables, {sum(is_free(r, me) for r in res)} GPU libres")


def cmd_exec(a):
    ms = select(a.rooms, a.hosts)
    with ThreadPoolExecutor(WORKERS) as ex:
        for m, (code, out, err) in zip(ms, ex.map(lambda m: ssh(m["url"], a.command, a.timeout), ms)):
            print(f"--- {m['name']} (exit {code})\n{out}{(chr(10) + err) if err else ''}")


# ---------------------------------------------------------------- jobs
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


PINS = {}   # job_id -> nom de machine imposée (syntaxe `job_id@machine | commande`)


def parse_jobs(path):
    """Une ligne = `job_id | commande`, ou juste `commande` (id auto). # = commentaire.
    `job_id@machine | commande` : le job ne peut tourner QUE sur cette machine
    (ex. évaluer un checkpoint resté sur son disque local /tmp)."""
    jobs = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            jid, cmd = [x.strip() for x in line.split("|", 1)]
            if "@" in jid:
                jid, host = jid.split("@", 1)
                PINS[jid] = host.strip().lower()
        else:
            jid, cmd = f"job{i:03d}", line
        jobs.append((jid, cmd))
    return jobs


def launch(m, jid, cmd, deadline_ts=None):
    d = f"{REMOTE_ROOT}/runs/{jid}"
    timed_cmd = cmd
    if deadline_ts is not None:
        # marge de securite de 5 min avant l'echeance reelle (temps pour que
        # SIGTERM soit intercepte proprement -> Lightning sauve un checkpoint -
        # avant la fin de la fenetre autorisee).
        remaining = int(deadline_ts - time.time()) - 300
        remaining = max(remaining, 60)
        timed_cmd = f"timeout --signal=TERM {remaining} {cmd}"
    inner = (f"cd {REMOTE_CODE} && git rev-parse HEAD > {d}/commit.txt 2>/dev/null; "
             f'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate {REMOTE_VENV} && '
             f"export PSC_RUN_DIR={d} && {REMOTE_ENV} && {timed_cmd}")
    # exit_code écrit à la fin -> permet de savoir si le job a fini, même sans connexion ouverte
    wrapper = (f"mkdir -p {d} && cd {d} && echo {shlex.quote(cmd)} > cmd.txt && hostname > host.txt || exit 1; "
               f"nohup nice -n 5 bash -c {shlex.quote(inner + '; echo $? > ' + d + '/exit_code')} "
               f"> {d}/log.txt 2>&1 < /dev/null & echo $! > {d}/pid")
    code, out, err = ssh(m["url"], wrapper)
    return code == 0, err


def remote_status(host, jid):
    d = f"{REMOTE_ROOT}/runs/{jid}"
    code, out, _ = ssh(host, f"cat {d}/exit_code 2>/dev/null || "
                             f"(kill -0 $(cat {d}/pid) 2>/dev/null && echo RUNNING || echo DEAD)")
    return out if code == 0 else "UNREACHABLE"


def refresh(state):
    """Met à jour en parallèle le statut de tous les jobs RUNNING."""
    running = [j for j, s in state.items() if s["status"] == "RUNNING"]
    ok = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for jid, st in zip(running, ex.map(lambda j: remote_status(state[j]["url"], j), running)):
            state[jid]["status"] = st
            if st == "0":
                ok += 1
            elif st != "RUNNING":
                print(f"[{state[jid]['name']}] {jid} ÉCHEC -> {st}  (log : ~/psc/runs/{jid}/log.txt)")
    if ok:
        print(f"{ok} jobs terminés avec succès")


def free_machines(a, state=None):
    me = os.environ.get("X_USER", "")
    busy = {s["url"] for s in (state or {}).values() if s["status"] == "RUNNING"}
    free = [r for r in probe(select(a.rooms, a.hosts))
            if is_free(r, me, allow_users=a.allow_users)
            and r["mem_total"] >= a.min_vram and r["url"] not in busy]
    free.sort(key=lambda r: -r["mem_total"])      # plus grosses cartes d'abord
    return free


def cmd_run(a):
    if is_pedagogical_time() and not a.force:
        print("REFUS : on est dans la fenêtre pédagogique (semaine, "
              f"{PEDAGOGICAL_START_HOUR}h-{PEDAGOGICAL_END_HOUR}h) -> voir le mail de la DSI "
              "du 24/09/2026 (ressources autorisées seulement la nuit et le week-end, "
              "récidive = blocage de compte). Relance après "
              f"{PEDAGOGICAL_END_HOUR}h, un week-end, ou avec --force si tu es sûr "
              "que c'est autorisé maintenant.")
        return
    try:
        deadline = datetime.datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
    except ValueError:
        print("--deadline invalide, format attendu : 'YYYY-MM-DD HH:MM' (heure locale), "
              "ex. --deadline \"2026-09-27 07:00\"")
        return
    deadline_ts = deadline.timestamp()
    launch_until_ts = None
    if a.launch_until:
        launch_until_ts = datetime.datetime.strptime(a.launch_until, "%Y-%m-%d %H:%M").timestamp()
    if deadline_ts <= time.time():
        print("--deadline est déjà passée, rien à lancer.")
        return
    print(f"Échéance absolue : {a.deadline} -> chaque job s'auto-arrête (SIGTERM, marge 5 min) "
          "avant cette heure, même sur une machine lente.")

    state = load_state()
    todo = [(j, c) for j, c in parse_jobs(a.jobs) if j not in state or
            (a.retry_failed and state[j]["status"] not in ("RUNNING", "0"))]
    print(f"{len(todo)} jobs à lancer, sur toutes les machines libres"
          + (f" des salles {a.rooms}" if a.rooms else " (salles 30-36)"))
    while todo:
        if time.time() >= deadline_ts:
            print(f"Échéance {a.deadline} atteinte, arrêt du lancement -> "
                  f"{len(todo)} jobs restés en attente ne seront pas lancés.")
            break
        if is_pedagogical_time() and not a.force:
            print("Fenêtre pédagogique atteinte pendant le lancement -> on s'arrête là, "
                  f"{len(todo)} jobs restés en attente.")
            break
        if launch_until_ts is not None and time.time() >= launch_until_ts:
            print(f"--launch-until {a.launch_until} atteint -> {len(todo)} jobs restés "
                  "en attente ne seront pas lancés.")
            break
        refresh(state)
        free = free_machines(a, state)
        # 1 job par machine libre, tout en parallèle. Les jobs épinglés passent
        # d'abord, sur leur machine ; les autres prennent les machines restantes.
        by_name = {r["name"].lower(): r for r in free}
        batch, used, waiting = [], set(), []
        for jid, cmd in todo:
            h = PINS.get(jid)
            if h is None:
                continue
            r = by_name.get(h)
            if r is not None and r["url"] not in used:
                batch.append((r, (jid, cmd)))
                used.add(r["url"])
            else:
                waiting.append((jid, cmd))
        unpinned = [j for j in todo if j[0] not in PINS]
        rest = [r for r in free if r["url"] not in used]
        batch += list(zip(rest, unpinned))
        todo = waiting + unpinned[len(rest):]
        with ThreadPoolExecutor(WORKERS) as ex:
            results = list(ex.map(lambda rj: launch(rj[0], *rj[1], deadline_ts=deadline_ts), batch))
        for (r, (jid, cmd)), (ok, err) in zip(batch, results):
            if ok:
                state[jid] = {"name": r["name"], "url": r["url"], "room": r["room"],
                              "cmd": cmd, "status": "RUNNING", "started": time.strftime("%F %T")}
                print(f"[{r['name']} / salle {r['room']}] lancé {jid}")
            else:
                print(f"[{r['name']}] échec lancement {jid}: {err}")
                todo.append((jid, cmd))
        save_state(state)
        n_run = sum(s["status"] == "RUNNING" for s in state.values())
        print(f"-> {n_run} jobs en cours, {len(todo)} en attente")
        if todo:
            time.sleep(a.poll)
    print("Tous les jobs sont lancés. `python cluster.py status` pour suivre.")


def cmd_free(a):
    free = free_machines(a)
    by_room = {}
    for r in free:
        by_room[r["room"]] = by_room.get(r["room"], 0) + 1
    print(len(free), "machines libres |", ", ".join(f"salle {k}: {v}" for k, v in sorted(by_room.items())))


def cmd_status(a):
    state = load_state()
    refresh(state)
    for jid, s in sorted(state.items()):
        tag = {"0": "OK", "RUNNING": "en cours"}.get(s["status"], s["status"])
        print(f"{jid:<28}{s['name']:<12}{tag:<12}{s['started']}  {s['cmd'][:70]}")
    counts = {}
    for s in state.values():
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print("\n" + ", ".join(f"{ {'0':'OK'}.get(k, k)}: {v}" for k, v in counts.items()))
    save_state(state)


def cmd_kill(a):
    s = load_state()[a.job]
    d = f"{REMOTE_ROOT}/runs/{a.job}"
    print(ssh(s["url"], f"pkill -TERM -P $(cat {d}/pid); kill $(cat {d}/pid); echo killed")[1])


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    def common(q):
        q.add_argument("--rooms", type=int, nargs="*")
        q.add_argument("--hosts", nargs="*")
    q = sp.add_parser("probe"); common(q); q.set_defaults(f=cmd_probe)
    q = sp.add_parser("exec"); common(q); q.add_argument("command")
    q.add_argument("--timeout", type=int, default=60); q.set_defaults(f=cmd_exec)
    def sched(q):
        q.add_argument("--min-vram", type=int, default=0, help="VRAM min en Mo (ex. 20000 pour exclure les 16 Go)")
        q.add_argument("--allow-users", action="store_true", help="utiliser aussi les machines où qqn est connecté")
    q = sp.add_parser("free"); common(q); sched(q); q.set_defaults(f=cmd_free)
    q = sp.add_parser("run"); common(q); sched(q); q.add_argument("jobs")
    q.add_argument("--poll", type=int, default=120)
    q.add_argument("--retry-failed", action="store_true")
    q.add_argument("--deadline", required=True,
                    help="Échéance absolue 'YYYY-MM-DD HH:MM' (heure locale) : chaque job "
                         "s'auto-arrête avant cette heure (règle DSI nuit/week-end, 24/09/2026).")
    q.add_argument("--launch-until", default=None,
                    help="'YYYY-MM-DD HH:MM' : ne plus lancer de nouveaux jobs après cette heure "
                         "(un job lancé trop tard ne finirait pas avant --deadline).")
    q.add_argument("--force", action="store_true",
                    help="Ignore le garde-fou horaire pédagogique (à utiliser seulement si tu es "
                         "sûr que c'est autorisé, ex. la DSI a donné des horaires différents).")
    q.set_defaults(f=cmd_run)
    q = sp.add_parser("status"); q.set_defaults(f=cmd_status)
    q = sp.add_parser("kill"); q.add_argument("job"); q.set_defaults(f=cmd_kill)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
