#!/usr/bin/env python3
"""
Instantané de l'occupation des 169 machines des salles info (lecture seule).

  python dashboard/snapshot.py            # -> dashboard/snapshot.json

Pour chaque machine : utilisateurs connectés (session physique ou SSH), charge CPU,
RAM, GPU (VRAM, util, température, puissance), processus GPU avec leur propriétaire,
et consommation CPU/RAM agrégée par utilisateur (mesurée par `top` sur ~0,7 s).
Les jobs PSC en cours (runs_state.json) sont rattachés à leur machine.
"""
import datetime, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cluster import MACHINES, GPU_BY_ROOM, STATE, WORKERS  # noqa: E402

SYSTEM_USERS = {"root", "gdm", "nscd", "nobody", "polkitd", "colord", "rtkit", "avahi",
                "messagebus", "systemd-resolve", "systemd-timesync", "systemd-network",
                "syslog", "nslcd", "chrony", "_chrony", "sssd", "daemon", "lp", "kernoops"}
SNAPSHOT = HERE / "snapshot.json"
REMOTE = (HERE / "probe_remote.sh").read_text().replace("\r", "")


def ssh_script(host, timeout=30):
    user = os.environ.get("X_USER")
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=accept-new"]
    if os.environ.get("X_JUMP"):
        args += ["-J", os.environ["X_JUMP"]]
    args += [f"{user}@{host}" if user else host, "bash -s"]
    try:
        # bytes : en mode texte, Windows convertirait les fins de ligne du script en CRLF
        p = subprocess.run(args, input=REMOTE.encode(), capture_output=True, timeout=timeout)
        dec = lambda b: b.decode("utf-8", "replace")
        return p.returncode, dec(p.stdout), dec(p.stderr)
    except subprocess.TimeoutExpired:
        return 255, "", "timeout"


def num(x, f=float):
    try:
        return f(x)
    except ValueError:
        return None


def probe_one(m):
    code, out, err = ssh_script(m["url"])
    r = {"name": m["name"], "url": m["url"], "room": m["room"], "up": code == 0,
         "gpu_model": GPU_BY_ROOM.get(m["room"])}
    if code != 0:
        lines = [l for l in err.splitlines() if not l.startswith("**")]
        r["err"] = (lines[-1] if lines else "")[:80]
        return r
    r.update(sessions=[], gpu_procs=[], users_cpu=[])
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "UP":
            r["uptime_h"] = round(float(v) / 3600, 1)
        elif k == "LOAD":
            r["load"] = [float(x) for x in v.split()]
        elif k == "NPROC":
            r["ncpu"] = int(v)
        elif k == "MEM":
            t, u = v.split(",")
            r["ram_total"], r["ram_used"] = int(t), int(u)
        elif k == "WHO":
            u, tty, frm = v.split(",", 2)
            r["sessions"].append({"user": u, "tty": tty,
                                  "remote": tty.startswith("pts") and frm.startswith("(") and not frm.startswith("(:")})
        elif k == "GPU":
            name, used, total, util, temp, power = [x.strip() for x in v.split(",")]
            r["gpu"] = {"name": name.replace("NVIDIA ", ""), "mem_used": num(used, int), "mem_total": num(total, int),
                        "util": num(util, int), "temp": num(temp, int), "power": num(power)}
        elif k == "GPROC":
            pid, mem, user, cmd = (v.split(",", 3) + ["", ""])[:4]
            r["gpu_procs"].append({"pid": num(pid, int), "mem": num(mem, int), "user": user, "cmd": cmd.strip()})
        elif k == "PSCJOB":
            jid, started = v.split(",", 1)
            r.setdefault("psc_jobs", []).append({"id": jid, "started": started})
        elif k == "PROC":
            u, cpu, mem, n, top = v.split(",", 4)
            if u in SYSTEM_USERS:
                continue
            r["users_cpu"].append({"user": u, "cpu": float(cpu), "mem_pct": float(mem), "nproc": int(n), "top": top})
    return r


def take_snapshot(write=True):
    """Sonde les 169 machines ; écrit dashboard/snapshot.json si `write`. Retourne le dict."""
    t0 = datetime.datetime.now()
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(probe_one, MACHINES))
    # commande d'origine depuis runs_state.json ; l'état vivant vient de la sonde
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for r in res:
        for j in r.get("psc_jobs", []):
            j["cmd"] = state.get(j["id"], {}).get("cmd", "")
    snap = {"taken_at": t0.strftime("%Y-%m-%d %H:%M:%S"), "me": os.environ.get("X_USER", ""),
            "duration_s": round((datetime.datetime.now() - t0).total_seconds(), 1), "machines": res}
    if write:
        SNAPSHOT.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    return snap


def main():
    snap = take_snapshot()
    up = sum(r["up"] for r in snap["machines"])
    print(f"{up}/{len(snap['machines'])} joignables en {snap['duration_s']} s -> {SNAPSHOT}")


if __name__ == "__main__":
    main()
