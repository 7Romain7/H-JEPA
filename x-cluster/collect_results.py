#!/usr/bin/env python3
"""Rapatrie et agrège les métriques d'une campagne (metrics.jsonl écrits par lewm.py).

  python collect_results.py --prefix i9            # télécharge + tableau + figures
  python collect_results.py --prefix i9 --no-fetch # réutilise resultats/i9/

Sorties dans resultats/<prefix>/ : summary.md, summary.csv, curves.png.
Métrique principale : pred_last_nmse en validation (erreur de prédiction du pas
suivant, dernière position de la fenêtre, normalisée par la variance de la cible
-> comparable entre bras malgré des espaces latents différents)."""
import argparse, io, json, math, os, re, subprocess, tarfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--prefix", default="i9")
p.add_argument("--host", default="ablette.polytechnique.fr")
p.add_argument("--no-fetch", action="store_true")
a = p.parse_args()
out = HERE / "resultats" / a.prefix
out.mkdir(parents=True, exist_ok=True)

if not a.no_fetch:
    user = os.environ.get("X_USER")
    target = f"{user}@{a.host}" if user else a.host
    cmd = (f"cd ~/psc/runs && tar czf - --ignore-failed-read "
           f"$(ls -d {a.prefix}_* | sed 's|$|/metrics.jsonl|') "
           f"$(ls -d {a.prefix}_*/exit_code {a.prefix}_*/host.txt {a.prefix}_*/cmd.txt 2>/dev/null) "
           f"$(ls -d {a.prefix}_*/eval.json ev_{a.prefix}_*/eval.json ev_{a.prefix}_*/exit_code 2>/dev/null) 2>/dev/null")
    data = subprocess.run(["ssh", "-o", "BatchMode=yes", target, cmd], capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(out / "runs", filter="data")

rows, curves = [], []
for d in sorted((out / "runs").glob(f"{a.prefix}_*")):
    m = re.fullmatch(rf"{a.prefix}_(.+)_s(\d+)", d.name)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    r = {"job": d.name, "arm": arm, "seed": seed,
         "exit": (d / "exit_code").read_text().strip() if (d / "exit_code").exists() else "running",
         "host": (d / "host.txt").read_text().strip() if (d / "host.txt").exists() else ""}
    f = d / "metrics.jsonl"
    if f.exists() and f.stat().st_size:
        df = pd.read_json(f, lines=True)
        tr = df[df.stage == "fit"]
        va = df[(df.stage != "fit") & (df.n > 5)]          # > 5 : exclut le sanity check
        r["steps"] = int(tr.step.max()) if len(tr) else 0
        r["hours"] = round(df.time_s.max() / 3600, 2)
        if len(va):
            last = va.iloc[-1]
            r["val_epoch"] = int(last.epoch)
            for k in ("pred_last_nmse", "pred_last_mse", "pred_loss", "sigreg_loss",
                      "scal_pdf", "scal_inv", "gamma_mean", "gamma_std", "jw_norm", "u_norm",
                      "u_scal_inv", "u_gamma_mean", "u_gamma_std", "u_jw_norm", "u_cos",
                      "abl_u0_nmse", "abl_ushuf_nmse"):
                if k in va:
                    r[f"val_{k}"] = last[k]
        if len(tr):
            tail = tr.tail(5)
            for k in ("pred_last_nmse", "scal_pdf", "scal_inv", "gamma_mean", "jw_norm"):
                if k in tail:
                    r[f"train_{k}"] = tail[k].mean()
            tr = tr.assign(arm=arm, seed=seed)
            curves.append(tr)
    # taux de succès en planification : évalué dans le job (eval.json) ou par un
    # job d'évaluation séparé ev_<job> (make_eval_jobs.py)
    for e in (d / "eval.json", out / "runs" / f"ev_{d.name}" / "eval.json"):
        if e.exists():
            r["success_rate"] = json.loads(e.read_text())["success_rate"]
    rows.append(r)

runs = pd.DataFrame(rows)
runs.to_csv(out / "runs.csv", index=False)
if runs.empty:
    raise SystemExit("aucun run trouvé")

ARM_ORDER = (["base", "phase"] + [f"{n}{l}" for n in ("pdf", "inv") for l in ("0p01", "0p1", "1")]
             + ["hist", "hist_u0p01", "hist_u0p1", "hist_r0p1"])
arms = [x for x in ARM_ORDER if x in set(runs.arm)] + sorted(set(runs.arm) - set(ARM_ORDER))
# référence pour isoler L_scal : l'Idée 9 sans L_scal de la même architecture
REF = "hist" if "hist" in arms else "phase"


def welch(x, y):
    """t de Welch et p bilatéral (approx. normale si scipy absent)."""
    x, y = x.dropna(), y.dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    vx, vy = x.var() / len(x), y.var() / len(y)
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    try:
        from scipy import stats
        dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        return t, 2 * stats.t.sf(abs(t), dof)
    except ImportError:
        return t, math.erfc(abs(t) / math.sqrt(2))


full_epoch = runs.val_epoch.max() if "val_epoch" in runs else None
done = runs[runs.get("val_epoch") == full_epoch] if full_epoch is not None else runs.iloc[:0]
cols = ["val_pred_last_nmse", "val_pred_loss", "val_scal_pdf", "val_scal_inv",
        "val_gamma_mean", "val_gamma_std", "val_jw_norm", "val_u_norm"]
# campagnes >= g9 : sonde le long de u_t et ablations de la vitesse
if "val_u_gamma_mean" in runs:
    cols = ["val_pred_last_nmse", "val_abl_ushuf_nmse", "val_abl_u0_nmse", "val_scal_inv",
            "val_u_scal_inv", "val_u_gamma_mean", "val_u_jw_norm", "val_u_cos", "val_jw_norm"]
lines = [f"# Campagne {a.prefix}\n",
         f"{len(runs)} jobs, {len(done)} avec la validation finale (epoch {full_epoch}). "
         "Valeurs : moyenne ± erreur standard sur les seeds.\n",
         "| bras | n | " + " | ".join(c.removeprefix("val_") for c in cols) + f" | p vs {REF} (nmse) | p vs base (nmse) |",
         "|" + "---|" * (len(cols) + 4)]
summ = []
for arm in arms:
    g = done[done.arm == arm]
    cells = []
    for c in cols:
        if c in g and g[c].notna().any():
            v = g[c].dropna()
            cells.append(f"{v.mean():.4g} ± {v.std() / math.sqrt(len(v)) if len(v) > 1 else 0:.2g}")
        else:
            cells.append("—")
    pv = [welch(g.get("val_pred_last_nmse", pd.Series(dtype=float)),
                done[done.arm == ref].get("val_pred_last_nmse", pd.Series(dtype=float)))[1]
          for ref in (REF, "base")]
    lines.append(f"| {arm} | {len(g)} | " + " | ".join(cells) + " | " +
                 " | ".join("—" if math.isnan(x) else f"{x:.3g}" for x in pv) + " |")
    summ.append({"arm": arm, "n": len(g), **{c: g[c].mean() if c in g else None for c in cols}})
pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
if "success_rate" in runs and runs.success_rate.notna().any():
    rnd = out / "runs" / f"ev_{a.prefix}_random" / "eval.json"
    lines += ["", "## Taux de succès en planification (%)", ""]
    if rnd.exists():
        lines.append(f"Politique aléatoire (référence) : {json.loads(rnd.read_text())['success_rate']:.1f} %\n")
    lines += [f"| bras | n | succès (%) | p vs {REF} | p vs base |", "|---|---|---|---|---|"]
    ev = runs[runs.success_rate.notna()]
    for arm in arms:
        v = ev[ev.arm == arm].success_rate
        if not len(v):
            continue
        pv = [welch(v, ev[ev.arm == ref].success_rate)[1] for ref in (REF, "base")]
        lines.append(f"| {arm} | {len(v)} | {v.mean():.1f} ± {v.std() / math.sqrt(len(v)) if len(v) > 1 else 0:.1f} | "
                     + " | ".join("—" if math.isnan(x) else f"{x:.3g}" for x in pv) + " |")
status = runs.exit.value_counts().to_dict()
lines += ["", f"Statut des jobs (exit_code) : {status}"]
(out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if curves:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.concat(curves)
    keys = [k for k in ("pred_last_nmse", "pred_loss", "scal_inv", "scal_pdf", "gamma_mean", "jw_norm") if k in c]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k in zip(axs.flat, keys):
        for arm in arms:
            g = c[c.arm == arm]
            if g[k].notna().any():
                g = g.assign(bin=(g.step // 1000) * 1000).groupby("bin")[k].mean()
                ax.plot(g.index, g.values, label=arm, color=f"C{arms.index(arm)}")
        ax.set_title(k); ax.set_xlabel("pas"); ax.set_yscale("log" if k not in ("gamma_mean",) else "linear")
    for ax in axs.flat:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=120)
    print("figures :", out / "curves.png")
