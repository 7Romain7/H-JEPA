"""Etape 4 : evaluation de planning avec une tete de cout a la place du L2.

Meme protocole que la baseline. Seul le calcul du cout dans JEPA.criterion est
remplace par head.cost, remis a l'echelle du L2 par egalite des medianes sur le
premier lot de candidats, pour que le learning rate du planner garde son sens.

A lancer depuis h_jepa/ :
  PSC_HEAD=heads/qrl.pt python ../psc/eval_head.py --config-name cube_flat seed=42 \
      policy=$RUN/cube_lewm_object.ckpt output.dir=$RUN/eval_qrl
"""
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

from heads import load_head  # noqa: E402
from models import jepa  # noqa: E402

_STATE = {}


def criterion(self, info_dict):
    pred = info_dict["predicted_embed_0"][:, :, 1:, :]
    cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
    window = min(cost_last_n, pred.shape[-2])
    pred = pred[:, :, -window:, :].float()
    goal = info_dict["goal_embed_0"].expand_as(pred).detach().float()
    if "head" not in _STATE:
        _STATE["head"] = load_head(os.environ["PSC_HEAD"], pred.device).requires_grad_(False)
    cost = _STATE["head"].cost(pred, goal)  # (B, S, W)
    if "scale" not in _STATE:
        l2 = ((pred - goal) ** 2).sum(-1).detach()
        _STATE["scale"] = l2.median() / cost.detach().abs().median().clamp_min(1e-8)
        print(f"[psc] tete {os.environ['PSC_HEAD']} echelle {float(_STATE['scale']):.4g}")
    return (cost * _STATE["scale"]).sum(-1)


jepa.JEPA.criterion = criterion

import hydra  # noqa: E402

from planning_eval import run_planning_eval  # noqa: E402


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "config", "eval"),
            config_name="pusht_flat")
def run(cfg):
    run_planning_eval(cfg)


if __name__ == "__main__":
    run()
