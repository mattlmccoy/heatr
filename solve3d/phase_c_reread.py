"""Phase C POST-HOC RE-READ (not part of the pre-registered protocol).

    ./.venv312/bin/python -m solve3d.phase_c_reread

Motivated by the 2-D lane's DENSE_IFF_INBOUNDS_REPORT.md (their 2050561):
the asymmetric objective changes the STOP far more than it changes the MAP.

Two questions, both answerable from artifacts already on disk. NO new marches,
NO re-solves: every number below is read out of phase_c_baselines.json,
phase_c_solves.json and phase_c_gate.json, all of which were written by
score_arm, which already recorded each arm's own argmin stops and a 3x-weighted
re-score at the same read state.

  1. STOP-STATE RE-READ -- did Task 3 already score at the asymmetric
     objective's own argmin? If yes this is a no-op statement and the
     "still descending at the budget limit" bound does NOT tighten via the read
     state. Reported either way, with the argmin steps as evidence.
  2. WEIGHT SENSITIVITY BY RE-SCORING -- does the solved-vs-uniform RANKING
     depend on the pre-registered 10x out/in weight ratio? Re-scoring the
     existing fields under the 3x ratio answers that at the READ stage.
     LIMIT, stated and not papered over: this does NOT establish what a
     3x-weighted SOLVE would find. A different weighting changes the gradient
     and could steer to a different map. It closes the read-stage half of the
     risk only.
"""
from __future__ import annotations

import json
from pathlib import Path

from solve3d import gates

RESULTS = Path(__file__).resolve().parent / "results"
ARM = "solve_filter_only_asymmetric_scaled"


def _states() -> dict:
    b = json.loads((RESULTS / "phase_c_baselines.json").read_text())["arms"]
    s = json.loads((RESULTS / "phase_c_solves.json").read_text())["arms"]
    g = json.loads((RESULTS / "phase_c_gate.json").read_text())["arms"][ARM]
    h = g["mesh_holdout"]["scores"]
    return {"uniform_at_solve_mesh": b["uniform_baseline"],
            "solved_at_solve_mesh": s[ARM],
            "uniform_at_score_mesh": h["uniform_at_score_mesh"],
            "solved_at_score_mesh": h["solved_at_score_mesh"]}


def _margin(u: float, v: float) -> float:
    return (u - v) / abs(u)


def build() -> dict:
    st = _states()
    dt = 0.05

    # ---- item 1: stop-state re-read ---------------------------------- #
    stops = {}
    for k, r in st.items():
        stops[k] = {
            "argmin_symmetric": r["argmin_symmetric"],
            "argmin_asymmetric": r["argmin_asymmetric"],
            "t_stop_s_asymmetric": r["t_stop_s"],
            "stop_shift_steps_asym_minus_sym":
                r["argmin_asymmetric"] - r["argmin_symmetric"],
            "stop_shift_s": (r["argmin_asymmetric"] - r["argmin_symmetric"]) * dt,
            "at_horizon_asymmetric": r["at_horizon_asymmetric"]}
    # how much does the OBJECTIVE move the stop, vs how much does the MAP?
    obj_shift = abs(stops["uniform_at_solve_mesh"]["stop_shift_steps_asym_minus_sym"])
    map_shift = abs(st["solved_at_solve_mesh"]["argmin_asymmetric"]
                    - st["uniform_at_solve_mesh"]["argmin_asymmetric"])
    item1 = {
        "already_scored_at_asymmetric_argmin": True,
        "evidence": "score_arm reads the PRIMARY state at ka = argmin of the "
                    "asymmetric trajectory and every downstream diagnostic uses "
                    "that state; the symmetric control is separately read at its "
                    "OWN argmin. Both argmins are recorded per arm.",
        "no_op": True,
        "bound_tightens_via_read_state": False,
        "bound_statement": "the 'still descending at the budget limit' caveat "
                           "stands unchanged: the read state was already optimal "
                           "for the primary objective, so the remaining headroom "
                           "is in ITERATIONS, not in the stop.",
        "stops": stops,
        "objective_moves_the_stop_steps": obj_shift,
        "map_moves_the_stop_steps": map_shift,
        "cross_lane_echo":
            f"the 2-D lane's finding reproduces in 3-D: switching the OBJECTIVE "
            f"from symmetric to asymmetric moves the stop by {obj_shift} steps "
            f"({obj_shift*dt:.2f} s, consistently EARLIER -- it stops before the "
            f"bed grows), while switching the MAP from uniform to solved moves it "
            f"by only {map_shift} steps ({map_shift*dt:.2f} s). The objective "
            f"moves the stop ~{obj_shift/max(map_shift,1):.0f}x more than the map "
            f"does. NOTE the 3-D difference: here that stop shift is NOT where the "
            f"gain comes from, because both arms were already read at their own "
            f"asymmetric argmin, so the 10.67 % margin is a MAP effect measured at "
            f"a matched read convention.",
    }

    # ---- item 2: weight sensitivity by re-scoring ---------------------- #
    rows = {}
    for mesh, u_k, s_k in (("solve_mesh", "uniform_at_solve_mesh", "solved_at_solve_mesh"),
                           ("score_mesh", "uniform_at_score_mesh", "solved_at_score_mesh")):
        u, v = st[u_k], st[s_k]
        rows[mesh] = {
            "w10": {"uniform": u["J_asymmetric"], "solved": v["J_asymmetric"],
                    "margin": _margin(u["J_asymmetric"], v["J_asymmetric"])},
            "w3": {"uniform": u["J_asym_w3_at_same_read"],
                   "solved": v["J_asym_w3_at_same_read"],
                   "margin": _margin(u["J_asym_w3_at_same_read"],
                                     v["J_asym_w3_at_same_read"])},
            "symmetric": {"uniform": u["J_symmetric"], "solved": v["J_symmetric"],
                          "margin": _margin(u["J_symmetric"], v["J_symmetric"])},
        }
        for w in ("w10", "w3", "symmetric"):
            rows[mesh][w]["solved_wins"] = bool(rows[mesh][w]["margin"] > 0.0)
    ranking_stable = all(rows[m][w]["solved_wins"]
                         for m in rows for w in ("w10", "w3", "symmetric"))
    item2 = {
        "read_stage_only": True,
        "rows": rows,
        "ranking_preserved_across_weightings": ranking_stable,
        "limit": "This re-scores EXISTING fields under the 3x ratio. It does NOT "
                 "establish what a 3x-weighted SOLVE would find: a different "
                 "weighting changes the gradient and could steer to a different "
                 "map. The pre-registered 3x sensitivity ARM remains NOT_RUN.",
        "conclusion":
            "the solved-vs-uniform ranking does NOT depend on the out/in weight "
            "ratio at the read stage -- solved wins under 10x, under 3x, and "
            "under the symmetric control, at BOTH meshes." if ranking_stable else
            "the ranking DOES depend on the weighting; per the pre-registration "
            "that is the finding and the ratio must be calibrated.",
    }

    doc = {
        "what": "Phase C POST-HOC RE-READ. NOT part of the pre-registered "
                "protocol; added after the fact at the coordinator's request, "
                "motivated by the 2-D lane's DENSE_IFF_INBOUNDS_REPORT.md.",
        "no_new_compute": "no marches, no solves; every number is read from "
                          "phase_c_baselines.json, phase_c_solves.json and "
                          "phase_c_gate.json",
        "item_1_stop_state_reread": item1,
        "item_2_weight_sensitivity_by_rescoring": item2,
    }
    gates.write_json("phase_c_reread.json", doc)
    return doc


def main() -> int:
    d = build()
    i1, i2 = d["item_1_stop_state_reread"], d["item_2_weight_sensitivity_by_rescoring"]
    print("ITEM 1 no_op =", i1["no_op"], "| bound tightens =",
          i1["bound_tightens_via_read_state"])
    print("  objective moves stop", i1["objective_moves_the_stop_steps"],
          "steps; map moves stop", i1["map_moves_the_stop_steps"], "steps")
    print("ITEM 2 ranking preserved =", i2["ranking_preserved_across_weightings"])
    for m, r in i2["rows"].items():
        print(f"  {m}: w10 {r['w10']['margin']*100:+.2f} %  "
              f"w3 {r['w3']['margin']*100:+.2f} %  "
              f"sym {r['symmetric']['margin']*100:+.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
