#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PB_DIR="${ROOT_DIR}/paperbanana_figures"
GEN_DIR="${PB_DIR}/generated"
EVAL_DIR="${PB_DIR}/evaluations"
CTX_DIR="${PB_DIR}/contexts"
DATA_DIR="${PB_DIR}/data"

mkdir -p "${GEN_DIR}" "${EVAL_DIR}"

if [[ -z "${GOOGLE_API_KEY:-}" && -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

if [[ -z "${GOOGLE_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[paperbanana] Missing API key."
  echo "Set GOOGLE_API_KEY (or OPENROUTER_API_KEY) in environment or .env, then rerun."
  exit 1
fi

echo "[paperbanana] Generating methodology diagrams..."
paperbanana generate \
  -i "${CTX_DIR}/method_pipeline.txt" \
  -c "RFAM end-to-end workflow: coupled EQS-thermal-density forward model with inverse geometry correction loop and measurable uniformity outputs." \
  -o "${GEN_DIR}/fig_pb_method_pipeline.png"

paperbanana generate \
  -i "${CTX_DIR}/prewarp_loop.txt" \
  -c "EPE-based RFAM geometry prewarp algorithm: simulation, boundary shrinkage prediction, error projection, polygon update, smoothing, and convergence check." \
  -o "${GEN_DIR}/fig_pb_prewarp_loop.png"

paperbanana generate \
  -i "${CTX_DIR}/turntable_mechanism.txt" \
  -c "Discrete turntable rotation strategy in RFAM, including remap equation, 4x90-degree schedule, and why frequent early rotation outperforms one late rotation." \
  -o "${GEN_DIR}/fig_pb_turntable_mechanism.png"

paperbanana generate \
  -i "${CTX_DIR}/synergy_mechanism.txt" \
  -c "Mechanistic causal diagram showing why prewarp plus turntable yields super-additive boundary uniformity improvement for the H-shape." \
  -o "${GEN_DIR}/fig_pb_synergy_mechanism.png"

echo "[paperbanana] Generating quantitative plots from manuscript table data..."
paperbanana plot \
  -d "${DATA_DIR}/square_turntable_metrics.json" \
  --intent "Create a two-panel publication figure for square strategies: panel A compares boundary std (lower is better), panel B compares min/max ratio r_bnd (higher is better). Highlight 'Prewarp + 4x 90 deg' as best. Keep exact values and clear labels." \
  -o "${GEN_DIR}/fig_pb_square_turntable_metrics.png"

paperbanana plot \
  -d "${DATA_DIR}/hshape_metrics.json" \
  --intent "Create a two-panel publication figure for H-shape strategies: panel A compares boundary std and panel B compares r_bnd. Explicitly annotate baseline versus combined improvements and keep the values exact." \
  -o "${GEN_DIR}/fig_pb_hshape_metrics.png"

echo "[paperbanana] Running comparative evaluation where reference figures exist..."
paperbanana evaluate \
  -g "${GEN_DIR}/fig_pb_square_turntable_metrics.png" \
  --context "${DATA_DIR}/square_turntable_metrics.json" \
  -c "Boundary density uniformity comparison for square strategies with baseline, turntable variants, prewarp, and combined method." \
  -r "${ROOT_DIR}/rfam_paper_overleaf/figures/fig_uniformity_comparison.png" \
  > "${EVAL_DIR}/square_turntable_eval.txt"

paperbanana evaluate \
  -g "${GEN_DIR}/fig_pb_hshape_metrics.png" \
  --context "${DATA_DIR}/hshape_metrics.json" \
  -c "H-shape boundary density metrics for baseline, prewarp, turntable, and combined correction." \
  -r "${ROOT_DIR}/rfam_paper_overleaf/figures/fig_H_uniformity.png" \
  > "${EVAL_DIR}/hshape_eval.txt"

echo "[paperbanana] Done."
echo "[paperbanana] Generated figures: ${GEN_DIR}"
echo "[paperbanana] Evaluation summaries: ${EVAL_DIR}"
