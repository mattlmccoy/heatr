# PaperBanana Figure Workflow for `rfam_prewarp_paper.tex`

This folder provides a reproducible workflow to generate new manuscript figures with
[PaperBanana](https://github.com/llmsresearch/paperbanana) while keeping quantitative claims tied to
explicit data files from this project.

## 1) One-time setup

From project root:

```bash
paperbanana setup
```

Or set keys manually in `.env` at repo root:

```bash
GOOGLE_API_KEY=your_key_here
# Optional alternative providers:
# OPENROUTER_API_KEY=your_key_here
```

## 2) Generate all proposed figures

```bash
./paperbanana_figures/run_batch.sh
```

Outputs go to:

- `paperbanana_figures/generated/` for diagram/plot PNGs
- `paperbanana_figures/evaluations/` for comparative VLM evaluations

## 3) Copy selected outputs into Overleaf tree

```bash
cp paperbanana_figures/generated/*.png rfam_paper_overleaf/figures/
```

## Figure set in this pack

- `fig_pb_method_pipeline.png`: RFAM full physics + correction pipeline
- `fig_pb_prewarp_loop.png`: EPE-based prewarp algorithm loop
- `fig_pb_turntable_mechanism.png`: discrete rotation schedule and remapping
- `fig_pb_synergy_mechanism.png`: why prewarp + turntable is synergistic
- `fig_pb_square_turntable_metrics.png`: quantitative square strategy comparison
- `fig_pb_hshape_metrics.png`: quantitative H-shape strategy comparison

## Accuracy guardrails used here

- All quantitative plots use fixed JSON values from the manuscript tables.
- Context files include exact symbols and parameter values (e.g., `N=512`, `alpha=1.0`, `lambda_s=0.15`).
- `paperbanana evaluate` compares generated plots to your current human-crafted figure assets when a close reference exists.
