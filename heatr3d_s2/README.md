# heatr3d_s2 -- the S2 convergence campaign

Isolation harness for Gate S2. `heatr3d.py` is DRIVEN, never edited (the
`heatr3d_s4_flir/` pattern).

    prereg.py        Task 0: the pre-registration, committed before any run
    bands.py         band + trend machinery (diverging FAILS, never bands)
    gauge.py         Task 2: the electrode-gauge arms and decision
    harness.py       one (shape, grid) case, dual read states, one shared EQS
    run_campaign.py  Task 3: the grid ladder, GRID-MAJOR, resumable
    aggregate.py     Task 3: bands + verdicts from the campaign
    densify.py       Task 4: the densify=True coupled march
    mechanisms.py    Task 5: L-shape corner + cylinder-null mechanisms
    make_tables.py   prints every report table from results/*.json

Read `S2_GATE_REPORT.md` for the verdict. Everything numeric there is printed
from `results/*.json`; nothing is transcribed.

Run order matters: `run_campaign.py` is grid-major and resumable, so an
interrupted campaign still leaves a 3-grid band -- the pre-registered minimum --
for every shape.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m heatr3d_s2.prereg
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m heatr3d_s2.run_campaign
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m heatr3d_s2.aggregate
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m heatr3d_s2.densify
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m heatr3d_s2.mechanisms
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m pytest heatr3d_s2/tests -q
