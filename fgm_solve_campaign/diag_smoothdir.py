"""Supplementary probe for the FILTERED layer: a smooth random direction.

A random unit vector in the design space v is dominated by cell-scale content,
which the filter is built to remove, so F(d) is small and the finite-difference
signal is damped while the objective's roundoff floor is unchanged. The
physically meaningful design perturbation is one the filter passes.
"""
import numpy as np
from adjoint2d.pins import build_case, load_cfg
from adjoint2d import library_solve as lib, density_objective as dobj, design_filter as df
from adjoint2d import adjoint, gradops, gate_rho as gr

case = build_case(load_cfg(lib.shape_config('square'))); pm = case.part_mask
ops = gradops.gradient_matrices(case.x, case.y)
v0 = gr.default_v(case); SIG = 1.5
tr = gr.run_forward(case, df.apply_filter(v0, pm, SIG))
st = dobj.rho_stop(tr, case); IDX = st.index
J0, seed = dobj.rho_J_and_seed(tr.rho_at_end(IDX), case)
g_s = adjoint.gradient(case, df.apply_filter(v0, pm, SIG), tr, {}, grad_ops=ops,
                       seeds_rho={IDX: seed})
g = df.filter_vjp(g_s, pm, SIG)
del tr
rng = np.random.default_rng(7)
d_raw = np.zeros(pm.shape); u = rng.standard_normal(int(pm.sum()))
d_raw[pm] = u / np.linalg.norm(u)
fd_raw = df.apply_filter(d_raw, pm, SIG, outside=0.0)
print('filter damping of the rough random direction: ||F d|| / ||d|| = %.4f'
      % (np.linalg.norm(fd_raw[pm]) / np.linalg.norm(d_raw[pm])), flush=True)
d_sm = np.zeros(pm.shape)
d_sm[pm] = fd_raw[pm] / np.linalg.norm(fd_raw[pm])
print('damping of the smooth direction: %.4f'
      % (np.linalg.norm(df.apply_filter(d_sm, pm, SIG, outside=0.0)[pm])), flush=True)
ana = float(np.sum(g * d_sm))
print('J0 %.6f  read %d  analytic along the smooth direction %+.6e' % (J0, IDX, ana), flush=True)
for e in (1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8):
    jp = dobj.rho_J_and_seed(gr.run_forward(case, df.apply_filter(v0 + e * d_sm, pm, SIG)).rho_at_end(IDX), case)[0]
    jm = dobj.rho_J_and_seed(gr.run_forward(case, df.apply_filter(v0 - e * d_sm, pm, SIG)).rho_at_end(IDX), case)[0]
    fdv = (jp - jm) / (2 * e)
    print('eps %.0e  fd %+.8e  rel %.3e' % (e, fdv, abs(fdv - ana) / abs(ana)), flush=True)
