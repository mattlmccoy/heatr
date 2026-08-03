# Pin every numerical library to ONE thread.
# MEASURED: unpinned, each solve process took 5.6 cores of BLAS threads and a
# single square forward still cost 10.7 s, against 9.9 s for the same call in
# the single-start library campaign. The thread parallelism buys nothing on a
# sparse 120 x 120 solve and destroys throughput when several shapes run at
# once, so every stream is pinned to one thread and parallelism is taken across
# shapes instead.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
