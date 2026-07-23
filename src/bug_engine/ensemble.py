
from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .config import SimParam
from .engine import neighbour_count,simulate
import os

def run_job(args):
    b0,d0,b1,d1,params,seed=args

    # Crucial: Initialize the RNG HERE, inside the process space!
    local_rng = np.random.default_rng(seed*100)
    
    return simulate(b0, d0, b1, d1, params, rng=local_rng)

def ensemble_simulate(
         b0: float,
    d0: float,
    b1: float,
    d1: float,
    params: SimParam,
    num_seeds: int,
    max_CPU:int | None=None
):
    if max_CPU is None:
        max_CPU=os.cpu_count()
    

    seed_list=np.arange(num_seeds)

    jobs=[(b0,d0,b1,d1,params,seed) for seed in seed_list]

    with ProcessPoolExecutor(max_workers=max_CPU) as executor:
        results=list(executor.map(run_job,jobs))
    N_ensemble=[]
    for t_val,N_val in results:
        N_ensemble.append(N_val)
    
    return t_val,N_ensemble
