from __future__ import annotations 

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .config import SimParam
from .engine import neighbour_count,simulate
from .ensemble import run_job,ensemble_simulate

def stat0(t_val,N_ensemble):
    stacks=np.stack(N_ensemble)
    mean_N=np.mean(stacks,axis=0)
    Var_N=np.var(stacks,axis=0,ddof=1)
    return t_val,mean_N,Var_N


def stat1( b0:float,
          d0:float,
          b1:float,
          d1:float,
          params:SimParam,
          num_seeds:int,
          max_CPU:int
        ):
    t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,num_seeds,max_CPU)

    return stat0(t_val,N_ensemble)









 