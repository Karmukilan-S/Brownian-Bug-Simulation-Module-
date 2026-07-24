from __future__ import annotations 
import os
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
          max_CPU:int|None=None
        ):
    if max_CPU is None:
        max_CPU=os.cpu_count()
    t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,num_seeds,max_CPU)

    return stat0(t_val,N_ensemble)

def stat2(          b0:float,
                    d0:float,
                    b1:float,
                      d1:float,
                      params:SimParam,
                      seeds:int,
                      max_CPU:int|None=None):
    t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,seeds,max_CPU)
    N_ensemble=np.array([seed[-1] for seed in N_ensemble])
    mean_N,var_N=np.mean(N_ensemble),np.var(N_ensemble)
    if mean_N==0:
        f=0
    else:
        f=var_N/mean_N
    return b0-d0,mean_N,var_N,f





 