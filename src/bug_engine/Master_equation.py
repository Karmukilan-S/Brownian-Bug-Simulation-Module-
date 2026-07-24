from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from .config import SimParam
from .engine import simulate
from .ensemble import run_job,ensemble_simulate
from .stat import stat2


def Mu_sweep_1(mu_val:np.ndarray,
            x:float,
             b1:float,
             d1:float,
             params:SimParam,
            num_seeds:int,
            max_CPU:int | None=None
             ):
    if x<1:
        raise ValueError(f"fraction entry {x} should be greater than 1")
    R=params.R
    mean_N=[]
    Var_N=[]
    eta_2=[]
    for mu in mu_val:
        b0=x*mu
        d0=(x-1)*mu
        t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,num_seeds,max_CPU)
        N_ensemble=np.array([seed[-1] for seed in N_ensemble])
        N_star=mu/(np.pi*R*R*(b1+d1))
        eta=N_ensemble-N_star
        mean_N.append(np.mean(N_ensemble))
        Var_N.append(np.var(N_ensemble))
        eta_2.append(np.mean(eta*eta))
    return mu_val,np.array(mean_N),np.array(Var_N),np.array(eta_2)

def Mu_sweep_2(b1:float,
                      d1:float,
                      b0_size:int,
                      params:SimParam,
                      num_seeds:int,
                      max_CPU:int|None=None):
    R=params.R
    b0_val=np.linspace(0.45,0.8,b0_size)
    pairs=[(b0,1-b0) for b0 in b0_val]
    mu_val=[]
    N_mean_1=[]
    N_mean_2=[]
    Var_N=[]
    for b0,d0 in pairs:
        t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,num_seeds,max_CPU)
        N_ensemble=np.array([seed[-1] for seed in N_ensemble])
        mu_val.append(b0-d0)
        Var_N.append(np.var(N_ensemble))
        N_mean_1.append(np.mean(N_ensemble))
        if b0!=d0:
            N_star=(b0-d0)/(np.pi*R*R*(b1+d1))
            eta=N_ensemble-N_star
            N_mean_2.append(N_star-np.mean(eta*eta)/N_star)
        else:
            N_mean_2.append(0)
        print(f"b0={b0:.3f},d0={d0:.3f},b1={b1:.3f},d1={d1:.3f}|" 
              f"N*={N_star:.3f}|"
              f"<N>={np.mean(N_ensemble):.3f}|"
              f"pred={N_star-np.mean(eta*eta)/N_star:.3f}|") 
    return np.array(mu_val),np.array(N_mean_1),np.array(N_mean_2),np.array(Var_N)

def s_sweep_1(
              b0:float,
              d0:float,
              s:float,
              d1_size:int,
              params:SimParam,
              num_seeds:int,
              max_CPU:int | None=None
              ):
    R=params.R
    if d1_size<1:
        raise ValueError("d1_size should be an integer greater than or equal to 1")
    
    d1_val=np.linspace(s/2,s,d1_size)
    b1_val=s-d1_val
    pairs=np.column_stack((b1_val,d1_val))
    mean_N=[]
    eta_2=[]
    for b1,d1 in pairs:
        t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,num_seeds,8)
        N_ensemble= np.array([seed[-1] for seed in N_ensemble])
        N_star=(b0-d0)/(np.pi*R*R*s)
        mean_N.append(np.mean(N_ensemble))
        eta=N_ensemble-N_star
        eta_2.append(np.mean(eta*eta))
        print(f"b0={b0:.3f},d0={d0:.3f},b1={b1:.3f},d1={d1:.3f}|" 
              f"N*={N_star:.3f}|"
              f"<N>={np.mean(N_ensemble):.3f}|"
              f"pred={N_star-np.mean(eta*eta)/N_star:.3f}|")
    return d1_val,np.array(mean_N),N_star-np.array(eta_2)/N_star

def mu_sweep_fano(b1:float,
                      d1:float,
                      b0_size:int,
                      params:SimParam,
                      seeds:int,
                      max_CPU:int|None=None):
    b0_val=np.linspace(0.55,1,b0_size)
    mu_val=[]
    mean_N_val=[]
    fano_val=[]
    for b0 in b0_val:
        mu,meanN,fano=stat2(b0,1-b0,b1,d1,params,seeds,max_CPU)
        mu_val.append(mu)
        mean_N_val.append(meanN)
        fano_val.append(fano)
        print(f"mu={2*b0-1:.3f}||<N>={meanN:.3f}||"
              f"pred<N>={(2*b0-1)*(params.L)*(params.L)/((b1+d1)*np.pi*params.R*params.R)-(1-b0)/(2*b0-1)+d1/(d1+b1):.3f}||"
              f"fano={fano}||fano_pred={(1-b0)/(2*b0-1)+d1/(d1+b1):.3f}")
    fano_pred=(1-b0_val)/(2*b0_val-1)+d1/(d1+b1)
    N_star=(2*b0_val-1)*(params.L)*(params.L)/((b1+d1)*np.pi*params.R*params.R)
    mean_pred=N_star-fano_pred
    
    return np.array(mu_val),np.array(mean_N_val),np.array(mean_pred),np.array(fano),np.array(fano_pred)
