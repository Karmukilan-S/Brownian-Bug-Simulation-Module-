from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from .config import SimParam
from .ensemble import ensemble_simulate
from .stat import stat0
from .Master_equation import Mu_sweep_1,s_sweep_1,Mu_sweep_2

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAVE_DIR_1 = ROOT / "Plots" / "Seed_convergence"
SAVE_DIR_2=ROOT/"Plots"/"master_equation"
SAVE_DIR_3=ROOT/"Plots"/"master_equation_s"
SAVE_DIR_1.mkdir(parents=True, exist_ok=True)
SAVE_DIR_2.mkdir(parents=True,exist_ok=True)
SAVE_DIR_3.mkdir(parents=True,exist_ok=True)

def plot_seed_convergence(b0:float,
          d0:float,
          b1:float,
          d1:float,
          params:SimParam,
          seeds_val:list,
          max_CPU:int|None=None,
          save=False,
    save_dir=None,
    filename="seed_convergence.png"
):
    n=max(seeds_val)
    t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,n,max_CPU)
    N_ensemble=np.array(N_ensemble)
    plt.figure(figsize=(17,8))
    plt.suptitle(f"Ensemble Mean and Variance for different number of seeds with (b0,d0,b1,d1)={(b0,d0,b1,d1)}"
                 f"and dt={params.dt},Dr={params.Dr},R={params.R}",fontsize=18)
    
    plt.subplot(1,2,1)
    plt.title("<N> vs time")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("<N>")

    plt.subplot(1,2,2)
    plt.title("Var(N) vs time")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Var(N)")

    Rng = np.random.default_rng(42)
    for seed in seeds_val:
        indices=Rng.choice(n, size=seed, replace=False)
        Ensemble_seed=N_ensemble[indices]
        t_val,mean_N,Var_N=stat0(t_val,Ensemble_seed)
        plt.subplot(1,2,1)
        plt.plot(t_val,mean_N,label=f"Seed={seed}")
        plt.subplot(1,2,2)
        plt.plot(t_val,Var_N,label=f"Seed={seed}")
    plt.subplot(1,2,2)
    plt.legend(loc="upper right",fontsize=14)
    filename=f"Seed_convergence_(b0,d0,b1,d1)={(b0,d0,b1,d1)}_Simparam={params}_seed_val_max={n}.png"
    if save:
        plt.savefig(SAVE_DIR_1/filename,dpi=300)
    else:
        plt.show()
    
def master_eqn_plot(mu_val:np.ndarray,
            x:float,
             b1:float,
             d1:float,
             params:SimParam,
            num_seeds:int,
            max_CPU:int | None=None,
            save_dir=None,
    filename="master_eqn.png"
            ):
    R=params.R
    mu_val_1 , mean_N_val , var_N_val , eta_2_val = Mu_sweep_1(mu_val,x,b1,d1,params,num_seeds,max_CPU)
    N_star_val=mu_val_1/((d1+b1)*R*R*np.pi)
    
    plt.figure(figsize=(15,8))
    plt.suptitle(f"1st Master equation prediction for b1={b1},d1={d1},frac={x},Simparams={params},num_seeds={num_seeds}")
    
    plt.subplot(1,2,1)
    plt.grid()
    plt.title(f"Mean Prediction")
    plt.plot(mu_val_1,mean_N_val,'-o',label='<N>')
    plt.plot(mu_val_1,N_star_val,'--',label='N*')
    plt.plot(mu_val_1,N_star_val-eta_2_val/N_star_val,'-s',label="N*-<eta^2>/N*")
    plt.legend(loc='upper right',fontsize=12)
    plt.xlabel("Mu")
    plt.ylabel("Population")
    
    plt.subplot(1,2,2)
    plt.grid()
    plt.title(f"Variance Prediction")
    plt.plot(mu_val_1,var_N_val,label="Var(N)")
    plt.plot(mu_val_1,mean_N_val*np.abs(mean_N_val-N_star_val),label="<N>|N*-<N>|")
    plt.xlabel("Mu")
    plt.ylabel("Variance")
    plt.legend(loc="upper right",fontsize=12)
    file_name=f"Mu_sweep_1_b1={b1}_d1={d1}_Simparam={params}.png"
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir/file_name,dpi=300)

def master_eqn_plot_2(
        
                      b1:float,
                      d1:float,
                      b0_size:int,
                      params:SimParam,
                      seeds:int,
                      max_CPU:int|None=None,
                      save_dir=None,
                      filename="plot.png"
                        ):
    mu_val,mean_N,pred,Var_N=Mu_sweep_2(b1,d1,b0_size,params,seeds,max_CPU)
    R=params.R
    N_star=mu_val/(np.pi*R*R*(d1+b1))

    plt.figure(figsize=(15,8))
    plt.suptitle(f"1st Master equation prediction:b0+d0=1_b1={b1},d1={d1},Simparams={params},num_seeds={seeds}")
    
    plt.subplot(1,2,1)
    plt.grid()
    plt.title(f"Mean Prediction")
    plt.plot(mu_val,mean_N,'-o',label='<N>')
    plt.plot(mu_val,N_star,'--',label='N*')
    plt.plot(mu_val,pred,'-s',label="N*-<eta^2>/N*")
    plt.legend(loc='upper right',fontsize=12)
    plt.xlabel("Mu")
    plt.ylabel("Population")
    
    plt.subplot(1,2,2)
    plt.grid()
    plt.title(f"Variance Prediction")
    plt.plot(mu_val,Var_N,label="Var(N)")
    plt.plot(mu_val,mean_N*np.abs(mean_N-N_star),label="<N>|N*-<N>|")
    plt.xlabel("Mu")
    plt.ylabel("Variance")
    plt.legend(loc="upper right",fontsize=12)
    file_name=f"Mu_sweep_2_b_0+d_0=1_b1={b1}_d1={d1}_simparam={params}_seed={seeds}.png"
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir/file_name,dpi=300)

def master_eqn_plot_s(val,s:float,d1_size:int,
                
                      params:SimParam,num_seeds:int,
                      save_dir=None,max_CPU:int|None=None):
    
    
    plt.figure(figsize=(8,8))
    plt.xlabel("d_1")
    plt.title(f"d1_sweep_for_s={s}_SimParam={params}_num_seeds={num_seeds}")
    plt.ylabel("Population")
    plt.grid()
    for b0,d0 in val:
        d1_val,mean_N,pred_N=s_sweep_1(b0,d0,s,d1_size,params,num_seeds)
        plt.plot(d1_val,mean_N,'-o',label=f"<N> for b0={b0},d0={d0}")
        plt.plot(d1_val,pred_N,'--',label=f"N*-<eta^2>/N* for b0={b0},d0={d0}")
    R=params.R
    N_star=(b0-d0)/(np.pi*R*R*s)
    plt.plot(d1_val,np.full_like(d1_val,N_star),':',label='N*',color='black')
    plt.legend(loc='upper right',fontsize=10)
    if save_dir is None:
        plt.show()
    else:
        file_name=f"d1_sweep_with_s={s}_interval_size={d1_size}_params={params}_num_seeds={num_seeds}_b0,d0_pairs={val}.png" 
        plt.savefig(save_dir/file_name,dpi=300)   
    
def plot_dist(b0,d0,b1,d1,params,seed):
    plt.figure(figsize=(8,8))
    plt.title(f"b0={b0},d0={d0},b1={b1},d1={d1},{params}")

    
    t_val,N_ensemble=ensemble_simulate(b0,d0,b1,d1,params,seed)
    N_ensemble=list(x[-1] for x in N_ensemble)
    plt.hist(N_ensemble,bins=200,density=True,label=f"seed={seed}")
    plt.grid()
    #plt.legend(loc='upper right',fontsize=10)
    plt.show()

