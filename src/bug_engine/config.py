from dataclasses import dataclass

@dataclass(frozen=True)
class SimParam:
    '''------------------------------------------
    ATTRIBUTES OF CLASS:

    -> SPATIAL
    L - Length of 2D Square Domain
    R - Interaction Radius 
    Dr - Diffusion Coeffecient

    ->POPULATION
    N_max - Allowed Maximum Population of the Model 

    ->TEMPORAL
    dt - resolution
    t0 - time allocated for the system to achieve the stationary state
    t - total run time

    '''
    L:float=1.0
    R:float=0.1
    Dr:float=1e-1
    
    N_max:int=50000
    N0:int=150
    
    dt:float=0.01
    t:float=50
    t0:float=0

