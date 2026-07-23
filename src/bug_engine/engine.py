from __future__ import annotations 

from scipy.spatial import cKDTree 
import numpy as np

from .config import SimParam

def neighbour_count(positions: np.ndarray, R: float, L: float) -> np.ndarray:

    """Count, for each particle, how many other particles lie within radius R.

    Uses a periodic (toroidal) KD-tree, so the domain wraps at L.

    Parameters
    ----------
    positions : (n, 2) array
        Particle coordinates in [0, L).
    R : float
        Neighbour radius.
    L : float
        Domain side length (periodic box size).

    Returns
    -------
    counts : (n,) int32 array
        Number of neighbours within R of each particle.
    """
    n = len(positions)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    tree = cKDTree(positions, boxsize=L)
    pairs = tree.query_pairs(R, output_type="ndarray")
    counts = np.bincount(pairs.ravel(), minlength=n)
    return counts.astype(np.int32)

def simulate(
    b0: float,
    d0: float,
    b1: float,
    d1: float,
    params: SimParam,
    rng: np.random.Generator | None = None,
):
    """Run one stochastic simulation of the spatial birth-death process.

    Particles diffuse on a periodic L x L domain. At each timestep, each
    particle's birth rate is ``b0 - b1 * (local neighbour count)`` (clipped
    at zero) and its death rate is ``d0 + d1 * (local neighbour count)``.
    Births duplicate a particle in place; deaths remove it.

    Parameters
    ----------
    b0, d0, b1, d1 : float
        Base birth/death rates and their linear dependence on local density.
    Initial Condition: N(t=0)=N0=mu/(s*pi*R^2)
    params : SimParams
        Fixed simulation settings (domain size, R , dt, N_max, Dr, t).
    rng : numpy.random.Generator, optional
        Random generator to use. If None, a fresh unseeded generator is
        created. Always pass an explicit rng when running in parallel
        (e.g. one per worker) to avoid correlated randomness across runs.

    Returns
    -------
    t_val : ndarray
        Time points.
    N_val : ndarray
        Population size at each time point (0 after extinction).
    """
    if rng is None:
        rng = np.random.default_rng()

    L, R, dt = params.L, params.R, params.dt
    N0,N_max, Dr, t_total =  params.N0,params.N_max, params.Dr, params.t
    t0=params.t0
    mu=b0-d0
    s=b1+d1
    
    t_val = [0]
    N_val = [N0]
    
    positions = rng.random((N0, 2)) * L
    m = 0
    total_steps = int(t_total / dt)

    while m < total_steps:
        n = len(positions)

        if n == 0:
            # Extinct: pad remaining timeline with zeros so output length
            # is always consistent regardless of when extinction occurred.
            remaining_steps = total_steps - m
            if remaining_steps > 0:
                remaining_times = (np.arange(m, total_steps) * dt).tolist()
                t_val.extend(remaining_times)
                N_val.extend([0] * remaining_steps)
            return np.array(t_val), np.array(N_val)

        if n > N_max:
            raise RuntimeError(f"Population exceeded N_max={N_max}.")

        positions += rng.normal(0, np.sqrt(2 * Dr * dt), size=positions.shape)
        positions = positions % L

        counts = neighbour_count(positions, R, L)
        birth_rate = b0 - b1 * counts
        birth_rate[birth_rate < 0] = 0
        birth_prob = birth_rate * dt
        death_prob = (d0 + d1 * counts) * dt
        total_prob = birth_prob + death_prob

        draw = rng.random(n)
        birth_events = draw < birth_prob
        death_events = draw >= birth_prob
        death_events &= draw < total_prob

        new_positions = np.repeat(positions[birth_events], 2, axis=0)

        positions = np.concatenate(
            (positions[~death_events & ~birth_events], new_positions)
        )
        m += 1
        t_val.append(m * dt)
        N_val.append(len(positions))
    t_val=np.array(t_val)
    N_val=np.array(N_val)
    mask = t_val >= t0
    t_val = t_val[mask]
    N_val = N_val[mask]

    return t_val, N_val

