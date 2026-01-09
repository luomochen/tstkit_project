import yaml
import numpy as np
import pandas as pd
from numba import njit
import multiprocessing as mp
from .rates import RatesTable
from .events import count_time, pad_ragged_2d, Events

@njit
def kmc_one_step(ini, rates, total_rate, targets, n_events):
    """Perform one KMC step from site ini."""
    rho = np.random.random() * total_rate[ini]
    acc = 0.0

    for k in range(n_events[ini]):
        acc += rates[ini, k]
        if acc > rho:
            fin = targets[ini, k]
            break

    dt = -np.log(np.random.random()) / total_rate[ini]
    return dt, fin

@njit
def kmc_iteration(t_limit, n_steps, rates, total_rate, targets, n_events, events_count, ini):
    """A kmc iteration."""
    t = 0.0
    steps = 0

    while steps < n_steps and t < t_limit:
        dt, fin = kmc_one_step(ini, rates, total_rate, targets, n_events)
        events_count[ini, fin] += 1
        ini = fin
        t += dt
        steps += 1

    return t, steps, events_count

def kmc_loop(queue, *args):
    """Warp it for multiprocessing."""
    t, steps, events_count = kmc_iteration(*args)
    queue.put((t, steps, events_count))

@count_time(print_args_index=0)
def run_kmc_at_temperature(temperature, cfg, rates: RatesTable, events: Events):
    print(f"Running {temperature} K...")

    events.precompute_jump_vectors()
    events_count_template = np.zeros((events.n_sites, events.n_sites), dtype=int)
    targets = pad_ragged_2d(events.targets, fill_value=events.n_sites, dtype=np.int64)
    print(f"{temperature} K KMC simulation setup done.")
    
    real_n_proc = mp.cpu_count()
    print(f"Available CPU cores: {real_n_proc}.")
    if cfg.n_proc <= 0 or cfg.n_proc > real_n_proc:
        cfg.n_proc = real_n_proc
        print(f"n_proc is reset to maximum available cores: {cfg.n_proc}.")

    queue = mp.Manager().Queue()
    pool = mp.Pool(processes=cfg.n_proc)

    print(f"Using {cfg.n_proc} CPU cores.")

    for _ in range(cfg.repeat_run):
        ini = np.random.randint(0, events.n_sites)
        pool.apply_async(
            kmc_loop,
            (queue, cfg.t_limit, cfg.n_steps, rates.rates, rates.total_rate, 
             targets, events.n_events, events_count_template, ini)
        )

    pool.close()
    pool.join()
    queue.put(None)

    records = []
    while True:
        item = queue.get()
        if item is None:
            break

        t, steps, events_count = item
        dx, dy, dz = events.cal_displacement(events_count)
        records.append([t, steps, dx, dy, dz])

    df = pd.DataFrame(records, columns=["t", "steps", "d_x", "d_y", "d_z"])
    df.to_csv(f"{str(temperature)}.csv", index=False)