import numpy as np
import pandas as pd
from numba import njit
import multiprocessing as mp
import unittest
from tstkit.rates import RatesTable
from tstkit.config import load_config
from tstkit.events import Events, pad_ragged_2d
from tstkit.kmc_core import kmc_iteration, run_kmc_at_temperature
from tstkit.process import postprocess_diffusion
cfg = load_config("input.yaml")
ev = Events(structure_file="sites.vasp", path=cfg.site_geometry)
postprocess_diffusion(cfg, fit=True)
print(ev.n_sites)
print(ev.site_symbols)
print(ev.site_symbol_ranges)
print(ev.site_symbol_of)
print(ev.path)

targets, n_events, equal_events = ev.build_events_from_site_geometry()
print(targets)
print(n_events)
print(equal_events)

#for T in cfg.temperatures:
#    targets, n_events, equal_events = ev.build_events_from_site_geometry()
#    rt = RatesTable(n_events, equal_events, ev.site_symbol_ranges, cfg.temperature_data, T)
#    run_kmc_at_temperature(T, cfg, rt, ev)

"""
def run_kmc_at_temperature(temperature, cfg, rates: RatesTable, events: Events, n_proc=None):
    print(f"Running {temperature} K...")

    events.precompute_jump_vectors()
    events_count_template = np.zeros((events.n_sites, events.n_sites), dtype=int)
    
    real_n_proc = mp.cpu_count()
    if n_proc is None or n_proc <= 0 or n_proc > real_n_proc:
        n_proc = real_n_proc
        print(f"n_proc is reset to maximum available cores: {n_proc}")

    for _ in range(cfg.repeat_run):
        ini = np.random.randint(0, events.n_sites)
        t, steps, events_count = kmc_iteration(cfg.t_limit, cfg.n_steps, rates.rates, rates.total_rate, 
             events.targets, events.n_events, events_count_template, ini)
        records = []
        dx, dy, dz = events.cal_displacement(events_count)
        records.append([t, steps, dx, dy, dz])

    df = pd.DataFrame(records, columns=["t", "steps", "d_x", "d_y", "d_z"])
    df.to_csv(f"{int(temperature)}.csv", index=False)

run_kmc_at_temperature(200, cfg, rt, ev, 1)
"""