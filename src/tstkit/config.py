import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict
from collections import OrderedDict

@dataclass
class SiteGeometry:
    distances: np.ndarray

@dataclass
class TempRateInfo:
    rates: OrderedDict[str, np.ndarray]

@dataclass
class SimulationConfig:
    t_limit: float
    n_steps: int
    repeat_run: int
    n_proc: int
    lattice_orientation: np.ndarray | None
    site_geometry: Dict[str, SiteGeometry]
    temperature_data: Dict[int, TempRateInfo]

    @property
    def temperatures(self):
        return sorted(self.temperature_data.keys())
    
def load_config(filename) -> SimulationConfig:
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    p = data["simulation_parameters"]

    # ---------- site geometry ----------
    site_geometry: Dict[str, SiteGeometry] = {}
    for site_label, v in p["site_geometry"].items():
        site_geometry[site_label] = SiteGeometry(
            distances=np.asarray(v["distances"], dtype=np.float64)
        )

    # ---------- temperature dependent rates ----------
    temperature_data: Dict[int, TempRateInfo] = {}
    temp_data_raw = p["temperature_data"]

    for T, site_dict in temp_data_raw.items():
        rates = OrderedDict()

        for site_label, v in site_dict.items():
            rates[site_label] = (
                np.asarray(v["rates"], dtype=np.float64) * 1e12
            )

        temperature_data[int(T)] = TempRateInfo(rates=rates)

    return SimulationConfig(
        t_limit=float(p["t_limit"]),
        n_steps=int(p["n_steps"]),
        repeat_run=int(p["repeat_run"]),
        n_proc=int(p.get("n_proc", 1)),
        lattice_orientation=(
            np.array(p["lattice_orientation"], dtype=np.int32)
            if p.get("lattice_orientation") is not None
            else None
        ),
        site_geometry=site_geometry,
        temperature_data=temperature_data,
    )