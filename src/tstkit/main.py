import argparse
from .config import load_config
from .events import Events
from .rates import RatesTable
from .kmc_core import run_kmc_at_temperature
from .process import postprocess_diffusion
from .calc_rate import calc_rates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",help="Run KMC simulation")
    parser.add_argument("--postprocess", action="store_true", help="Post-process diffusion data")
    parser.add_argument("--fit", action="store_true", help="Enable post-processing fit and plotting")
    parser.add_argument("--neb", action="store_true", help="Use NEB barriers for rate calculations")
    parser.add_argument("--read_frequency", action="store_true", help="Read frequency data from existing files")
    parser.add_argument("--calc_rates", action="store_true", help="Calculate rates from frequency data")
    
    args = parser.parse_args()
    
    if args.run:
        print("Running KMC simulation...")
        cfg = load_config(filename="input.yaml")
        ev = Events(structure_file="sites.vasp", path=cfg.site_geometry)
        for T in cfg.temperatures:
            targets, n_events, equal_events = ev.build_events_from_site_geometry()
            rt = RatesTable(n_events, equal_events, ev.site_symbol_ranges, cfg.temperature_data, T)
            run_kmc_at_temperature(T, cfg, rt, ev)
    if args.postprocess:
        cfg = load_config(filename="input.yaml")
        print("Post-processing diffusion data...")
        postprocess_diffusion(cfg, fit=args.fit)
    if args.calc_rates:
        print("Calculating rates from frequency data (phonopy yaml file)...")
        calc_rates()
    
if __name__ == "__main__":
    main()