#!/usr/bin/env python3
#--------------------------------------------------
# This script is used to extract the frequencies
# from the OUTCAR files in the calculation folders
# and save them into yaml files.
#--------------------------------------------------
import os
import re
import time
import yaml
import numpy as np
from ase.io import read

def get_calculated_point():
    """
    Identify folders matching 'stable', 'stable_*', 'saddle', or 'saddle_*'.
    These are considered calculation directories.
    """
    folders = [entry.name for entry in os.scandir('.') if entry.is_dir()]
    # This ^ means the start of the string, $ means the end of the string
    # and the (_.*) means it can be followed by an underscore and any characters, or nothing.
    # and ? means the preceding element is optional (0 or 1 times).
    saddle_list = [f for f in folders if re.match(r"^saddle(_.*)?$", f)]
    stable_list = [f for f in folders if re.match(r"^stable(_.*)?$", f)]
    return stable_list, saddle_list

def check_outcar_qpoints_exists(folder):
    """
    Check if OUTCAR file exists in the folder.
    """
    check_bool = True
    outcar_path = os.path.join(folder, "OUTCAR")
    qpoints_path = os.path.join(folder, "qpoints.yaml")
    # Check if the OUTCAR file exists.
    if not os.path.isfile(outcar_path):
        if os.path.isfile(qpoints_path):
            print(f"Warning: OUTCAR not found in {folder}, but qpoints.yaml exists. Skipping.")
            check_bool = False
        else:
            raise FileNotFoundError(f"OUTCAR not found in {folder}.")
    return check_bool

def get_frequency(folder, max_lines=100000):
    """
    Read frequency data from the OUTCAR file in the specified folder.
    """
    f_list, fi_list = [], []
    path = os.path.join(folder, "OUTCAR")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            block_size = 1024
            blocks, lines_found = [], 0
            while file_size > 0 and lines_found < max_lines:
                file_size = max(0, file_size - block_size)
                f.seek(file_size)
                block = f.read(block_size)
                blocks.append(block)
                lines_found = sum(block.count('\n') for block in blocks)
            tail = ''.join(reversed(blocks)).splitlines()[-max_lines:]
    except Exception:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            tail = f.readlines()

    for line in tail:
        if "THz" in line and "f" in line:
            match_real = re.search(r"f\s+=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            match_img = re.search(r"f/i=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match_real:
                f_list.append(float(match_real.group(1)))
            if match_img:
                fi_list.append(float(match_img.group(1)))
    # The imaginary frequencies will be set to minus.
    return np.array(f_list), np.array(fi_list)*-1

def get_poscar_info(folder):
    """
    Use ASE to read POSCAR, return natom and reciprocal lattice.
    """
    poscar_path = os.path.join(folder, "POSCAR")
    atoms = read(poscar_path, format="vasp")
    natom = len(atoms)
    reciprocal_lattice = atoms.cell.reciprocal()
    return natom, reciprocal_lattice.tolist()

def save_phonopy_yaml(name, f_all, natom, reciprocal_lattice):
    """
    Save the frequencies into phonopy yaml format.
    Only real frequencies will be written; imaginary can be added if needed.
    """
    data = {
        "nqpoint": 1,
        "natom": natom,
        "reciprocal_lattice": reciprocal_lattice,
        "phonon": [
            {
                "q-position": [0.0, 0.0, 0.0],
                "band": [
                    {"frequency": float("{: .10f}".format(f))} for f in f_all
                ]
            }
        ]
    }
    with open(f"{name}.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(data, fout, allow_unicode=True, sort_keys=False)

def main():
    start_time = time.perf_counter()
    stable_dirs, saddle_dirs = get_calculated_point()
    for folder in stable_dirs + saddle_dirs:
        check_bool = check_outcar_qpoints_exists(folder)
        if check_bool:
            # The end=" " is to avoid the newline after print 
            # and flush=True is to force the output to appear immediately.
            # flush=True can avoid the buffer.
            print(f"Start saving {folder}.yaml ...", end=" ", flush=True)
            real_f, imag_f = get_frequency(folder)
            # Merge real and imaginary frequencies.
            all_freq = np.concatenate((imag_f, real_f))
            # sort frequencies.
            all_freq_sorted = np.sort(all_freq)
            natom, reciprocal_lattice = get_poscar_info(folder)
            save_phonopy_yaml(folder, all_freq_sorted, natom, reciprocal_lattice)
            print(f"done.")
    end_time = time.perf_counter()
    print(f"Time cost: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    main()