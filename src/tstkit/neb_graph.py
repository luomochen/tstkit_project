#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#--------------------------------------------------------
# This scripts is used to plot the free energy difference
# and generate a new flod to store the saddle point for
# frequency calculation. It can be also used to merge the
# path. 
#--------------------------------------------------------
import os
import re
import numpy as np
import pandas as pd
from subprocess import getstatusoutput
from ase.io import vasp
from scipy import interpolate
from scipy.constants import physical_constants
from matplotlib import pyplot as plt

def image_distance(directs1, directs2, latt_vec_matrix):
    """ Calculate the distance between each image using pbc.
        The distance was defined as $\sum_{i=1}^n(x^\prime_n-x_n)$.
    
    Args:
        position1 (list[list[]]): high dimension coordiante of image 1.
        position2 (_type_): high dimension coordiante of image 2.

    Returns:
        float: distance.
    """
    delta_dirs = directs1 - directs2
    size = delta_dirs.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if delta_dirs[i, j] <= -0.5:
                delta_dirs[i, j] += 1.0
            elif delta_dirs[i, j] > 0.5:
                delta_dirs[i, j] -= 1.0
    differences = np.linalg.norm(delta_dirs@latt_vec_matrix)
    return differences

def neb_data_process():
    """Process the outcome.
    """
    # List all the files in the directory.
    # Use the regular expersion to match the filename
    # and find the image created in the path.
    free_energy_list = []
    dist_list = []
    files = os.listdir()
    image_number = 0
    for file in files:
        if re.match(r"\d", file):
            image_number = image_number + 1

    for i in range(image_number):
        # Use ase to read the OUTCAR.
        image = vasp.read_vasp_out(f"{i:02d}/OUTCAR")
        latt_vec_matrix = image.cell
        free_energy = image.get_total_energy()
        directs = image.get_scaled_positions()
        if i == 0:
            dist = 0
            latt_vec_matrix_p = latt_vec_matrix
        else:
            dist = image_distance(directs, directsp, latt_vec_matrix_p)
            if np.linalg.norm(latt_vec_matrix_p-latt_vec_matrix) > 1E-6:
                raise ValueError(f"Image{i} lattice vector matrix has changed!!!")
        directsp = directs
        free_energy_list.append(free_energy)
        dist_list.append(dist)
        # Compare the initial state and final state 
        # to find the lower one as the energy baseline.
    if free_energy_list[0] > free_energy_list[-1]:
        free_energy_diff_list = [free_energy - free_energy_list[-1] 
                                for free_energy in free_energy_list]
    else:
        free_energy_diff_list = [free_energy - free_energy_list[0] 
                                for free_energy in free_energy_list]
    # When calculating reaction coordinates, 
    # you need to calculate the distance 
    # between the previous image and the next image.
    # and then add them together.
    for i in range(len(dist_list)):
        if i == 0:
            dist_list[i] = dist_list[i]
        else:
            dist_list[i] = dist_list[i] + dist_list[i-1]   
    data = pd.DataFrame({"Reaction coordinate": dist_list, 
                        "Energy difference": free_energy_diff_list})
    data.to_csv("neboutcome.csv", index=False)
    print(data)
    return dist_list, free_energy_diff_list

def creat_saddle_point(free_energy_diff_list):
    """Determine the saddle point file for further calculation.
    """
    barrier = max(free_energy_diff_list)
    saddle_point = free_energy_diff_list.index(barrier)
    os.makedirs('saddle', exist_ok=True)
    status, output = getstatusoutput(f"cp {saddle_point:02d}/CONTCAR saddle/POSCAR")
    if status == 0:
        print(f'\nSaddle point is in image {saddle_point}.')
        print(f'The energy barrier is {round(barrier, 3)} eV.')
        print('Saddle file is generated!')

def plot_reaction_path_graph(dist_list, free_energy_diff_list):
    """Plot the barrier-reaction cooridinate graph.
    """
    # Interpolate to get smooth curve.
    model = interpolate.interp1d(dist_list, 
                                 free_energy_diff_list, 
                                 kind="quadratic")
    xs = np.linspace(0, dist_list[-1], 500)
    ys = model(xs)
    # Plot the free energy - reaction coordianate diagram.
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    max_value_index = free_energy_diff_list.index(max(free_energy_diff_list))
    ax.set_title(f"Barrier = {round(free_energy_diff_list[max_value_index], 3)} eV", 
                 fontsize=25, color='black')
    ax.set_xlabel(r"Reaction coordiante ($\mathrm{\AA}$)", fontsize=20)
    ax.set_ylabel(r"$\mathrm{\Delta}$H (eV)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.scatter(dist_list, free_energy_diff_list, edgecolors='red', 
               facecolors='white', s=400, linewidths=3, zorder=2)
    ax.plot(xs, ys, c="black", linewidth=3, zorder=1)
    ax.grid(True)
    
    fig.tight_layout()
    fig.savefig("neboutcome.png")
    
    plt.close()
    
def main():
    dist_list, free_energy_diff_list = neb_data_process()
    creat_saddle_point(free_energy_diff_list)
    plot_reaction_path_graph(dist_list, free_energy_diff_list)

if __name__ == "__main__":
    main()