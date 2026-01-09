import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from scipy.optimize import curve_fit
from scipy.constants import physical_constants
from scipy.stats import normaltest

KB_eV = physical_constants['Boltzmann constant in eV/K'][0]

def load_displacement_data(csv_file) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads displacement data from a CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing displacement data.
    
    Returns
    -------
    t : (N,) np.ndarray
        Total time of each trajectory
    d_matrix : (N, 3) np.ndarray
    """
    df = pd.read_csv(csv_file)
    t = pd.to_numeric(df['t']).to_numpy()
    d_matrix = df[['d_x', 'd_y', 'd_z']].to_numpy()
    return t, d_matrix

def calculate_total_diffusion_coefficient(t, d_matrix) -> dict:
    """
    Calculate diffusion coefficient and statistical properties.

    Parameters
    ----------
    t : (N,) np.ndarray
        Total time of each trajectory
    d_matrix : (N, 3) np.ndarray
        Total displacement of each trajectory (Angstrom)

    Returns
    -------
    result : dict
        {
            "D_mean": float,
            "D_std": float,
            "D_se": float,
            "CI_95": (float, float),
            "D_3d": np.ndarray,
            "D_list": np.ndarray,
            "normaltest_p": float
        }
    """
    # Diffusion coefficient
    squared = np.square(d_matrix) * 1e-20 # \AA^2 -> m^2

    if np.any(t <= 0):
        raise ValueError("Time array contains non-positive values.")

    D_3d_list = squared / (2 * t[:, np.newaxis]) # (N, 3)
    D_3d_mean = np.mean(D_3d_list, axis=0)

    D_list = np.mean(D_3d_list, axis=1) # (N,)
    D_mean = np.mean(D_list)

    diff_coeff = {
        "D_mean": D_mean,
        "D_3d_mean": D_3d_mean,
        "D_list": D_list,
        "D_3d_list": D_3d_list,
    }

    return diff_coeff

#def error_estimation(D_list) -> dict:
#    """
#    Calculates error estimation for diffusion coefficient.
#    
#    Parameters
#    ----------
#    diff_coeff : dict
#        Output from calculate_total_diffusion_coefficient function.
#    Returns
#    -------
#    error_dict : dict
#        {
#            "D_std": float,
#            "D_se": float,
#            "CI_95": (float, float)
#            "normaltest_p": float
#        }
#    """
#    D_mean = np.mean(D_list)
#    D_std = np.std(D_list, ddof=1)
#    D_se  = D_std / np.sqrt(len(D_list))
#
#    CI_95 = (
#        D_mean - 1.96 * D_se,
#        D_mean + 1.96 * D_se
#    )

    # Normality test
    stat, p_value = normaltest(D_list)
    return {
        "D_std": D_std,
        "D_se": D_se,
        "CI_95": CI_95,
        "normaltest_p": p_value
    }

#def project_displacement(displacements, crystal_orientation, vasp_file):
#    """Calculates the projection of displacements onto a crystal orientation.
#    """
#    atoms = read(vasp_file)
#    orientation_vector = np.dot(crystal_orientation, atoms.cell)
#    unit_vector = orientation_vector / np.linalg.norm(orientation_vector)
#    return np.dot(unit_vector, displacements)

#def calculate_projected_diffusion_coefficient(t, d_matrix, orientation, vasp_file):
#    """Calculates the diffusion coefficient projected onto a crystal orientation."""
#    projection = project_displacement(d_matrix, orientation, vasp_file)
#    D = np.sum(np.square(projection) * 1e-20) / (2 * np.sum(t))
#    return D

def linear_func(x, a, b):
    return a * x + b

def r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fit_diffusion_data(x, y):
    popt, _ = curve_fit(linear_func, x, y)
    slope, intercept = popt
    barrier = -slope * KB_eV * 1000 * np.log(10)
    D0 = 10 ** intercept
    R2 = r_squared(y, linear_func(x, *popt))
    return slope, intercept, barrier, D0, R2

def plot_fit(x, y, slope, intercept, barrier, R2, output_file):
    x_fit = np.linspace(min(x) - 0.1, max(x) + 0.1, 500)
    y_fit = linear_func(x_fit, slope, intercept)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.scatter(x, y, edgecolors='red', facecolors='white', s=100, linewidths=2, label='Data')
    ax.plot(x_fit, y_fit, 'k-', linewidth=3, label='Fit')
    ax.set_xlabel(r"1000/T (K$^{-1}$)", fontsize=20)
    ax.set_ylabel("log10(D)", fontsize=20)
    ax.set_title(f"Barrier = {barrier:.3f} eV", fontsize=20)
    ax.text(x_fit.mean(), y_fit.mean(), f"$R^2$ = {R2:.3f}", fontsize=18, color='blue')
    ax.grid(True)
    ax.legend()
    fig.savefig(output_file)
    plt.close(fig)

def plot_D_vs_T(temperatures, D_log10, slope=None, intercept=None, R2=None, output_file="D_vs_T.png"):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.scatter(temperatures, D_log10, edgecolors='black', facecolors='cyan', s=100, label='Data')
    if slope is not None and intercept is not None:
        x_fit = np.linspace(min(temperatures) - 0.1, max(temperatures) + 0.1, 500)
        y_fit = linear_func(x_fit, slope, intercept)
        ax.plot(x_fit, y_fit, 'r-', label='Fit')
        ax.text(x_fit.mean(), y_fit.mean(), f"$R^2$ = {R2:.3f}", fontsize=18, color='blue')
    ax.set_xlabel(r"1000/(Temperature(K))", fontsize=20)
    ax.set_ylabel(r"log10(D($\mathrm{m^2/s}$))", fontsize=20)
    ax.grid(True)
    ax.legend()
    fig.savefig(output_file)
    plt.close(fig)

def save_fit_data(filename, slope, intercept, barrier, D0, R2):
    fit_data = {
        "fit_equation": {
            "slope": round(float(slope), 5),
            "intercept": round(float(intercept), 5),
            "equation": f"y = {slope:.5f}x + {intercept:.5f}"
        },
        "r_squared": round(float(R2), 5),
        "diffusion_barrier_eV": round(float(barrier), 5),
        "prefactor_D0_m2_per_s": float(f"{D0:.5e}")
    }
    with open(filename, "w") as f:
        yaml.safe_dump(fit_data, f, sort_keys=False)

def save_D_vs_T(temperatures, D_log10, csv_file, yaml_file):
    df = pd.DataFrame({"Temperature(K)": temperatures, "log10(D)": D_log10})
    df.to_csv(csv_file, index=False)
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(df.to_dict(orient='list'), f, sort_keys=False)

def postprocess_diffusion(cfg, fit=False):
    D_all = []
    for T in cfg.temperatures:
        t, d_matrix = load_displacement_data(f"{T}.csv")
        D_dict = calculate_total_diffusion_coefficient(t, d_matrix)
        D_all.append(np.log10(D_dict["D_mean"]))
    
    D_all = np.array(D_all)
    T_array = np.array(cfg.temperatures)
    x_data = 1000 / T_array
    y_data = D_all

    save_D_vs_T(T_array.tolist(), y_data.tolist(), "D_vs_T.csv", "D_vs_T.yaml")
    plot_D_vs_T(x_data, y_data, output_file="D_vs_T.png")

    if fit:
        slope, intercept, barrier, D0, R2 = fit_diffusion_data(x_data, y_data)
        #plot_fit(x_data, y_data, slope, intercept, barrier, R2, output_file="linear_fit.png")
        plot_D_vs_T(x_data, y_data, slope=slope, intercept=intercept, R2=R2, output_file="D_vs_T_fitted.png")
        save_fit_data("fitted_data.yaml", slope, intercept, barrier, D0, R2)

    #if cfg.lattice_orientation is not None:
    #    D_proj = []
    #    for T in cfg.temperatures:
    #        t, d_matrix = load_displacement_data(f"{T}.csv")
    #        D = calculate_projected_diffusion_coefficient(t, d_matrix, cfg.lattice_orientation, 'sites.vasp')
    #        D_proj.append(np.log10(D))
    #    D_proj = np.array(D_proj)

    #    save_D_vs_T(T_array.tolist(), D_proj.tolist(), "D_vs_T_projected.csv", "D_vs_T_projected.yaml")
    #    plot_D_vs_T(T_array, D_proj, output_file="D_vs_T_projected.png")

    #    if fit:
    #        slope, intercept, barrier, D0, R2 = fit_diffusion_data(1000 / T_array, D_proj)
    #        plot_fit(1000 / T_array, D_proj, slope, intercept, barrier, R2, output_file="linear_fit_projected.png")
    #        plot_D_vs_T(T_array, D_proj, slope=slope, intercept=intercept, output_file="D_vs_T_projected_fitted.png")
    #        save_fit_data("fitted_data_projected.yaml", slope, intercept, barrier, D0, R2)
