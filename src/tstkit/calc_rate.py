import os
import yaml
import time
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.constants import physical_constants

# Physical constants.
e = physical_constants["elementary charge"][0]
h = physical_constants["Planck constant"][0]
h_in_eV_s = physical_constants["Planck constant in eV s"][0]
kB = physical_constants["Boltzmann constant"][0]
k_B_eV_per_K = physical_constants["Boltzmann constant in eV/K"][0]

def load_input_from_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    barrier_data = data.get('barrier_data', [])
    return barrier_data

def load_freq(f_file):
    """
    Load processed frequency file.
    """
    if not os.path.exists(f_file):
        raise FileNotFoundError(f"Frequency file not found: {f_file}")

    with open(f_file, 'r', encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    f_data = []
    fi_data = []
    
    for phonon in data.get('phonon', []):
        for band in phonon.get('band', []):
            freq = band.get('frequency')
            if freq is None:
                continue
            if freq >= 0:
                f_data.append(freq)
            else:
                fi_data.append(freq)
    
    f_data = np.array(f_data)
    fi_data = np.array(fi_data)*-1  # Turn to positive.
    return f_data, fi_data

def imag_freq_check(saddle_fi, stable_fi):
    """
    Avoid incorrect saddle points with more than one large imaginary frequency.
    If there is no second imaginary frequency, just ignore.
    """
    print("Checking imaginary frequencies ...", end=" ", flush=True)
    if len(saddle_fi) != len(stable_fi) + 1:
        raise ValueError("Saddle points only should have one imaginary frequency.")
    else:
        print("done")

def zero_point_energy(freq_THz):
    """
    Compute total zero-point energy (ZPE) from vibrational frequencies (THz).
    ZPE = 0.5 * h * nu
    """
    freq_Hz = freq_THz * 1e12
    zpe = 0.5 * h_in_eV_s * freq_Hz
    return zpe.sum()

def quantum_correction(f_THz, T):
    """
    Compute quantum harmonic statistical correction factor at temperature T (K).
    sinh(h$/mu/$2kT) / (h$/mu$/2kT)
    """
    f = f_THz * 1e12
    x = 0.5 * h * f / (kB * T)
    return np.sinh(x) / x

def tunneling_correction(T, deltaE0_eV, fi_THz, down_limit):
    """
    Compute tunneling correction factor at temperature T (K).    
    """
    of_warning = False
    deltaE0 = deltaE0_eV * e
    fi = fi_THz * 1e12
    beta = 1 / (T * kB)
    i_energy = h * fi

    theta_up_limit = np.pi * deltaE0 / i_energy

    def integrand(theta):
        return 0.5 * np.exp(beta * i_energy * theta / np.pi) * (np.cosh(theta)**-2)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, abserr = quad(integrand, down_limit, theta_up_limit, epsabs=1e-10, epsrel=1e-10)
        for warn in w:
            if issubclass(warn.category, RuntimeWarning) and "overflow encountered in cosh" in str(warn.message):
                of_warning = True
                
    surf_term = np.exp(beta * deltaE0) / (1 + np.exp(2 * np.pi * deltaE0 / i_energy))
    gamma = result + surf_term

    return gamma, abserr, of_warning

def crossover_temperature(deltaE0_eV, fi_THz):
    """
    Compute crossover temperature Tc (K).
    """
    fi = fi_THz * 1e12
    i_energy = h * fi
    deltaE0 = deltaE0_eV * e
    numerator = i_energy * deltaE0
    denominator = kB * (2 * np.pi * deltaE0 - i_energy * np.log(2))
    return numerator / denominator

def jump_freq(stable_f, saddle_f):
    """
    Calculate raw TST reaction rate constant k0 using frequency ratios.
    """
    log_k0 = np.sum(np.log(stable_f)) - np.sum(np.log(saddle_f))
    return np.exp(log_k0)

def jump_freq_zpe_correction(stable_f, saddle_f, T):
    """
    Calculate quantum hamming correction factor for TST rate constant at temperature T (K).
    """
    stable_corr = quantum_correction(stable_f, T)
    saddle_corr = quantum_correction(saddle_f, T)
    log_corr = np.sum(np.log(stable_corr)) - np.sum(np.log(saddle_corr))
    return np.exp(log_corr)

def Arrhenius(T, k0, deltaE0):
    """
    Calculate the Arrhenius rate constant at temperature T (K).
    """
    return k0 * np.exp(-deltaE0 / k_B_eV_per_K / T)

def calc_rates():
    start_time = time.perf_counter()
    
    barrier_data = load_input_from_yaml("qtst_input.yaml")
    for b in barrier_data:
        name = b.get("name", b.get("id", "unknown"))
        stable_file = b.get("stable_frequency_file")
        saddle_file = b.get("saddle_frequency_file")
        deltaE0 = b.get("adiabatic_barrier_height")
        down_limit = b.get("integral_down_limit", None)
        temps = b.get("temperatures", [])
        
        stable_f_data, stable_fi_data = load_freq(stable_file)
        saddle_f_data, saddle_fi_data = load_freq(saddle_file)
        imag_freq_check(saddle_fi_data, stable_fi_data)
        
        # ZPE and correction.
        zpe_stable = zero_point_energy(stable_f_data)
        zpe_saddle = zero_point_energy(saddle_f_data)
        zpe_corr = zpe_saddle - zpe_stable
        zpe_corr_barrier = deltaE0 + zpe_corr

        # Raw TST rate constant.
        k0 = jump_freq(stable_f_data, saddle_f_data)
        
        # Quantum harmonic correction factors and tunneling corrections.
        saddle_fi = saddle_fi_data[0]
        if down_limit is not None:
            T_corssover = crossover_temperature(zpe_corr_barrier, saddle_fi)
        harmonic_corr_factors = {}
        tunneling_corr_factors = {}
        errors = {}
        warnings_list = {}
        k0s_in_tem = {}
        harmonic_corr_k0s_in_tem = {}
        tunneling_corr_k0s_in_tem = {}
        for T in temps:
            harmonic_corr = jump_freq_zpe_correction(stable_f_data, saddle_f_data, T)
            harmonic_corr_factors[T] = float(harmonic_corr)
            k0s_in_tem[T] = float(Arrhenius(T, k0, deltaE0))
            harmonic_corr_k0s_in_tem[T] = float(Arrhenius(T, k0 * harmonic_corr, deltaE0))
            if down_limit is not None:
                gamma, err, of_warning = tunneling_correction(T, zpe_corr_barrier, saddle_fi, down_limit)
                tunneling_corr_factors[T] = float(gamma)
                errors[T] = float(err)
                warnings_list[T] = of_warning
                tunneling_corr_k0s_in_tem[T] = float(harmonic_corr_k0s_in_tem[T] * gamma)

        # Write results back to barrier data.
        b["zpe_corrected_barrier_height"] = float(zpe_corr_barrier)
        b["zpe_corrected_value"] = float(zpe_corr)
        b["prefactor_of_rate"] = float(k0)
        b["rates_in_temperature"] = k0s_in_tem
        b["quantum_harmonic_correction_factor"] = harmonic_corr_factors
        b["corrected_rates_in_temperature"] = harmonic_corr_k0s_in_tem
        if down_limit is not None:
            b["tunneling_crossover_temperature"] = float(T_corssover) 
            b["tunneling_correction_factor"] = tunneling_corr_factors
            b["integration_errors"] = errors
            b["overflow_warnings"] = warnings_list
            b["both_corrected_rates_in_temperature"] = tunneling_corr_k0s_in_tem

    with open("qtst_rates.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"barrier_data": barrier_data}, f, allow_unicode=True, sort_keys=False)

    end_time = time.perf_counter()
    print(f"\nTime cost: {end_time - start_time:.4f} s")