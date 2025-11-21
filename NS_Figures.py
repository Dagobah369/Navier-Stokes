import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(1729)

# --- Core Function: Spectral Coherence Coefficient C_N ---
def compute_C_stats(s, Ns):
    stats = []
    for N in Ns:
        c_values = []
        stride = max(1, N // 2)
        for i in range(0, len(s) - N, stride):
            window = s[i : i+N]
            num = np.sum(window[:-1]) # Sum of first N-1
            den = np.sum(window)      # Sum of all N
            if den > 0:
                c_values.append(num / den)
        
        c_values = np.array(c_values)
        if len(c_values) > 0:
            mean_c = np.mean(c_values)
            var_c = np.var(c_values)
            stats.append({'N': N, 'mean': mean_c, 'var': var_c, 'count': len(c_values)})
    return pd.DataFrame(stats)

# --- Simulation of Energy Cascades ---

# 1. Navier-Stokes (Regular/Viscous) - Short-range correlations
def generate_ns_gaps(n_gaps):
    phi = -0.36 
    noise = np.random.normal(1, 0.3, n_gaps) 
    gaps = np.zeros(n_gaps); gaps[0] = 1.0
    for t in range(1, n_gaps):
        gaps[t] = 1.0 + phi * (gaps[t-1] - 1.0) + (noise[t] - 1.0)
    gaps = np.maximum(gaps, 0.01)
    gaps = gaps / np.mean(gaps)
    return gaps

# 2. Euler/Singularity (Inviscid/Blow-up) - Long-range correlations
def generate_euler_gaps(n_gaps):
    white = np.random.normal(0, 1, n_gaps)
    freqs = np.fft.rfftfreq(n_gaps)
    alpha = 0.8 
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.power(np.maximum(freqs, 1e-10), alpha/2)
    scale[0] = 0
    long_range = np.fft.irfft(np.fft.rfft(white) * scale, n=n_gaps)
    gaps = np.exp(long_range)
    gaps = gaps / np.mean(gaps)
    return gaps

# --- Main Execution ---
n_gaps = 200000
window_sizes = [5, 10, 20, 40, 80, 160, 320]

s_ns = generate_ns_gaps(n_gaps)
s_euler = generate_euler_gaps(n_gaps)

df_ns = compute_C_stats(s_ns, window_sizes)
df_euler = compute_C_stats(s_euler, window_sizes)

# Plotting code omitted for brevity (see output images)