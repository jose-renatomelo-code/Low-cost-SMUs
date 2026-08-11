import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Force UTF-8 encoding for standard output/error on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure matplotlib for publication-quality scientific plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": "white",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.color": "#e9ecef",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 11,
})

# Directory setup
OUTPUT_BASE = Path("output JV")
SMU_COLORS = {
    "KEITHLEY": "#1f77b4",   # Deep Blue (Reference SMU)
    "ADALM1000": "#ff7f0e",  # Safety Orange (Low-cost SMU)
    "USMU": "#2ca02c"        # Emerald Green (Low-cost uSMU)
}

def gaussian_kde_numpy(vals, x_grid, bw_factor=0.6):
    """
    Computes Gaussian Kernel Density Estimation using numpy only
    to avoid hard external dependencies like scipy.
    """
    n = len(vals)
    sigma = np.std(vals)
    if sigma == 0:
        sigma = 0.01  # Provide small variance if all values are identical
    h = (n ** -0.2) * sigma * bw_factor
    diff = vals[:, np.newaxis] - x_grid[np.newaxis, :]
    kernel_val = np.exp(-0.5 * (diff / h)**2) / (h * np.sqrt(2 * np.pi))
    density = np.sum(kernel_val, axis=0) / n
    return density

def load_data():
    """
    Loads all jv_metrics_per_scanrate.csv files from output JV/ recursively,
    extracts the SMU name from the path, and aggregates them.
    """
    if not OUTPUT_BASE.exists():
        print(f"Error: Output directory {OUTPUT_BASE} not found.")
        sys.exit(1)
        
    files = list(OUTPUT_BASE.rglob("jv_metrics_per_scanrate.csv"))
    if not files:
        print("No 'jv_metrics_per_scanrate.csv' files found in output JV/")
        sys.exit(1)
        
    print(f"Found {len(files)} metrics files.")
    data = []
    for file in files:
        # Relativize path and get the second directory level (the SMU name)
        parts = file.relative_to(OUTPUT_BASE).parts
        if not parts:
            continue
        smu = parts[0].upper()
        
        # Load and append
        df = pd.read_csv(file)
        df['SMU'] = smu
        df['source_file'] = str(file)
        data.append(df)
        
    df_all = pd.concat(data, ignore_index=True)
    df_all['scan_rate_val'] = df_all['scan_rate(V/s)']
    return df_all

def generate_raincloud_plot(df, title, filename_suffix):
    """
    Generates a 2x2 publication-quality grid of raincloud plots for the PV metrics.
    """
    scan_rates = sorted(df['scan_rate_val'].unique())
    smus = ['KEITHLEY', 'USMU', 'ADALM1000']
    
    # Parameters to plot
    params = {
        'Pmax(mW/cm²)': ('Power (mW/cm²)', 'Power (mW/cm²)'),
        'Voc(V)': ('Voc (V)', 'Open Circuit Voltage (V)'),
        'Jsc(mA/cm²)': ('Jsc (mA/cm²)', 'Short Circuit Current (mA/cm²)'),
        'FF(%)': ('FF (%)', 'Fill Factor (%)')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    for ax_idx, (col_name, (short_name, long_name)) in enumerate(params.items()):
        ax = axes[ax_idx]
        
        # Plot styling
        ax.set_title(long_name, fontsize=13, pad=15, fontweight='bold', color='#212529')
        ax.set_yticks(range(len(scan_rates)))
        ax.set_yticklabels([f"{sr} V/s" for sr in scan_rates], fontsize=10)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        for sr_idx, sr in enumerate(scan_rates):
            for smu_idx, smu in enumerate(smus):
                # Filter data for this SMU and scan rate
                sub_df = df[(df['scan_rate_val'] == sr) & (df['SMU'] == smu)].dropna(subset=[col_name])
                vals = sub_df[col_name].values
                
                if len(vals) == 0:
                    continue
                
                # Offset y positions slightly for each SMU to separate them vertically
                # Keithley: +0.2, uSMU: 0.0, ADALM1000: -0.2
                offset = 0.2 - smu_idx * 0.2
                y_pos = sr_idx + offset
                color = SMU_COLORS[smu]
                
                # 1. THE CLOUD (Half-violin using numpy-based KDE)
                if len(vals) >= 2:
                    try:
                        ptp = np.ptp(vals)
                        if ptp == 0:
                            ptp = 0.02
                        x_grid = np.linspace(vals.min() - 0.25 * ptp, vals.max() + 0.25 * ptp, 100)
                        density = gaussian_kde_numpy(vals, x_grid, bw_factor=0.6)
                        
                        # Normalize density height to fit nicely inside the offset gap
                        density = (density / density.max()) * 0.07
                        
                        # Fill cloud
                        ax.fill_between(x_grid, y_pos, y_pos + density, color=color, alpha=0.25, zorder=2)
                        # Cloud edge outline
                        ax.plot(x_grid, y_pos + density, color=color, linewidth=1.0, alpha=0.6, zorder=2)
                    except Exception:
                        pass
                
                # 2. THE BOX (Custom flat styled boxplot showing median and quartiles)
                if len(vals) >= 1:
                    q1 = np.percentile(vals, 25)
                    median = np.percentile(vals, 50)
                    q3 = np.percentile(vals, 75)
                    val_min = vals.min()
                    val_max = vals.max()
                    
                    # Whisker line
                    ax.plot([val_min, val_max], [y_pos - 0.03, y_pos - 0.03], color='#495057', linewidth=1.0, zorder=3)
                    # Whisker end caps
                    ax.plot([val_min, val_min], [y_pos - 0.05, y_pos - 0.01], color='#495057', linewidth=0.8, zorder=3)
                    ax.plot([val_max, val_max], [y_pos - 0.05, y_pos - 0.01], color='#495057', linewidth=0.8, zorder=3)
                    
                    # Box rectangle
                    ax.fill_between([q1, q3], y_pos - 0.06, y_pos - 0.00, facecolor=color, edgecolor='#343a40', linewidth=0.8, alpha=0.7, zorder=4)
                    
                    # Median line (bold contrast)
                    ax.plot([median, median], [y_pos - 0.06, y_pos - 0.00], color='#212529', linewidth=1.5, zorder=5)
                
                # 3. THE RAIN (Jittered individual points plotted below the boxplot)
                jitter = np.random.uniform(-0.015, 0.015, len(vals))
                ax.scatter(vals, y_pos - 0.09 + jitter, color=color, s=15, alpha=0.7, edgecolors='none', zorder=1)

        # Axes cleanup
        ax.set_ylim(-0.5, len(scan_rates) - 0.5)
        ax.invert_yaxis()  # Invert so 0.01 V/s is at the top
        ax.set_xlabel(short_name, fontsize=11, color='#212529')
        
    # Single global legend for all subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SMU_COLORS[smu], edgecolor='#343a40', alpha=0.7, label=smu) 
        for smu in smus
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99, color='#111111')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save the plot
    output_path = OUTPUT_BASE / f"raincloud_comparison_{filename_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved raincloud plot to: {output_path}")

def main():
    print("=== Loading data from CSV files ===")
    df = load_data()
    
    # Clean data (drop rows with NaN metrics or zero PCE which denote failed runs)
    df_clean = df.dropna(subset=['PCE(%)', 'Jsc(mA/cm²)', 'FF(%)'])
    df_clean = df_clean[df_clean['PCE(%)'] > 0.01]
    
    # 1. Generate plot for ALL measured devices (excluding failed/dark sweeps)
    print("Generating plot for all measured devices...")
    generate_raincloud_plot(
        df_clean, 
        "Photovoltaic Parameter Distributions (All Measured Devices)", 
        "all_devices"
    )
    
    # 2. Generate plot for the Standard Cell only (where Voc is between 0.9V and 1.2V)
    # This filters out the high-voltage cells (RKJ01/RKJ02 with Voc > 5V) and the low-voltage silicon reference (Voc ~ 0.57V)
    print("Generating plot for the standard cell only (Voc 0.9V - 1.2V)...")
    df_standard = df_clean[(df_clean['Voc(V)'] >= 0.9) & (df_clean['Voc(V)'] <= 1.2)]
    generate_raincloud_plot(
        df_clean,
        "Photovoltaic Parameter Distributions (Standard Silicon Cell)",
        "standard_cell"
    )
    
    print("=== Processing completed successfully ===")

if __name__ == "__main__":
    main()
