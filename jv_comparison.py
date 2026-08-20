import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

SMU_MARKERS = {
    "KEITHLEY": "o",
    "ADALM1000": "^",
    "USMU": "s"
}

def load_data():
    """
    Loads jv_metrics_per_scanrate.csv files from output JV/ recursively.
    Only keeps files that belong to a REFERENCE device subfolder
    (i.e. path pattern: <SMU>/REFERENCE/jv_metrics_per_scanrate.csv).
    Extracts the SMU name from parts[0] and the device from parts[1].
    """
    if not OUTPUT_BASE.exists():
        print(f"Error: Output directory {OUTPUT_BASE} not found.")
        sys.exit(1)

    files = list(OUTPUT_BASE.rglob("jv_metrics_per_scanrate.csv"))
    if not files:
        print("No 'jv_metrics_per_scanrate.csv' files found in output JV/")
        sys.exit(1)

    print(f"Found {len(files)} metrics files total.")
    data = []
    for file in files:
        parts = file.relative_to(OUTPUT_BASE).parts
        # Expect at least <SMU>/<DEVICE>/jv_metrics_per_scanrate.csv
        if len(parts) < 2:
            continue
        smu    = parts[0].upper()
        device = parts[1].upper()

        # Only keep REFERENCE device data
        if device != "REFERENCE":
            continue

        df = pd.read_csv(file)
        df['SMU']         = smu
        df['device']      = device
        df['source_file'] = str(file)
        data.append(df)

    if not data:
        print("No REFERENCE data found in output JV/. Aborting.")
        sys.exit(1)

    print(f"Loaded {len(data)} REFERENCE file(s): "
          + ", ".join(str(Path(d['source_file'].iloc[0]).relative_to(OUTPUT_BASE)) for d in data))
    df_all = pd.concat(data, ignore_index=True)
    df_all['scan_rate_val'] = df_all['scan_rate(V/s)']
    return df_all

def generate_datapoints_plot(df, title, filename_suffix):
    """
    Generates a 2x2 publication-quality grid of strip/datapoint plots for the PV metrics.
    Replaces raincloud plots with clean individual data points since sample size per group is small.
    """
    if df.empty:
        print(f"Warning: Empty DataFrame provided for plot '{filename_suffix}'. Skipping.")
        return

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
    
    # Set seed for reproducible jitter
    np.random.seed(42)
    
    for ax_idx, (col_name, (short_name, long_name)) in enumerate(params.items()):
        ax = axes[ax_idx]
        
        # Plot styling
        ax.set_title(long_name, fontsize=13, pad=15, fontweight='bold', color='#212529')
        ax.set_yticks(range(len(scan_rates)))
        ax.set_yticklabels([f"{sr} V/s" for sr in scan_rates], fontsize=10)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)
        
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
                marker = SMU_MARKERS[smu]
                
                # Jitter individual points slightly along Y axis for readability
                jitter = np.random.uniform(-0.04, 0.04, len(vals))
                
                # Plot datapoints
                ax.scatter(
                    vals, 
                    y_pos + jitter, 
                    color=color, 
                    marker=marker,
                    s=50, 
                    alpha=0.85, 
                    edgecolors='#212529',
                    linewidths=0.8,
                    zorder=3
                )
                
                # Draw mean line marker for visual reference
                mean_val = np.mean(vals)
                ax.plot(
                    [mean_val, mean_val], 
                    [y_pos - 0.06, y_pos + 0.06], 
                    color=color, 
                    linewidth=2.0, 
                    alpha=0.9,
                    zorder=4
                )

        # Axes cleanup
        ax.set_ylim(-0.5, len(scan_rates) - 0.5)
        ax.invert_yaxis()  # Invert so 0.01 V/s is at the top
        ax.set_xlabel(short_name, fontsize=11, color='#212529')
        
    # Single global legend for all subplots
    legend_elements = [
        Line2D([0], [0], marker=SMU_MARKERS[smu], color='w', label=smu,
               markerfacecolor=SMU_COLORS[smu], markeredgecolor='#212529', markersize=9)
        for smu in smus
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99, color='#111111')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save the plot
    output_path = OUTPUT_BASE / f"datapoints_comparison_{filename_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved datapoints plot to: {output_path}")
    
    # Also save as raincloud_comparison_{filename_suffix}.png for backward compatibility
    legacy_path = OUTPUT_BASE / f"raincloud_comparison_{filename_suffix}.png"
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    print(f"Updated legacy file: {legacy_path}")

# Maintain function alias for backwards compatibility
generate_raincloud_plot = generate_datapoints_plot

def main():
    print("=== Loading data from CSV files ===")
    df = load_data()
    
    # Clean data (drop rows with NaN metrics or zero PCE which denote failed runs)
    df_clean = df.dropna(subset=['PCE(%)', 'Jsc(mA/cm²)', 'FF(%)'])
    df_clean = df_clean[df_clean['PCE(%)'] > 0.01]
    
    # 1. Generate plot for ALL measured devices (excluding failed/dark sweeps)
    print("Generating plot for all measured devices...")
    generate_datapoints_plot(
        df_clean, 
        "Photovoltaic Parameter Distributions (All Measured Devices)", 
        "all_devices"
    )
    
    # 2. Generate plot for the Standard Cell only (where Voc is between 0.9V and 1.2V)
    print("Generating plot for the standard cell only (Voc 0.9V - 1.2V)...")
    df_standard = df_clean[(df_clean['Voc(V)'] >= 0.9) & (df_clean['Voc(V)'] <= 1.2)]
    generate_datapoints_plot(
        df_standard,
        "Photovoltaic Parameter Distributions (Standard Silicon Cell)",
        "standard_cell"
    )
    
    print("=== Processing completed successfully ===")

if __name__ == "__main__":
    main()
