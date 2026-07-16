# uSMU Characterization Report
**Date:** 2026-07-16 12:29:18

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.059 s  (= 5 × τ)
- **Load (nominal)**: R = 9.90 kΩ,  C = 102.2 µF,  τ = 1.012 s
- **Compliance current**: 100 mA

## Statistics
| Range | OSR | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | 45.04 | 0.108 | 1.694 | 22.08 | 21.91 |
| 1 | 10 | 5.60 | 0.170 | 1.715 | 177.79 | 178.61 |
| 1 | 25 | 2.27 | 0.185 | 1.721 | 437.79 | 439.23 |
| 1 | 50 | 1.15 | 0.000 | 1.825 | 871.05 | 871.73 |
| 2 | 1 | 45.25 | 0.069 | 1.524 | 23.22 | 21.76 |
| 2 | 10 | 5.61 | 0.000 | 1.641 | 178.10 | 178.28 |
| 2 | 25 | 2.28 | 0.000 | 1.523 | 439.33 | 437.91 |
| 2 | 50 | 1.15 | 0.289 | 1.827 | 870.65 | 871.59 |
| 3 | 1 | 44.61 | 0.067 | 1.520 | 20.84 | 22.08 |
| 3 | 10 | 5.62 | 0.000 | 1.651 | 178.81 | 177.74 |
| 3 | 25 | 2.28 | 0.000 | 1.529 | 438.06 | 438.04 |
| 3 | 50 | 1.15 | 0.289 | 1.819 | 871.78 | 871.36 |
| 4 | 1 | 46.10 | 0.266 | 1.527 | 21.86 | 20.96 |
| 4 | 10 | 5.63 | 0.321 | 1.655 | 178.69 | 178.41 |
| 4 | 25 | 2.28 | 0.181 | 1.724 | 438.03 | 438.09 |
| 4 | 50 | 1.15 | 0.000 | 1.823 | 870.65 | 871.57 |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.