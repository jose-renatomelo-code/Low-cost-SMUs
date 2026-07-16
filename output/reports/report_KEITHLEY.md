# Keithley 2450 Characterization Report
**Date:** 2026-07-16 11:35:55

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.059 s  (= 5 × τ)
- **Load (nominal)**: R = 9.90 kΩ,  C = 102.2 µF,  τ = 1.012 s
- **Compliance current**: 100 mA

## Statistics
| NPLC | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 193.27 | 0.000 | 2.068 | 4.55 | 4.68 |
| 0.1 | 77.87 | 0.000 | 1.789 | 12.45 | 12.11 |
| 1 | 10.39 | 0.000 | 1.640 | 83.66 | 83.87 |
| 10 | 0.47 | 0.000 | 5.617 | N/A | N/A |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.