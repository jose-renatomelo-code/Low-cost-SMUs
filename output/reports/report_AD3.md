# Analog Discovery 3 Characterization Report
**Date:** 2026-07-16 15:25:21

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.171 s  (= 5 × τ)
- **Load (nominal)**: R = 10.12 kΩ,  C = 102.2 µF,  τ = 1.034 s
- **Compliance current**: 100 mA

## Statistics
| Scope Rate (Sa/s) | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100000 | 29.83 | 0.324 | 1.448 | 31.24 | 35.90 |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.