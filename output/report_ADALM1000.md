# ADALM1000 Characterization Report
**Date:** 2026-08-14 11:50:52

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.010 s  (= 5 × τ)
- **Load (nominal)**: R = 9.96 kΩ,  C = 100.6 µF,  τ = 1.002 s
- **Compliance current**: 100 mA

## Statistics
| Sample Rate (Sa/s) | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100000 | 12.83 | 0.007 | 3.642 | 88.41 | 79.23 |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.