# ADALM1000 Characterization Report
**Date:** 2026-07-16 15:54:15

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.059 s  (= 5 × τ)
- **Load (nominal)**: R = 9.90 kΩ,  C = 102.2 µF,  τ = 1.012 s
- **Compliance current**: 100 mA

## Statistics
| Sample Rate (Sa/s) | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100000 | 3.68 | 0.004 | 2.976 | 264.65 | 267.86 |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.