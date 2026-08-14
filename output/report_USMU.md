# uSMU Characterization Report
**Date:** 2026-08-14 10:29:18

## Measurement Conditions
- **Stimulus**: Square wave  0.0 V ↔ 2.0 V
- **Half-period**: 5.010 s  (= 5 × τ)
- **Load (nominal)**: R = 9.96 kΩ,  C = 100.6 µF,  τ = 1.002 s
- **Compliance current**: 100 mA

## Statistics
| Range | OSR | Sample Rate (Hz) | V noise (mV rms) | I noise (µA rms) | Rise time (ms) | Fall time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | 43.14 | 0.227 | 1.194 | 23.68 | 21.49 |

## Notes
> [!NOTE]
> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).

> [!NOTE]
> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) threshold crossing, using linear interpolation between samples.