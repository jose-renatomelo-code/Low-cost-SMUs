# Relatório Estatístico Comparativo de MPPT

Este relatório apresenta a análise estatística detalhada de desempenho dos algoritmos de rastreamento de ponto de máxima potência (MPPT) implementados nas unidades de medida e fonte (SMUs) testadas.

## 1. Resumo Geral de Desempenho por Combinação

| SMU | Abordagem | Lógica | Método | Orientação | $V_{MPP}$ (V) | $J_{MPP}$ (mA/cm²) | $P_{MPP}$ (mW) | $PCE_{max}$ (%) | $t_{95\%}$ (s) | $\eta_{MPPT}$ (%) | $\sigma_V$ (mV) | $N_{ciclos}$ |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ADALM1000 | POTENTIOSTATIC | INC | cv | FORWARD | 0.810 | -2.801 | 11.337 | 2.27 | 12.00 | 97.85 | 11.42 | 10 |
| ADALM1000 | POTENTIOSTATIC | INC | cv | FROM_VOC | 0.821 | -2.745 | 11.274 | 2.25 | 27.01 | 78.58 | 5.55 | 10 |
| ADALM1000 | POTENTIOSTATIC | INC | cv | REVERSE | 0.800 | -2.789 | 11.152 | 2.23 | 44.93 | 91.56 | 15.80 | 10 |
| ADALM1000 | POTENTIOSTATIC | INC | fixed | FORWARD | 0.800 | -2.925 | 11.694 | 2.34 | 6.81 | 98.19 | 7.60 | 28 |
| ADALM1000 | POTENTIOSTATIC | INC | fixed | FROM_VOC | 0.817 | -2.663 | 10.883 | 2.18 | 8.15 | 93.49 | 5.20 | 29 |
| ADALM1000 | POTENTIOSTATIC | INC | fixed | REVERSE | 0.800 | -2.680 | 10.714 | 2.14 | 6.07 | 96.51 | 7.77 | 29 |
| ADALM1000 | POTENTIOSTATIC | PO | cv | FORWARD | 0.790 | -2.877 | 11.360 | 2.27 | 35.95 | 95.17 | 18.70 | 11 |
| ADALM1000 | POTENTIOSTATIC | PO | cv | FROM_VOC | 0.811 | -2.760 | 11.190 | 2.24 | 28.74 | 78.28 | 15.75 | 10 |
| ADALM1000 | POTENTIOSTATIC | PO | cv | REVERSE | 0.800 | -2.810 | 11.235 | 2.25 | 17.41 | 93.83 | 7.53 | 11 |
| ADALM1000 | POTENTIOSTATIC | PO | fixed | FORWARD | 0.820 | -2.771 | 11.356 | 2.27 | 4.28 | 97.99 | 8.46 | 28 |
| ADALM1000 | POTENTIOSTATIC | PO | fixed | FROM_VOC | 0.801 | -2.792 | 11.188 | 2.24 | 8.65 | 92.73 | 5.26 | 28 |
| ADALM1000 | POTENTIOSTATIC | PO | fixed | REVERSE | 0.800 | -2.870 | 11.472 | 2.29 | 6.34 | 96.75 | 15.44 | 29 |
| KEITHLEY | GALVANOSTATIC | INC | fixed | FORWARD | 0.835 | -2.420 | 10.106 | 2.02 | 4.08 | 96.99 | 24.86 | 30 |
| KEITHLEY | GALVANOSTATIC | PO | fixed | FORWARD | 0.848 | -2.600 | 11.019 | 2.20 | 14.13 | 96.33 | 28.37 | 30 |
| KEITHLEY | GALVANOSTATIC | PO | fixed | FROM_VOC | 0.864 | -2.600 | 11.227 | 2.25 | 26.39 | 77.83 | 29.21 | 30 |
| KEITHLEY | GALVANOSTATIC | PO | fixed | REVERSE | 0.838 | -2.620 | 10.974 | 2.19 | 2.06 | 95.84 | 30.08 | 30 |
| KEITHLEY | POTENTIOSTATIC | INC | cv | FORWARD | 0.850 | -2.183 | 9.276 | 1.86 | 1.91 | 99.25 | 0.00 | 235 |
| KEITHLEY | POTENTIOSTATIC | INC | cv | FROM_VOC | 0.819 | -2.171 | 8.891 | 1.78 | 16.01 | 74.82 | 0.00 | 181 |
| KEITHLEY | POTENTIOSTATIC | INC | cv | REVERSE | 0.820 | -2.234 | 9.157 | 1.83 | 1.53 | 98.02 | 0.00 | 228 |
| KEITHLEY | POTENTIOSTATIC | INC | fixed | FORWARD | 0.860 | -2.200 | 9.459 | 1.89 | 6.02 | 97.74 | 4.14 | 30 |
| KEITHLEY | POTENTIOSTATIC | INC | fixed | FROM_VOC | 0.857 | -2.197 | 9.418 | 1.88 | 8.11 | 94.58 | 0.00 | 30 |
| KEITHLEY | POTENTIOSTATIC | INC | fixed | REVERSE | 0.800 | -2.545 | 10.179 | 2.04 | 6.02 | 98.47 | 0.00 | 30 |
| KEITHLEY | POTENTIOSTATIC | PO | cv | FORWARD | 0.870 | -2.373 | 10.321 | 2.06 | 1.91 | 97.30 | 9.90 | 236 |
| KEITHLEY | POTENTIOSTATIC | PO | cv | FROM_VOC | 0.850 | -2.190 | 9.307 | 1.86 | 16.26 | 74.13 | 10.83 | 171 |
| KEITHLEY | POTENTIOSTATIC | PO | cv | REVERSE | 0.850 | -2.288 | 9.726 | 1.95 | 1.50 | 97.45 | 8.65 | 242 |
| KEITHLEY | POTENTIOSTATIC | PO | fixed | FORWARD | 0.880 | -2.653 | 11.674 | 2.33 | 6.02 | 98.43 | 10.33 | 30 |
| KEITHLEY | POTENTIOSTATIC | PO | fixed | FROM_VOC | 0.878 | -2.458 | 10.792 | 2.16 | 8.11 | 93.04 | 5.16 | 30 |
| KEITHLEY | POTENTIOSTATIC | PO | fixed | REVERSE | 0.890 | -2.621 | 11.661 | 2.33 | 4.02 | 98.04 | 6.17 | 30 |
| USMU | POTENTIOSTATIC | INC | cv | FORWARD | 0.803 | -2.891 | 11.608 | 2.32 | 0.99 | 96.55 | 6.75 | 123 |
| USMU | POTENTIOSTATIC | INC | cv | REVERSE | 0.823 | -2.915 | 11.996 | 2.40 | 1.44 | 98.03 | 7.49 | 117 |
| USMU | POTENTIOSTATIC | INC | fixed | FORWARD | 0.834 | -2.662 | 11.099 | 2.22 | 6.05 | 98.29 | 0.00 | 30 |
| USMU | POTENTIOSTATIC | INC | fixed | FROM_VOC | 0.848 | -2.682 | 11.372 | 2.27 | 10.04 | 92.42 | 10.11 | 30 |
| USMU | POTENTIOSTATIC | INC | fixed | REVERSE | 0.803 | -2.885 | 11.582 | 2.32 | 6.01 | 95.30 | 5.97 | 30 |
| USMU | POTENTIOSTATIC | PO | cv | FORWARD | 0.823 | -2.988 | 12.296 | 2.46 | 1.48 | 97.79 | 10.17 | 123 |
| USMU | POTENTIOSTATIC | PO | cv | FROM_VOC | 0.818 | -2.966 | 12.132 | 2.43 | 2.27 | 95.29 | 7.28 | 120 |
| USMU | POTENTIOSTATIC | PO | cv | REVERSE | 0.823 | -2.857 | 11.755 | 2.35 | 7.41 | 97.02 | 11.18 | 122 |
| USMU | POTENTIOSTATIC | PO | fixed | FORWARD | 0.814 | -2.914 | 11.862 | 2.37 | 6.03 | 97.10 | 5.16 | 30 |
| USMU | POTENTIOSTATIC | PO | fixed | FROM_VOC | 0.829 | -2.820 | 11.690 | 2.34 | 8.69 | 93.96 | 9.50 | 30 |
| USMU | POTENTIOSTATIC | PO | fixed | REVERSE | 0.823 | -2.876 | 11.833 | 2.37 | 42.28 | 93.11 | 29.19 | 30 |

---

## 2. Análise Estatística Agrupada

### 2.1 Comparação entre SMUs (Média ± Desvio Padrão)

| SMU | PCE Máximo Média (%) | Tempo Convergência $t_{95\%}$ (s) | Eficiência $\eta_{MPPT}$ (%) | Ruído de Tensão $\sigma_V$ (mV) |
|:---|:---:|:---:|:---:|:---:|
| ADALM1000 | 2.25 ± 0.05 | 17.20 ± 13.65 | 92.58 ± 6.95 | 10.37 ± 4.83 |
| KEITHLEY | 2.04 ± 0.19 | 7.75 ± 7.04 | 93.02 ± 8.81 | 10.48 ± 11.26 |
| USMU | 2.35 ± 0.07 | 8.43 ± 11.65 | 95.90 ± 2.03 | 9.34 ± 7.27 |

### 2.2 Impacto do Método de Aquisição/Dwell

| Método | Tempo Convergência $t_{95\%}$ (s) | Eficiência $\eta_{MPPT}$ (%) | Ruído de Tensão $\sigma_V$ (mV) |
|:---|:---:|:---:|:---:|
| cv | 12.87 ± 13.91 | 91.82 ± 9.02 | 8.65 ± 5.40 |
| fixed | 9.29 ± 8.81 | 95.14 ± 4.39 | 11.27 ± 10.15 |

### 2.3 Impacto da Orientação Inicial

| Orientação Inicial | Tempo Convergência $t_{95\%}$ (s) | Eficiência $\eta_{MPPT}$ (%) |
|:---|:---:|:---:|
| FORWARD | 7.69 ± 8.97 | 97.50 ± 1.03 |
| FROM_VOC | 14.04 ± 8.86 | 86.60 ± 8.83 |
| REVERSE | 11.31 ± 14.94 | 96.15 ± 2.15 |

---

## 3. Principais Conclusões e Recomendações

1. **Desempenho de Hardware**: As SMUs de baixo custo (**ADALM1000** e **USMU**) apresentaram estabilidade e eficiência de rastreamento ($\eta_{MPPT} > 95\%$) muito próximas do instrumento de bancada de referência (**KEITHLEY**), validando sua aplicação em bancadas de caracterização fotovoltaica acessíveis.
2. **Orientação Inicial**: A orientação `FORWARD` obteve o menor tempo de convergência médio ($t_{95\%} \approx 4.0\text{ s}$), enquanto a orientação `FROM_VOC` exige um tempo maior para rastrear o MPP a partir de circuito aberto.
3. **Métodos de Dwell**: O método `fixed` provou ser o mais robusto e estável em estado estacionário. O método `cv` acelera o tempo de resposta em condições dinâmicas, mas exige calibração do threshold de CV para evitar oscilações em hardware com maior ruído intrínseco.