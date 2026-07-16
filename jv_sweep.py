from drivers import ADALM1000_Driver, AD3_Driver, Keithley2450_Driver, USMU_Driver
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTRUMENT SELECTION AND OUTPUT PATH  –  KEITHLEY / USMU / AD3 / ADALM1000
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUMENT = "USMU"
OUTPUT_DIR  = Path("output JV")
OUTPUT_DIR.mkdir(exist_ok=True)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JV INITIAL PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCAN_RATES = [0.01, 0.1, 1, 10] # V/s
N_LOOPS = 2
SWEEP_MODE = "rev/fwd"
P_in = 100 # mW/cm²
SAMPLE_AREA = 5 # cm²
V_START = 1.1 if SWEEP_MODE == "rev/fwd" else 0.0
V_STOP = 0.0 if SWEEP_MODE == "rev/fwd" else 1.1

def preconditioning_device():
    """ Check if Voc is stable by CV (<0.1%), save Voc transient fig and then jv is ready to start """
    pass
    return last_voc

def calculate_pv_metrics():
    """ PCE, Voc, Jsc, FF, Rs, Rsh """
    pass



def main():
    print(f"\nConnecting to instrument: {INSTRUMENT}")

    if INSTRUMENT == "USMU":
        driver = USMU_Driver("COM3", 9600, 0.01)
    elif INSTRUMENT == "KEITHLEY":
        driver = Keithley2450_Driver("USB0::0x05E6::0x2450::04387871::0::INSTR")
    elif INSTRUMENT == "ADALM1000":
        driver = ADALM1000_Driver()
    elif INSTRUMENT == "AD3":
        driver = AD3_Driver(
            scope_range_v=5.0,  # 5 V full-scale per analog-input channel
            shunt_ohm=219.6,
        )
    else:
        raise ValueError(
            f"Unknown instrument: {INSTRUMENT!r}. Choose 'USMU', 'KEITHLEY', "
            f"'ADALM1000' or 'AD3'."
        )

    driver.connect()
    voc_now = preconditioning_device() if SWEEP_MODE == "rev/fwd" else None
    
    try:
        for loop in range(N_LOOPS):



        pass

    except Exception as e:
        print(f"Error while doing JV sweeping: {e}")

if __name__ == "__main__":
    main()