import time
from drivers import Keithley2450_Driver

driver = Keithley2450_Driver("USB0::0x05E6::0x2450::04387871::0::INSTR")
print("Connecting...")
driver.connect(curr_sour=True)
print("Connected")
try:
    v, i = driver.set_current_and_measure(0.0)
    print(f"V: {v}, I: {i}")
except Exception as e:
    print(f"Error: {e}")
finally:
    driver.disconnect()
    print("Disconnected")