try:
    from usmu_py.smu import USMU, USMUSerialError
except ImportError:
    class USMUSerialError(Exception):
        pass
    USMU = None

try:
    from pysmu import Session, Mode
except ImportError:
    Session = None
    Mode = None

import pyvisa
import dwfpy as dwf

import time
import numpy as np

# ADALM1000
N_SETTLE_CALLS         = 3      # discard calls after constant()

SHUNT_RESISTANCE_OHM = 219.6

V_MAX_V         = 7.0         # square-wave high level  (V)
V_MIN_V         = 0.0        # square-wave low level   (V)
CURRENT_LIMIT_A = 0.1         # compliance / source current limit (A) = 100 mA
# Fixed current-MEASUREMENT range. Auto-range was selecting a µA range and
# corrupting the source/measure pair (source V != measured V). 0.01 A = 10 mA
# fits this cell (~1-2 mA); raise to 0.1 if your cell exceeds ~10 mA/cm².
CURRENT_MEAS_RANGE_A = 0.01   # 10 mA fixed sense range

# Number of scope samples averaged per measurement point.
N_MEASURE_SAMPLES = 1000

# Scope sample rate (Hz)
SCOPE_SAMPLE_RATE = 1e5
CURRENT_RANGES = [1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0]

class ADALM1000_Driver:
    """ Simple ADALM1000 (pysmu) driver handling SMU logic.

    The sweep parameter is the **sample rate**, set via `session.configure()`
    (see ADALM1000_SR). The number of samples fetched per `get_samples()` call
    is kept fixed at 1000; varying the sample rate changes the per-call duration
    and thus the effective acquisition sample rate (analogous to NPLC/OSR).
    """
    def __init__(self, device_index: int = 0):
        self.device_index        = device_index
        self._session            = None
        self._device             = None
        self._chan_a             = None
        self._chan_b             = None
        self._i_offset           = 0.0      # measured DC current offset (A)
        self._n_samples          = 500     # samples per get_samples() call (fixed)
        self._sample_rate        = 100000   # Sa/s (sweep param)
        self._integration_time_s = 0.05    # averaging window per measurement

    def connect(self, curr_sour=False):
        if Session is None:
            raise ImportError("pysmu package not found.")
        print(f"Opening pysmu Session (device_index={self.device_index})")
        self._session = Session()
        assert len(self._session.devices) > self.device_index, (
            f"ADALM1000 device_index={self.device_index} not found. "
            f"Devices connected: {len(self._session.devices)}"
        )
        self._device = self._session.devices[self.device_index]
        self._device.ignore_dataflow = True
        self._chan_a = self._device.channels["A"]
        # On the ADALM1000 each channel is a source-measure unit:
        # - SVMI mode: Source Voltage, Measure Current (para MPPT potenziostático)
        # - SIMV mode: Source Current, Measure Voltage (para MPPT galvânico)
        if curr_sour:
            self._chan_a.mode = Mode.SIMV
            self._chan_a.constant(0.0)  # 0 A como ponto inicial
        else:
            self._chan_a.mode = Mode.SVMI
            self._chan_a.constant(0.0)  # 0 V como ponto inicial
        self._device.channels["B"].mode = Mode.HI_Z
        for _ in range(N_SETTLE_CALLS):
            self._device.get_samples(1000)

        print(f"ADALM1000 connected (device {self.device_index}, curr_sour={curr_sour})")

    def disconnect(self):
        if self._chan_a is not None:
            try:
                # Return to SVMI 0V before ending session
                self._chan_a.mode = Mode.SVMI
                self._chan_a.constant(0.0)
                for _ in range(N_SETTLE_CALLS):
                    self._device.get_samples(1000)
            except Exception:
                pass
        if self._session is not None:
            try:
                self._session.end()
            except Exception:
                pass
            self._session = None
        self._device  = None
        self._chan_a  = None
        self._chan_b  = None
        print("ADALM1000 disconnected")

    # -- sweep configuration --------------------------------------------------
    def configure_integration(self, sample_rate: int):
        """Set the acquisition sample rate (sweep param) via session.configure()."""
        self._sample_rate = int(sample_rate)
        self._session.configure(self._sample_rate)
        print(f"ADALM1000: sample_rate={self._sample_rate} Sa/s "
              f"(~{self._n_samples/self._sample_rate*1e3:.2f} ms per "
              f"get_samples({self._n_samples}))")

    def configure(self, integration_time_s: float = 0.05, **kwargs):
        self._integration_time_s = integration_time_s
        print(f"ADALM1000: integration_time_s={integration_time_s}s")

    def _settle(self):
        for _ in range(N_SETTLE_CALLS):
            self._device.get_samples(self._n_samples)

    def _measure_raw_i(self) -> float:
        """Raw current reading from channel A (CHA reports its own current in SVMI)."""
        t0 = time.perf_counter()
        all_i = []
        samples = self._device.get_samples(self._n_samples)
        # sample = [chA_sample, chB_sample]; chA[1] = current through channel A
        all_i.extend([s[0][1] for s in samples])
        return float(np.nanmean(all_i))

    def set_voltage_and_measure(self, voltage: float) -> tuple[float, float]:
        self._chan_a.mode = Mode.SVMI
        self._chan_a.constant(voltage)
        self._settle()
        return self.measure()

    def set_current_and_measure(self, current: float):
        self._chan_a.mode = Mode.SIMV
        self._chan_a.constant(current)
        self._settle()
        return self.measure()

    def measure(self) -> tuple[float, float]:
        """Average several get_samples() calls for `integration_time_s`.

        Voltage and current are both read from channel A (the M1K SMU channel);
        the DC offset measured at connect() is subtracted from the current.

        Sign convention: the ADALM1000 in SVMI reports the current as *leaving*
        the channel (positive when sourcing). For a solar cell the generated
        photocurrent flows *into* the channel, so we negate it to match the
        USMU/Keithley convention (negative current for a generator).
        """
        t0     = time.perf_counter()
        all_v, all_i = [], []
        # Guarantee at least one acquisition even for a zero-length window.
        samples = self._device.get_samples(self._n_samples)
        all_v.extend([s[0][0] for s in samples])
        all_i.extend([s[0][1] for s in samples])
        v_avg = float(np.nanmean(all_v))
        i_avg = float(np.nanmean(all_i))
        return v_avg, i_avg

class USMU_Driver:
    """Thin wrapper around the usmu_py USMU object."""

    def __init__(self, port: str, baudrate: int, command_delay: float):
        if USMU is None:
            raise ImportError("usmu_py package not found.")
        self._dev = USMU(port=port, baudrate=baudrate, command_delay=command_delay)

    # -- lifecycle ------------------------------------------------------------
    def connect(self, curr_sour=False):
        idn = self._dev.read_idn()
        print(f"  Connected to uSMU: {idn}")
        self._dev.enable_output()
        if curr_sour == True:
            self._dev.set_current_limit(0)  # uSMU uses mA
        else:
            self._dev.set_current_limit(CURRENT_LIMIT_A * 1e3)   # uSMU uses mA

    def disconnect(self):
        self._dev.set_voltage(0)
        self._dev.disable_output()
        self._dev.close()

    # -- sweep configuration --------------------------------------------------
    def configure_range(self, current_range: int):
        """Lock the uSMU hardware current range (1–4)."""
        self._dev.lock_current_range(current_range)

    def configure_integration(self, osr: int):
        """Set oversample rate."""
        self._dev.set_oversample_rate(osr)

    # -- measurement ----------------------------------------------------------
    def set_voltage_and_measure(self, voltage_v: float) -> tuple[float, float]:
        """Apply voltage, return (measured_V, measured_A)."""
        v, i = self._dev.set_voltage_and_measure(voltage_v)
        return float(v), float(i)

    def set_current_and_measure(self, current_A: float) -> tuple[float, float]:
        """Apply current setpoint (in A), return (measured_V, measured_A)."""
        if hasattr(self._dev, "set_current_and_measure"):
            v, i = self._dev.set_current_and_measure(current_A * 1e3)
            return float(v), float(i)
        elif hasattr(self._dev, "set_current"):
            self._dev.set_current(current_A * 1e3)
            v, i = self._dev.measure() if hasattr(self._dev, "measure") else (0.0, current_A)
            return float(v), float(i)
        else:
            raise NotImplementedError("uSMU does not support galvanostatic sourcing mode.")

class Keithley2450_Driver:
    """SCPI driver for Keithley 2450 SourceMeter via PyVISA."""

    def __init__(self, visa_address: str):
        rm = pyvisa.ResourceManager()
        self._dev = rm.open_resource(visa_address)
        self._dev.timeout = 5000   # ms

    # -- lifecycle ------------------------------------------------------------
    def connect(self, curr_sour=False):
        self._dev.write("*RST")
        self._dev.write("*CLS")

        if curr_sour:
            # Source current, measure voltage
            self._dev.write("SOUR:FUNC CURR")
            self._dev.write("SENS:FUNC VOLT")

        else:
            # Source voltage, measure current
            self._dev.write("SOUR:FUNC VOLT")
            self._dev.write("SENS:FUNC \"CURR\"")

        self._dev.write(f"SOUR:VOLT:RANG {max(abs(V_MAX_V), abs(V_MIN_V))}")
        self._dev.write(f"SOUR:VOLT:ILIM {CURRENT_LIMIT_A}")
        self._dev.write("SOUR:CURR:VLIM 10.0")

        self._dev.write("SENS:CURR:RANG:AUTO ON")
        self._dev.write("OUTP ON")

        idn = self._dev.query("*IDN?").strip()
        print(f"  Connected to Keithley: {idn}")

    def set_source_mode(self, curr_sour: bool):
        """Switch the Keithley between current-source and voltage-source mode
        WITHOUT sending *RST (which would wipe integration/NPLC settings)."""
        if curr_sour:
            self._dev.write("SOUR:FUNC CURR")
            self._dev.write("SENS:FUNC VOLT")
        else:
            self._dev.write("SOUR:FUNC VOLT")
            self._dev.write("SENS:FUNC \"CURR\"")
        self._dev.write("OUTP ON")

    def disconnect(self):
        self._dev.write("SOUR:VOLT 0")
        self._dev.write("OUTP OFF")
        self._dev.close()

    # -- sweep configuration --------------------------------------------------
    def configure_range(self, _current_range):
        """Keithley uses auto-range; parameter ignored."""
        pass

    def configure_integration(self, nplc: float):
        """Set NPLC (Number of Power Line Cycles) for current measurement."""
        self._dev.write(f"SENS:CURR:NPLC {nplc}")

    # -- measurement ----------------------------------------------------------
    def set_voltage_and_measure(self, voltage_v: float) -> tuple[float, float]:
        """Apply voltage setpoint, trigger measurement, return (V, A)."""
        self._dev.write(f"SOUR:VOLT {voltage_v:.6f}")
        # Read voltage and current in one query
        raw_i = self._dev.query("MEAS:CURR?").strip()
        return voltage_v, float(raw_i)

    def set_current_and_measure(self, current_A: float) -> tuple[float, float]:
        """Apply voltage setpoint, trigger measurement, return (V, A)."""
        self._dev.write(f"SOUR:CURR {current_A}")
        # Read voltage and current in one query
        raw_V = self._dev.query("MEAS:VOLT?").strip()
        return float(raw_V), current_A

class AD3_Driver:

    def __init__(self, scope_range_v: float = 5.0, shunt_ohm: float = SHUNT_RESISTANCE_OHM):
        """
        Parameters
        ----------
        scope_range_v : float
            Input range of the analog input channels in volts (default +-5V).
        shunt_ohm : float
            Shunt resistor value for current calculation (ohms).
        """
        self.scope_range_v = scope_range_v
        self.shunt_ohm     = shunt_ohm
        self._device       = None
        self._scope_rate_hz = int(SCOPE_SAMPLE_RATE)
        self._settle_s     = 0.01   # settle time between set_voltage and measure

    def connect(self):
        print("Opening Analog Discovery 3")
        self._device = dwf.Device()
        self._device.open()
        # Configure analog output channel (W1) as DC voltage source.
        # dwfpy sets the carrier node via setup(function='dc', offset=...).
        w1 = self._device.analog_output["ch1"]
        w1.setup(function="dc", offset=0.0, enabled=True, start=True)
        # Configure analog input (scope) ranges for voltage + shunt sensing
        self._device.analog_input.frequency = self._scope_rate_hz
        self._device.analog_input["ch1"].range = self.scope_range_v
        self._device.analog_input["ch2"].range = self.scope_range_v
        print(f"AD3 connected. Shunt = {self.shunt_ohm} Ohm")

    def disconnect(self):
        if self._device is not None:
            self._device.analog_output["ch1"].setup(function="dc", offset=0.0, start=True)
            self._device.close()
            self._device = None
            print("AD3 disconnected")

    def configure(self, **kwargs):
        # AD3 has no software-configurable current limit.
        # Voltage range and shunt are set in __init__.
        print("AD3: configuration applied at connect time")

    # -- sweep configuration (compatible interface) --------------------------
    def configure_range(self, _current_range):
        """AD3 frontend range is fixed at connect time; parameter ignored."""
        pass

    def configure_integration(self, scope_rate_hz: int):
        """Set the analog-input (scope) sample rate used during measurement."""
        self._scope_rate_hz = int(scope_rate_hz)
        self._device.analog_input.frequency = self._scope_rate_hz
        print(f"AD3: scope_rate={self._scope_rate_hz} Hz")

    # -- measurement (compatible interface) -----------------------------------
    def set_voltage(self, voltage: float):
        self._device.analog_output["ch1"].setup(function="dc", offset=voltage, start=True)
        print(f"AD3: W1 set to {voltage:.4f} V")

    def measure(self) -> tuple[float, float]:
        # Acquire a single burst on the scope, then read both channels.
        ai = self._device.analog_input
        ai.single(
            sample_rate=self._scope_rate_hz,
            buffer_size=N_MEASURE_SAMPLES,
            start=True,
        )
        samples_ch1 = ai["ch1"].get_data()   # voltage on W1 (sense)
        samples_ch2 = ai["ch2"].get_data()   # shunt voltage -> current
        v_avg = float(np.mean(samples_ch1))
        i_avg = float(np.mean(samples_ch2) / self.shunt_ohm)
        print(f"AD3: {v_avg:.4f} V, {i_avg:.4e} A")
        return v_avg, i_avg

    def set_voltage_and_measure(self, voltage: float) -> tuple[float, float]:
        """Apply voltage setpoint, wait a brief settle, return (V, A).

        AD3 sources via analog-out and senses via the scope (ch1 = voltage,
        ch2 = shunt voltage), so voltage and measurement are separate hardware
        blocks. We translate the combined-call interface used by main.py into
        the underlying set_voltage() + measure().
        """
        self.set_voltage(voltage)
        # Brief settle so the step settles before the scope capture
        time.sleep(self._settle_s)
        return self.measure()
