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

V_MAX_V         = 2.0         # square-wave high level  (V)
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

    Supports connecting via `device_index` or device `serial` number, as well
    as sharing an existing `pysmu.Session` across multiple driver instances.
    """
    def __init__(self, device_index: int = 0, serial: str = None, session = None):
        self.device_index        = device_index
        self.serial              = serial
        self.device_id           = f"M1K_dev{device_index}"
        self._session            = session
        self._owns_session       = (session is None)
        self._device             = None
        self._chan_a             = None
        self._chan_b             = None
        self._i_offset           = 0.0      # measured DC current offset channel A (A)
        self._i_offset_b         = 0.0      # measured DC current offset channel B (A)
        self._n_samples          = 1000     # samples per get_samples() call (fixed)
        self._sample_rate        = 100000   # Sa/s (sweep param)
        self._integration_time_s = 0.03    # averaging window per measurement
        self._last_v_a           = None
        self._last_v_b           = None

    @staticmethod
    def list_devices() -> list[dict]:
        """Discover and list all connected ADALM1000 devices."""
        if Session is None:
            print("pysmu package not found.")
            return []
        try:
            sess = Session()
            info = []
            for idx, dev in enumerate(sess.devices):
                dev_serial = getattr(dev, "serial", f"dev_{idx}")
                info.append({"index": idx, "serial": dev_serial, "label": f"M1K_dev{idx}_{dev_serial[:6]}"})
            sess.end()
            return info
        except Exception as e:
            print(f"Error listing ADALM1000 devices: {e}")
            return []

    def connect(self, curr_sour=False):
        if Session is None:
            raise ImportError("pysmu package not found.")

        if self._owns_session and self._session is None:
            print(f"Opening pysmu Session for device_index={self.device_index}")
            self._session = Session()

        assert len(self._session.devices) > 0, "No ADALM1000 devices found connected."

        # Locate target device by serial or device_index
        target_dev = None
        if self.serial:
            for d in self._session.devices:
                if getattr(d, "serial", "") == self.serial:
                    target_dev = d
                    break
            assert target_dev is not None, f"ADALM1000 device with serial '{self.serial}' not found."
        else:
            assert len(self._session.devices) > self.device_index, (
                f"ADALM1000 device_index={self.device_index} not found. "
                f"Devices connected: {len(self._session.devices)}"
            )
            target_dev = self._session.devices[self.device_index]

        self._device = target_dev
        self._device.ignore_dataflow = True
        self._chan_a = self._device.channels["A"]
        self._chan_b = self._device.channels["B"]

        dev_serial = getattr(self._device, "serial", f"dev_{self.device_index}")
        self.device_id = f"M1K_dev{self.device_index}_{dev_serial[:6]}"

        if curr_sour:
            if self._chan_a.mode != Mode.SIMV:
                self._chan_a.mode = Mode.SIMV
            if self._chan_b.mode != Mode.SIMV:
                self._chan_b.mode = Mode.SIMV
            self._chan_a.constant(0.0)
            self._chan_b.constant(0.0)
        else:
            if self._chan_a.mode != Mode.SVMI:
                self._chan_a.mode = Mode.SVMI
            if self._chan_b.mode != Mode.HI_Z:
                self._chan_b.mode = Mode.HI_Z
            self._chan_a.constant(0.0)
            self._chan_b.constant(0.0)
        self._last_v_a = None
        self._last_v_b = None

        # Discard initial settling samples before capturing DC current offset
        self._settle()

        samples = self._device.get_samples(1000)
        i_offset_a = [s[0][1] for s in samples]
        self._i_offset   = float(np.nanmean(i_offset_a))
        self._i_offset = 0

        print(f"ADALM1000 [{self.device_id}] connected (index={self.device_index}, curr_sour={curr_sour}),"
              f" i_offset_A={self._i_offset:.6e} A")

    def disconnect(self):
        if self._chan_a is not None:
            try:
                self._chan_a.mode = Mode.SVMI
                self._chan_a.constant(0.0)
                for _ in range(N_SETTLE_CALLS):
                    self._device.get_samples(1000)
            except Exception:
                pass
        if self._owns_session and self._session is not None:
            try:
                self._session.end()
            except Exception:
                pass
            self._session = None
        self._device  = None
        self._chan_a  = None
        self._chan_b  = None
        self._last_v_a = None
        self._last_v_b = None
        print(f"ADALM1000 [{self.device_id}] disconnected")

    # -- sweep configuration --------------------------------------------------
    def configure_integration(self, sample_rate: int):
        """Set the acquisition sample rate (sweep param) via session.configure()."""
        self._sample_rate = int(sample_rate)
        if self._session is not None:
            self._session.configure(self._sample_rate)
        print(f"ADALM1000 [{self.device_id}]: sample_rate={self._sample_rate} Sa/s")

    def configure(self, integration_time_s: float = 0.05, **kwargs):
        self._integration_time_s = integration_time_s
        print(f"ADALM1000 [{self.device_id}]: integration_time_s={integration_time_s}s")

    def _settle(self):
        for _ in range(N_SETTLE_CALLS):
            self._device.get_samples(self._n_samples)

    def blink_led(self, on_off_time, n_blinks):
        for _ in range(n_blinks):
            time.sleep(on_off_time)
            self._device.set_led(0b010)
            time.sleep(on_off_time)
            self._device.set_led(0b001)


    def _measure_raw_i(self) -> float:
        """Raw current reading from channel A (CHA reports its own current in SVMI)."""
        t0 = time.perf_counter()
        all_i = []
        samples = self._device.get_samples(self._n_samples)
        all_i.extend([s[0][1] for s in samples])
        return float(np.nanmean(all_i))

    def set_voltage_and_measure(self, voltage: float) -> tuple[float, float]:
        if self._chan_a.mode != Mode.SVMI:
            self._chan_a.mode = Mode.SVMI
        if self._last_v_a != voltage:
            self._chan_a.constant(voltage)
            self._last_v_a = voltage
        return self.measure()

    def set_current_and_measure(self, current: float):
        if self._chan_a.mode != Mode.SIMV:
            self._chan_a.mode = Mode.SIMV
        if self._last_v_a != current:
            self._chan_a.constant(current)
            self._last_v_a = current
        return self.measure()

    def measure(self) -> tuple[float, float]:
        """Average several get_samples() calls for `integration_time_s`."""
        samples = self._device.get_samples(self._n_samples)
        all_v = [s[0][0] for s in samples]
        all_i = [s[0][1] for s in samples]
        v_avg = float(np.nanmean(all_v))
        i_avg = float(np.nanmean(all_i)) - self._i_offset
        return v_avg, i_avg

    def measure_both(self) -> tuple[float, float, float, float]:
        """Measure both channels A and B simultaneously."""
        samples = self._device.get_samples(self._n_samples)
        all_va = np.fromiter((s[0][0] for s in samples), dtype=float)
        all_ia = np.fromiter((s[0][1] for s in samples), dtype=float)
        all_vb = np.fromiter((s[1][0] for s in samples), dtype=float)
        all_ib = np.fromiter((s[1][1] for s in samples), dtype=float)

        v_a = float(np.nanmean(all_va))
        i_a = float(np.nanmean(all_ia)) - self._i_offset
        v_b = float(np.nanmean(all_vb))
        i_b = float(np.nanmean(all_ib)) - self._i_offset_b
        return v_a, i_a, v_b, i_b

    def set_voltage_both_and_measure(self, voltage: float) -> tuple[float, float, float, float]:
        """Apply the same voltage on both channels A and B (SVMI) and measure."""
        return self.set_channel_voltages_and_measure(voltage, voltage)

    def set_channel_voltages_and_measure(self, v_a: float, v_b: float) -> tuple[float, float, float, float]:
        """Apply independent voltages on channels A and B (SVMI) and measure."""
        if self._chan_a.mode != Mode.SVMI:
            self._chan_a.mode = Mode.SVMI
        if self._chan_b.mode != Mode.SVMI:
            self._chan_b.mode = Mode.SVMI

        if self._last_v_a != v_a:
            self._chan_a.constant(v_a)
            self._last_v_a = v_a
        if self._last_v_b != v_b:
            self._chan_b.constant(v_b)
            self._last_v_b = v_b

        return self.measure_both()

class MultiADALM1000_Driver:
    """ Driver managing multiple connected ADALM1000 devices simultaneously
    under a shared pysmu Session.

    Parameters
    ----------
    device_indices : list[int], optional
        List of device indices to connect (e.g. [0, 1]).
    serials : list[str], optional
        List of device serial numbers to connect. If neither is specified,
        connects ALL connected ADALM1000 devices found on the system.
    """
    def __init__(self, device_indices: list[int] = None, serials: list[str] = None):
        self.device_indices = device_indices
        self.serials = serials
        self._session = None
        self.devices: dict[str, ADALM1000_Driver] = {}  # key: device_id, val: ADALM1000_Driver

    def connect(self, curr_sour=False):
        if Session is None:
            raise ImportError("pysmu package not found.")

        print("Opening pysmu Multi-Device Session...")
        self._session = Session()
        available = self._session.devices
        assert len(available) > 0, "No ADALM1000 devices found."
        print(f"Discovered {len(available)} connected ADALM1000 device(s).")

        # Determine which devices to open
        targets = []  # list of (index, serial)
        if self.serials:
            for s in self.serials:
                idx = next((i for i, dev in enumerate(available) if getattr(dev, "serial", "") == s), None)
                if idx is not None:
                    targets.append((idx, s))
                else:
                    print(f"Warning: device with serial '{s}' not found.")
        elif self.device_indices:
            for idx in self.device_indices:
                if idx < len(available):
                    dev_serial = getattr(available[idx], "serial", f"dev_{idx}")
                    targets.append((idx, dev_serial))
                else:
                    print(f"Warning: device_index={idx} not found (only {len(available)} available).")
        else:
            # Default: connect all available devices
            for idx, dev in enumerate(available):
                dev_serial = getattr(dev, "serial", f"dev_{idx}")
                targets.append((idx, dev_serial))

        assert len(targets) > 0, "No valid target ADALM1000 devices to connect."

        for idx, ser in targets:
            drv = ADALM1000_Driver(device_index=idx, serial=ser, session=self._session)
            drv.connect(curr_sour=curr_sour)
            self.devices[drv.device_id] = drv

        print(f"MultiADALM1000: connected {len(self.devices)} device(s): {list(self.devices.keys())}")

    def configure_integration(self, sample_rate: int):
        """Set sample rate across all devices via the shared pysmu Session."""
        if self._session is not None:
            self._session.configure(int(sample_rate))
        for drv in self.devices.values():
            drv._sample_rate = int(sample_rate)

    def set_voltage_all_and_measure(self, voltage: float) -> dict[str, tuple[float, float, float, float]]:
        """Set voltage setpoint on both channels for all connected devices and measure."""
        v_dict = {dev_id: (voltage, voltage) for dev_id in self.devices.keys()}
        return self.set_channel_voltages_all_and_measure(v_dict)

    def set_channel_voltages_all_and_measure(self, v_dict: dict) -> dict[str, tuple[float, float, float, float]]:
        """Set independent voltages per device/channel and measure.

        v_dict: dict mapping dev_id -> (v_a, v_b)
        returns dict mapping dev_id -> (v_a_meas, i_a_meas, v_b_meas, i_b_meas)
        """
        for dev_id, drv in self.devices.items():
            v_a, v_b = v_dict.get(dev_id, (0.0, 0.0))
            if drv._chan_a.mode != Mode.SVMI:
                drv._chan_a.mode = Mode.SVMI
            if drv._chan_b.mode != Mode.SVMI:
                drv._chan_b.mode = Mode.SVMI

            if drv._last_v_a != v_a:
                drv._chan_a.constant(v_a)
                drv._last_v_a = v_a
            if drv._last_v_b != v_b:
                drv._chan_b.constant(v_b)
                drv._last_v_b = v_b

        results = {}
        for dev_id, drv in self.devices.items():
            results[dev_id] = drv.measure_both()
        return results

    def set_voltage_channel_a_all_and_measure(self, v_dict: dict) -> dict[str, tuple[float, float]]:
        """Set voltage on Channel A for each connected device and measure Channel A (V_A, I_A).

        v_dict: dict mapping dev_id -> v_a
        returns dict mapping dev_id -> (v_a_meas, i_a_meas)
        """
        for dev_id, drv in self.devices.items():
            v_a = v_dict.get(dev_id, 0.0)
            if drv._chan_a.mode != Mode.SVMI:
                drv._chan_a.mode = Mode.SVMI
            if drv._last_v_a != v_a:
                drv._chan_a.constant(v_a)
                drv._last_v_a = v_a

        results = {}
        for dev_id, drv in self.devices.items():
            results[dev_id] = drv.measure()
        return results

    def disconnect(self):
        for drv in self.devices.values():
            drv.disconnect()
        if self._session is not None:
            try:
                self._session.end()
            except Exception:
                pass
            self._session = None
        self.devices.clear()
        print("MultiADALM1000 disconnected")

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
