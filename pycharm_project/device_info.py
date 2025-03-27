import subprocess
import psutil
import platform

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU

    nvml_available = True
    nvmlInit()
except ImportError:
    nvml_available = False

try:
    import wmi  # Windows-specific
    wmi_available = True
except ImportError:
    wmi_available = False

try:
    import clr  # the pythonnet module.

    clr.AddReference(r'ohm/OpenHardwareMonitorLib')

    ohm_available = True
    from OpenHardwareMonitor.Hardware import Computer

    c = Computer()
    c.CPUEnabled = True  # get the Info about CPU
    c.GPUEnabled = True  # get the Info about GPU
    c.Open()
except ImportError:
    ohm_available = False


def get_device_info():
    system = platform.system()

    if system == 'Windows':
        return Windows()

    elif system == 'Linux':
        return Linux()

    raise Exception('Unsupported OS')

class DefaultDevice:
    @staticmethod
    def get_cpu_usage() -> float:
        return psutil.cpu_percent(interval=0.5)

    @staticmethod
    def get_mem_usage():
        return psutil.virtual_memory().percent

    @staticmethod
    def get_swap_usage():
        """Returns swap memory usage in percentage."""
        return psutil.swap_memory().percent

class Windows(DefaultDevice):
    @staticmethod
    def get_cpu_temperature():
        for a in range(0, len(c.Hardware[0].Sensors)):
            if "/temperature" in str(c.Hardware[0].Sensors[a].Identifier):
                temp = c.Hardware[0].Sensors[a].get_Value()
                c.Hardware[0].Update()
                return temp

        return -17

    @staticmethod
    def get_gpu_usage():
        for sensor in c.Hardware[1].Sensors:
            if "/load" in str(sensor.Identifier) and "gpu" in str(sensor.Identifier).lower():
                usage = sensor.get_Value()
                c.Hardware[1].Update()
                return usage

        return -17

    @staticmethod
    def get_gpu_temperature():
        for a in range(0, len(c.Hardware[1].Sensors)):
            if "/temperature" in str(c.Hardware[1].Sensors[a].Identifier):
                temp = c.Hardware[1].Sensors[a].get_Value()
                c.Hardware[0].Update()
                return temp

        return -17


class Linux(DefaultDevice):
    @staticmethod
    def get_cpu_temperature():
        """Returns CPU temperature in Celsius."""

        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                return max(temp.current for temp in temps["coretemp"])

            if "k10temp" in temps:  # My laptop is special
                return max(temp.current for temp in temps["k10temp"])

        except AttributeError:
            return -17

        return -17

    @staticmethod
    def get_gpu_usage():
        if nvml_available:  # If nvidia
            handle = nvmlDeviceGetHandleByIndex(0)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu

        try:
            amd_usage = subprocess.check_output("rocm-smi --showuse --json", shell=True).decode()
            return float(eval(amd_usage)["card0"]["GPU use (%)"])
        except Exception as e:
            pass

        return -17

    @staticmethod
    def get_gpu_temperature():
        if nvml_available:
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)  # First GPU
                return nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            except Exception as e:
                raise Exception(f"Error getting NVIDIA GPU temp: {e}")

        try:
            amd_temp = subprocess.check_output("rocm-smi --showtemp --json", shell=True).decode()
            return float(eval(amd_temp)["card0"]["Temperature (Sensor edge) (C)"])
        except Exception:
            pass

        return -17

