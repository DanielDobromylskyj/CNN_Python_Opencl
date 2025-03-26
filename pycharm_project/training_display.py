import pygame
import psutil
import math
import subprocess
import threading
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

def get_gpu_usage():
    """Returns GPU utilization percentage. Supports NVIDIA and AMD GPUs."""
    if nvml_available:  # If nvidia
        handle = nvmlDeviceGetHandleByIndex(0)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu

    # Try AMD (Linux)
    try:
        amd_usage = subprocess.check_output("rocm-smi --showuse --json", shell=True).decode()
        return float(eval(amd_usage)["card0"]["GPU use (%)"])
    except Exception as e:
        raise Exception("Unsupported GPU:", e)


def get_gpu_temperature():
    if nvml_available:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)  # First GPU
            return nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        except Exception as e:
            raise Exception(f"Error getting NVIDIA GPU temp: {e}")

    # Try AMD (Linux)
    try:
        amd_temp = subprocess.check_output("rocm-smi --showtemp --json", shell=True).decode()
        return float(eval(amd_temp)["card0"]["Temperature (Sensor edge) (C)"])
    except Exception:
        raise Exception("No valid gpu (OR error)")


def get_cpu_temperature():
    """Returns CPU temperature in Celsius."""
    system = platform.system()
    if str(system) == "Linux":
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                return max(temp.current for temp in temps["coretemp"])

            if "k10temp" in temps:
                return max(temp.current for temp in temps["k10temp"])

        except AttributeError:
            raise Exception("Sensor data not available")

    elif system == "Windows" and wmi_available:
        try:
            w = wmi.WMI(namespace="root\WMI")
            sensors = w.MSAcpi_ThermalZoneTemperature()
            if sensors:
                return round(sensors[0].CurrentTemperature / 10 - 273.15, 2)  # Convert to Celsius
        except Exception:
            raise Exception("WMI error")

    elif system == "Darwin":  # macOS
        try:
            output = subprocess.check_output(["istats", "cpu", "temp"]).decode()
            return float(output.split(":")[-1].strip().replace("°C", ""))
        except Exception:
            raise Exception("Install iStats: 'gem install iStats'")

    raise Exception("CPU temperature not supported")

pygame.init()

def get_cpu_usage() -> float:
    return psutil.cpu_percent(interval=0.5)

def get_mem_usage():
    return psutil.virtual_memory().percent

def get_swap_usage():
    """Returns swap memory usage in percentage."""
    return psutil.swap_memory().percent

def get_datapoints_test():
    return [(1, 200), (2, 150), (3, 120), (4, 110), (5, 108), (6, 106), (7, 98), (8, 50), (9, 40), (10, 38), (11, 37)]


class Utilisation:
    def __init__(self, name, get_usage_func, softness_factor=0.1):
        self.size = 200
        self.surface = pygame.Surface((self.size, self.size))
        self.name = name
        self.get_usage = get_usage_func

        self.font = pygame.sysfont.SysFont("monospace", 24)
        self.last_utilisation = 0
        self.current_usage = 0

        self.softness_factor = softness_factor

        threading.Thread(target=self.__update_usage, daemon=True).start()


    def __update_usage(self):
        while True:
            self.current_usage = self.get_usage()


    def display_load(self, frame_counter):
        pygame.draw.arc(self.surface, (255, 255, 255),
                        self.surface.get_rect(),
                        math.radians(270-45 - (270 * (min(60, frame_counter) / 60))),
                        math.radians(270-45),
                        1)


        if frame_counter > 60:
            colour = [round(255 * (min(60, frame_counter - 60) / 60)) for i in range(3)]  # for RGB

            text = self.font.render(self.name, True, colour)
            self.surface.blit(
                text,
                (
                    self.size // 2 - text.get_width() // 2,
                    self.size - text.get_width()
                )
            )

            pygame.draw.arc(self.surface, [50 + round(150 * (min(60, max(0, frame_counter-61)) / 60)) for i in range(3)],
                            [3, 3, self.size-6, self.size-6],
                            math.radians(270 - 45 - (270 * (min(60, max(0, frame_counter-60)) / 60))),
                            math.radians(270 - 45 - (270 * (min(60, max(0, frame_counter-61)) / 60))),
                            3)


        return self.surface

    def display(self):
        new = pygame.Surface((self.size, self.size))
        new.blit(self.surface, (0, 0))

        if self.last_utilisation > 0:
            self.last_utilisation -= (self.last_utilisation - self.current_usage) * self.softness_factor

        else:
            self.last_utilisation = self.current_usage

        angle = math.radians(225 - (2.7 * self.last_utilisation))
        dx = math.cos(angle) * ((self.size / 2) - 10)
        dy = math.sin(angle) * ((self.size / 2) - 10)

        pygame.draw.line(
            new,
            (255, 255, 255),
            (self.size//2, self.size//2),
            (self.size//2 + dx, self.size//2 - dy),
        )
        return new


class BoxWindow:
    def __init__(self, w, h):
        self.width, self.height = w, h

        self.surface = pygame.Surface((self.width, self.height))

    def display_load(self, frame_count):
        section_A_progress = min(60, frame_count) / 60
        section_B_progress = min(60, max(0, frame_count - 60)) / 60

        pygame.draw.line(
            self.surface,
            (200, 200, 200),
            (0, 0),
            (0, self.height * section_A_progress),
            width=3,
        )

        pygame.draw.line(
            self.surface,
            (200, 200, 200),
            (0, 0),
            (self.width * section_A_progress, 0),
            width=5,
        )

        if section_B_progress > 0:
            pygame.draw.line(
                self.surface,
                (200, 200, 200),
                (0, self.height),
                (self.width * section_B_progress, self.height),
                width=5,
            )

            pygame.draw.line(
                self.surface,
                (200, 200, 200),
                (self.width, 0),
                (self.width, self.height * section_B_progress),
                width=5,
            )



        return self.surface


    def display(self):
        return self.surface


class Graph:
    def __init__(self, width, height, get_datapoints_func, x_axis_label, y_axis_label):
        self.width = width
        self.height = height

        self.datapoints = []
        self.surface = pygame.Surface((width, height))

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        self.get_datapoints = get_datapoints_func

        self.font = pygame.sysfont.SysFont("monospace", 20)

    def display_load(self, frame_count):
        section_A_progress = min(60, frame_count) / 60
        section_B_progress = min(60, max(0, frame_count - 60)) / 60

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (20, self.height - 20),
            (20, (self.height - 20) * (1 - section_A_progress)),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (20, self.height - 20),
            (20 + (self.width - 20) * section_A_progress, self.height - 20),
            width=2,
        )

        if section_B_progress > 0:
            colour = [round(255 * (min(60, frame_count - 60) / 60)) for i in range(3)]
            x_label = self.font.render(self.x_axis_label, True, colour)
            self.surface.blit(x_label, (self.width // 2 - x_label.get_width() // 2, self.height - x_label.get_height()))

            y_label = pygame.transform.rotate((self.font.render(self.y_axis_label, True, colour)), 90)
            self.surface.blit(y_label, (0, self.height // 2 - y_label.get_height() // 2))


            for i, x in enumerate(range(0, self.width-20, (self.width-20) // 10)):
                pygame.draw.line(
                    self.surface,
                    (150, 150, 150),
                    (38 + x, self.height - 24),
                    (38 + x, (self.height - 24) * (1 - section_B_progress)),
                    width=1,
                )



            for y in range(0, self.height-20, (self.height-20) //10):
                pygame.draw.line(
                    self.surface,
                    (150, 150, 150),
                    (self.width + 2, 20 + y),
                    (22 + (self.width - 20) * section_B_progress, 20 + y),
                    width=1,
                )

        return self.surface

    def display(self):
        return self.surface


class Display:
    def __init__(self):
        screen_size = pygame.display.get_desktop_sizes()[0]
        self.screen = pygame.display.set_mode(screen_size)

        self.running = False
        self.display_loading = True
        self.clock = pygame.time.Clock()

        self.display_elements = [
            [BoxWindow(470, 470), (self.screen.get_width() - 510, 10)],
            [Utilisation("CPU", get_cpu_usage), (self.screen.get_width() - 500, 30)],
            [Utilisation("GPU", get_gpu_usage, softness_factor=0.01), (self.screen.get_width() - 250, 30)],
            [Utilisation("MEM", get_mem_usage, softness_factor=0.01), (self.screen.get_width() - 500, 270)],
            [Utilisation("SWP", get_swap_usage, softness_factor=0.01), (self.screen.get_width() - 250, 270)],

            [BoxWindow(470, 240), (self.screen.get_width() - 510, 10+500)],
            [Utilisation("C-TEMP", get_cpu_temperature), (self.screen.get_width() - 500, 30+500)],
            [Utilisation("G-TEMP", get_gpu_temperature), (self.screen.get_width() - 250, 30+500)],

            [Graph(500, 500, get_datapoints_test, "Epoch", "Error"), (self.screen.get_width() // 2 - 250, 20)],
        ]


    def run(self):
        self.running = True

        frame_counter = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False


                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False


            if self.display_loading:
                for element, [x, y] in self.display_elements:
                    self.screen.blit(element.display_load(frame_counter), (x, y))
                frame_counter += 1

                if frame_counter == 121:
                    self.display_loading = False

            else:
                for element, [x, y] in self.display_elements:
                    self.screen.blit(element.display(), (x, y))

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    display = Display()
    display.run()