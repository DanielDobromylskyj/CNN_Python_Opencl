import tkinter as tk
import threading
import time
import datetime
import sys
import subprocess
import os
import hashlib

import tkinter as tk
from evdev import InputDevice, categorize, ecodes, list_devices
import threading
import os
import sys


import device_info

import builtins

log_path = "output.log"
log_file = open(log_path, "a")

def print(*args, **kwargs):
    builtins.print(*args, **kwargs, file=log_file)
    log_file.flush()

print("\n\n--- Script started ---")

device = device_info.get_device_info()  # Only has AMD support (that is tested) + Linux uses ROCm
os.environ["PYOPENCL_CTX"] = ''

def check_password(password):
    return hashlib.sha256(password.encode()).digest() == b'\x16Y@\x94\n\x02\xa1\x87\xe4F?\xf4g\t\t0\x03\x8cZ\xf8\xfc&\x10{\xf3\x01\xe7\x14\xf5\x99\xa1\xda'


def require_root():
    if os.geteuid() != 0:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        env = os.environ.copy()
        display = env.get("DISPLAY")
        xauth = env.get("XAUTHORITY")

        if not display or not xauth:
            print("Missing DISPLAY or XAUTHORITY. Cannot launch GUI.")
            sys.exit(1)

        process = subprocess.Popen([
            'pkexec',
            'env',
            f'DISPLAY={display}',
            f'XAUTHORITY={xauth}',
            'bash',
            '-c',
            f'cd "{script_dir}" && {sys.executable} "{script_path}"'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read output (optional live)
        out, err = process.communicate()
        print("STDOUT:", out)
        print("STDERR:", err)
        sys.exit()


class Device:
    def __init__(self, device):
        self.device = device

    def grab(self):
        return self.device.grab()

    def read_loop(self):
        return self.device.read_loop()

    def ungrab(self):
        return self.device.ungrab()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        return self.ungrab()

def find_keyboard():
    for path in list_devices():
        dev = InputDevice(path)
        if "keyboard" in dev.name.lower():
            return Device(dev)
    return None



class Utilisation:
    def __init__(self, get_usage_func):
        self.get_usage = get_usage_func
        self.current_usage = 0

        threading.Thread(target=self.__update_usage, daemon=True).start()

    def __update_usage(self):
        while True:
            self.current_usage = self.get_usage()
            time.sleep(0.2)


# noinspection PyAttributeOutsideInit
class Display(tk.Tk):
    def __init__(self, net=None, lock=False):
        super().__init__()

        self.net = net
        self.net_eta_history = []
        self.net_eta = "N/A"
        self.net_finish_eta = "N/A"

        self.unlock_pin = ""

        self.title("Network Monitor")
        self.configure(bg="#121212")
        self.geometry("1000x500")
        self.resizable(False, False)

        self.is_locked = lock
        self.lock_keyboard = None

        if lock is True:
            #self.attributes('-fullscreen', True)
            #self.geometry("1920x1080")
            self.lock()

        self.main_frame = tk.Frame(self, bg="#121212", width=1000)
        self.main_frame.pack()

        self.left_frame = tk.Frame(self.main_frame, bg="#121212", width=500)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.middle_frame = tk.Frame(self.main_frame, bg="#121212", width=200)
        self.middle_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = tk.Frame(self.main_frame, bg="#1e1e1e", width=500)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.right_frame.pack_propagate(False)

        self.create_stat_boxes()
        self.create_network_info()
        self.create_graph_panel()

        self.update_stats()

    def unlock(self):
        """Unblocks keyboard and mouse input."""
        self.destroy()
        self.quit()

        raise Exception("GET ME OUTA HERE")

    def password_manager(self, device):
        device.grab()

        print("Grabbed device")

        with device:
            for event in device.read_loop():
                print("EVENT", event.type, event)
                if event.type == ecodes.EV_KEY:
                    self.key_press(event)


    def key_press(self, event):
        print("keypress", event)
        if event.char in "01234567891012":
            self.unlock_pin += event.char

        if event.char == '\r':
            print("Password Check:", check_password(self.unlock_pin))
            if check_password(self.unlock_pin):
                self.unlock()

            self.unlock_pin = ""


    def lock(self):
        require_root()

    def update_eta(self):
        if self.net is not None:
            epoch, max_epoches, _, _ = self.net.get_extra_display_data()
            self.net_eta_history.append((epoch, time.time()))

            if len(self.net_eta_history) > 50:
                self.net_eta_history.pop(0)

            e1, t1 = self.net_eta_history[0]
            e2, t2 = self.net_eta_history[-1]

            epoch_range = e2 - e1
            time_range = t2 - t1

            if epoch_range <= 0:
                self.net_eta = "N/A"
                return

            time_per_epoch = time_range / epoch_range
            time_left = time_per_epoch * (max_epoches - epoch)
            self.net_eta = str(datetime.timedelta(seconds=round(time_left)))

            now = datetime.datetime.now()
            finish_time = now + datetime.timedelta(seconds=round(time_left))
            self.net_finish_eta = finish_time.strftime('%H:%M')


    @staticmethod
    def get_scale(value):
        if value > 1024**3:
            return f"{round(value / (1024**3))}GB"
        elif value > 1024**2:
            return f"{round(value / (1024**2))}MB"
        elif value > 1024:
            return f"{round(value / 1024)}KB"
        return f"{round(value)}B"

    def create_network_info(self):
        tk.Label(self.middle_frame, text="", font=("Segoe UI", 9), fg="gray",
                                     bg="#121212").pack(pady=(0, 10), anchor="w")

        epoch_status_frame = tk.LabelFrame(self.middle_frame, text="Progress", fg="white", bg="#1e1e1e",
                              font=("Segoe UI", 10, "bold"), bd=2, relief="groove")
        epoch_status_frame.pack(fill="x", pady=5)

        self.epoch_label = self.make_stat_label(epoch_status_frame, "Epoch: --/--")
        self.batch_label = self.make_stat_label(epoch_status_frame, "Batch:  --/--")

        eta_frame = tk.LabelFrame(self.middle_frame, text="Training ETA", fg="white", bg="#1e1e1e",
                                           font=("Segoe UI", 10, "bold"), bd=2, relief="groove")
        eta_frame.pack(fill="x", pady=5)

        self.eta_remaining_label = self.make_stat_label(eta_frame, "Remaining:   --:--:--")
        self.eta_finish_label = self.make_stat_label(eta_frame, "Finish Time: --:--:--")

        size_frame = tk.LabelFrame(self.middle_frame, text="Network", fg="white", bg="#1e1e1e",
                                  font=("Segoe UI", 10, "bold"), bd=2, relief="groove")
        size_frame.pack(fill="x", pady=5)

        mem_size = "---MB" if not self.net else self.get_scale(self.net.get_network_mem_size())
        gpu_size = "---MB" if not self.net else self.get_scale(self.net.get_gpu_buffer_size())
        layer_count = "--" if not self.net else len(self.net.layout)
        neuron_count = "--" if not self.net else self.net.get_neuron_count()
        bias_count = "--" if not self.net else self.net.get_bias_count()

        self.mem_size_label = self.make_stat_label(size_frame, f"Aprox MemSize: {mem_size}")
        self.gpu_size_label = self.make_stat_label(size_frame, f"Aprox BufferSize: {gpu_size}")
        self.layout_count_label = self.make_stat_label(size_frame, f"Layer Count: {layer_count}")
        self.neuron_count_label = self.make_stat_label(size_frame, f"Neuron Count: {neuron_count}")
        self.bias_count_label = self.make_stat_label(size_frame, f"Bias Count: {bias_count}")

    def create_stat_boxes(self):
        self.device_label = tk.Label(self.left_frame, text=device.get_name(), font=("Segoe UI", 9), fg="gray",
                                     bg="#121212")
        self.device_label.pack(pady=(0, 10), anchor="w")

        self.cpu_box = self.make_box("CPU Info")
        self.cpu_label = self.make_stat_label(self.cpu_box, "Usage: --%")
        self.cpu_usage = Utilisation(device.get_cpu_usage)

        self.cpu_temp_label = self.make_stat_label(self.cpu_box, "Temp: --°C")
        self.cpu_temp = Utilisation(device.get_cpu_temperature)

        self.gpu_box = self.make_box("GPU Info")
        self.gpu_label = self.make_stat_label(self.gpu_box, "Usage: --%")
        self.gpu_usage = Utilisation(device.get_gpu_usage)

        self.gpu_temp_label = self.make_stat_label(self.gpu_box, "Temp: --°C")
        self.gpu_temp = Utilisation(device.get_gpu_temperature)

        self.mem_box = self.make_box("Memory Info")
        self.ram_label = self.make_stat_label(self.mem_box, "Usage: --GB (--%)")
        self.ram_usage = Utilisation(device.get_mem_usage)
        self.ram_max = device.get_mem_max()

        self.swap_label = self.make_stat_label(self.mem_box, "Swap: --GB (--%)")
        self.swap_usage = Utilisation(device.get_swap_usage)
        self.swap_max = device.get_swap_max()

        self.power_box = self.make_box("Power Info")
        self.cpu_power_label = self.make_stat_label(self.power_box, "CPU: --W")
        self.cpu_power = Utilisation(device.get_cpu_power)

        self.gpu_power_label = self.make_stat_label(self.power_box, "GPU: --W")
        self.gpu_power = Utilisation(device.get_gpu_power)

        self.after(5000, self.unlock)

    def create_graph_panel(self):
        title = tk.Label(self.right_frame, text="Epoch vs Error", font=("Segoe UI", 12, "bold"),
                         fg="white", bg="#1e1e1e", width=420)
        title.pack(anchor="n", pady=5)

        self.graph_canvas = tk.Canvas(self.right_frame, bg="#2e2e2e", height=340, width=450, highlightthickness=0)
        self.graph_canvas.pack(pady=10)

        self.datapoints = [0]
        self.draw_datapoints()

    def draw_datapoints(self):
        if self.net is not None:
            self.datapoints = self.net.get_error_history()[-50:]

            if len(self.datapoints) < 2:
                self.after(1000, self.draw_datapoints)
                return


        self.graph_canvas.delete("all")
        w = int(self.graph_canvas["width"]) - 20
        h = int(self.graph_canvas["height"]) - 10
        spacing = w // (len(self.datapoints) + 1)
        prev_x, prev_y = None, None

        max_y = max(*self.datapoints)
        min_y = min(*self.datapoints)

        if all(x == self.datapoints[0] for x in self.datapoints):  # if all the same
            for i, error in enumerate(self.datapoints):
                x = 20 + spacing * (i + 1)
                y = h // 2

                self.graph_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="lime", outline="lime")

                if prev_x is not None:
                    self.graph_canvas.create_line(prev_x, prev_y, x, y, fill="lime", width=1)

                prev_x, prev_y = x, y

            self.graph_canvas.create_text(35, h // 2 - 8,
                                          text=f"{max_y:.2e}" if max_y > 1e5 else str(
                                              round(max_y)) if min_y > 100 else str(
                                              round(min_y, 4)), font=("Helvetica", 12), fill="white")
        else:
            for i, error in enumerate(self.datapoints):
                x = 20 + spacing * (i + 1)
                y = h - (((error - min_y) / max_y) * h) + 5

                self.graph_canvas.create_oval(x-2, y-2, x+2, y+2, fill="lime", outline="lime")

                if prev_x is not None:
                    self.graph_canvas.create_line(prev_x, prev_y, x, y, fill="lime", width=1)

                prev_x, prev_y = x, y

            self.graph_canvas.create_text(35, h,
                                          text=f"{min_y:.2e}" if min_y > 1e5 else str(
                                              round(min_y)) if min_y > 100 else str(
                                              round(min_y, 4)), font=("Helvetica", 12), fill="white")

            self.graph_canvas.create_text(35, 16,
                                          text=f"{max_y:.2e}" if max_y > 1e5 else str(round(max_y)) if max_y > 100 else str(
                                              round(max_y, 4)), font=("Helvetica", 12), fill="white")

        self.after(1000, self.draw_datapoints)


    def make_box(self, title):
        frame = tk.LabelFrame(self.left_frame, text=title, fg="white", bg="#1e1e1e",
                              font=("Segoe UI", 10, "bold"), bd=2, relief="groove")
        frame.pack(fill="x", pady=5)
        return frame

    def make_stat_label(self, parent, text):
        label = tk.Label(parent, text=text, font=("Segoe UI", 11), fg="white", bg="#1e1e1e")
        label.pack(anchor="w", padx=10, pady=2)
        return label

    def update_stats(self):
        try:
            self.cpu_label.config(text=f"Usage: {round(self.cpu_usage.current_usage, 1)}%")
            self.cpu_temp_label.config(text=f"Temp: {round(self.cpu_temp.current_usage, 1)}°C")
            self.ram_label.config(text=f"Usage: {round((self.ram_max / (1024**3)) * (self.ram_usage.current_usage / 100), 1)}GB ({self.ram_usage.current_usage}%)")
            self.swap_label.config(text=f"Swap: {round((self.swap_max / (1024**3)) * (self.swap_usage.current_usage / 100), 1)}GB ({self.swap_usage.current_usage}%)")

            self.gpu_label.config(text=f"Usage: {round(self.gpu_usage.current_usage, 1)}%")
            self.gpu_temp_label.config(text=f"Temp: {round(self.gpu_temp.current_usage, 1)}°C")

            self.cpu_power_label.config(text=f"CPU: {round(self.cpu_power.current_usage, 1)}W")
            self.gpu_power_label.config(text=f"GPU: {round(self.gpu_power.current_usage, 1)}W")
        except Exception as e:
            print("Error updating stats:", e)

        if self.net is not None:
            self.update_eta()
            epoches_done, epoches_total, batch_complete, batch_total = self.net.get_extra_display_data()

            self.epoch_label.config(text=f"Epoch: {epoches_done}/{epoches_total}")
            self.batch_label.config(text=f"Batch: {batch_complete}/{batch_total}")
            self.eta_remaining_label.config(text=f"Remaining: {self.net_eta}")
            self.eta_finish_label.config(text=f"Finish Time: {self.net_finish_eta}")

        self.after(200, self.update_stats)

    def run(self):
        print("Running")
        if self.is_locked:
            print("As Locked")
            self.focus_force()

            keyboard = find_keyboard()
            if not keyboard:
                print("No keyboard found.")
                sys.exit(1)

            t = threading.Thread(target=self.password_manager, args=(keyboard, ), daemon=True)
            t.start()

            try:
                print("Main loop")
                self.mainloop()
                device.ungrab()

            except:
                device.ungrab()
                raise


        else:
            self.mainloop()

    @staticmethod
    def create_and_run(*args, **kwargs):
        Display(*args, **kwargs).run()



    @staticmethod
    def launch_threaded(*args, **kwargs):
        threading.Thread(target=Display.create_and_run, args=args, kwargs=kwargs, daemon=True).start()


if __name__ == "__main__":
    import activations
    import layers
    import network

    import numpy as np

    USE_LOCK = True

    if not os.geteuid() != 0 and USE_LOCK:
        print("READY")
    a, b = np.random.random((100, 100)), np.random.random((100, 100))

    training_data = [
        [a, np.array([1, 0])],
        [b, np.array([0, 1])],
    ]

    net = network.Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=5, colour_depth=1, stride=2),
        layers.FullyConnectedLayer(11520, 2, activations.ReLU)
    ))

    net.save("display_lite_test.pyn")

    Display.launch_threaded(net, lock=False)

    net.train(training_data, training_data, 500, -0.001, show_stats=False)
