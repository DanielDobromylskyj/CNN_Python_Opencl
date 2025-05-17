import string
import time
import pygame
import math
import threading
import sys


import device_info

device = device_info.get_device_info()  # Only has AMD support (that is tested) + Linux uses ROCm

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
                    self.size - text.get_height()
                )
            )

            pygame.draw.arc(self.surface, [50 + round(150 * (min(60, max(0, frame_counter-61)) / 60)) for i in range(3)],
                            [3, 3, self.size-6, self.size-6],
                            math.radians(270 - 45 - (270 * (min(60, max(0, frame_counter-60)) / 60))),
                            math.radians(270 - 45 - (270 * (min(60, max(0, frame_counter-61)) / 60))),
                            3)

        if frame_counter == 120:
            threading.Thread(target=self.__update_usage, daemon=True).start()

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

        self.datapoints = get_datapoints_func()
        self.surface = pygame.Surface((width, height))

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        self.get_datapoints = get_datapoints_func

        self.font = pygame.sysfont.SysFont("monospace", 20)

    def display_load(self, frame_count):
        section_A_progress = min(40, frame_count) / 40
        section_B_progress = min(40, max(0, frame_count - 40)) / 40
        section_C_progress = min(40, max(0, frame_count - 80)) / 40

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
            colour = [round(255 * (min(40, frame_count - 40) / 40)) for i in range(3)]
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

        if section_C_progress > 0:
            max_x = max(self.datapoints, key=lambda p: p[0])[0]
            max_y = max(self.datapoints, key=lambda p: p[1])[1]
            min_x = min(self.datapoints, key=lambda p: p[0])[0]
            min_y = min(self.datapoints, key=lambda p: p[1])[1]

            range_x = max_x - min_x
            range_y = max_y - min_y

            for i, segment in enumerate(self.datapoints[:-1]):
                next_segment = self.datapoints[i+1]

                x = 20 + (segment[0] - min_x) * ((self.width-20)/range_x)
                y = self.height - 20 - (segment[1] - min_y) * ((self.height-20)/range_y)

                next_x = 20 + (next_segment[0] - min_x) * ((self.width-20)/range_x)
                next_y = self.height - 20 - (next_segment[1] - min_y) * ((self.height-20)/range_y)


                pygame.draw.line(
                    self.surface,
                    (255, 255, 255),
                    (x, y),
                    (x + (next_x - x) * section_C_progress,
                     y +  (next_y - y) * section_C_progress),
                    width=2
                )


        return self.surface

    def display(self):
        return self.surface


class TerminalOutput:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.surface = pygame.Surface((width, height))
        self.font = pygame.sysfont.SysFont("monospace", 14)

        self.lines = [""]
        self.input_text = ""
        self.awaiting_input = False

        sys.stdout = self
        sys.stdin = self

    def flush(self):
        pass

    def write(self, text):
        for char in text:
            if char == "\n":
                self.lines.append("")

            elif char == "\r":
                self.lines[-1] = ""

            else:
                self.lines[-1] += char

        font_height = self.font.get_height()
        max_lines_in_terminal = (self.height - 30) // font_height
        redundant_lines = len(self.lines) - max_lines_in_terminal

        if redundant_lines > 0:
            for i in range(redundant_lines):
                self.lines.pop(0)


    def on_key_press(self, event):
        if self.awaiting_input:
            if event.unicode == "\r":
                self.awaiting_input = False

            elif event.key == pygame.K_BACKSPACE:
                if self.input_text != "":
                    self.input_text = self.input_text[:-1]

            elif event.unicode in string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation:
                self.input_text += event.unicode

    def readline(self):
        self.input_text = ""
        self.awaiting_input = True

        while self.awaiting_input:
            time.sleep(0.1)

        return self.input_text


    def display_load(self, frame_count):
        section_A_progress = min(60, frame_count) / 60
        section_B_progress = min(60, max(0, frame_count - 60)) / 60

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (self.width * section_A_progress, 0),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (0, (self.height-30) * section_A_progress),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (self.width, self.height-30),
            (self.width * (1 - section_A_progress), self.height-30),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (self.width-2, self.height-30),
            (self.width-2, (self.height-30) * (1 - section_A_progress)),
            width=2,
        )

        if section_B_progress > 0:
            pygame.draw.rect(
                self.surface,
                (0, 0, 0),
                (0, self.height-28, self.width, 30),
            )

            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (0, (self.height - 30) + (28 * section_B_progress)),
                (self.width, (self.height - 30) + (28 * section_B_progress)),
                width=2,
            )

            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (0, (self.height - 30)),
                (0, (self.height - 30) + (28 * section_B_progress)),
                width=2,
            )

            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (self.width-2, (self.height - 30)),
                (self.width-2, (self.height - 30) + (28 * section_B_progress)),
                width=2,
            )

        return self.surface

    def display(self):
        new = pygame.Surface((self.width, self.height))
        new.blit(self.surface, (0, 0))

        dy = 5
        for line in self.lines:
            text = self.font.render(line, True, (255, 255, 255))
            new.blit(text, (5, dy))
            dy += text.get_height()

        if self.awaiting_input:
            text = self.font.render(self.input_text, True, (255, 255, 255))
            new.blit(text, (5, self.height - 28))

        return new


class PowerOutput:
    def __init__(self, width, height, name, measure_func):
        self.width = width
        self.height = height

        self.name = name
        self.name_pixel_width = 0

        self.surface = pygame.Surface((self.width, self.height))
        self.get_reading = measure_func

        self.current_reading = 0
        self.summed_readings = 0
        self.readings = 0
        self.peak_reading = 0

        self.font_large = pygame.font.Font("misc/data-latin.ttf", 50)
        self.font_small = pygame.font.Font("misc/data-latin.ttf", 20)

    def __update_usage(self):
        while True:
            self.current_reading = self.get_reading()
            self.summed_readings += self.current_reading

            if self.current_reading > self.peak_reading:
                self.peak_reading = self.current_reading

            self.readings += 1


    def display_load(self, frame_count):
        section_A_progress = min(60, frame_count) / 60
        section_B_progress = min(60, frame_count - 60) / 60

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (self.width * section_A_progress, 0),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (0, self.height * section_A_progress),
            width=2,
        )

        if section_B_progress > 0:
            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (0, self.height- 2),
                (self.width * section_B_progress, (self.height - 2)),
                width=2,
            )

            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (self.width - 2, 0),
                (self.width - 2, self.height * section_B_progress),
                width=2,
            )

            colour = [round(255 * section_B_progress) for i in range(3)]

            text = self.font_large.render(self.name, True, colour)
            self.surface.blit(text, (15, (self.height / 2) - (text.get_height() / 2)))
            self.name_pixel_width = text.get_width()

            wattage = self.font_large.render(f"---", True, colour)
            avg_wattage = self.font_small.render(f"avg: ---",
                                                 True, colour)
            peak_wattage = self.font_small.render(f"max: ---", True, colour)

            total_height = wattage.get_height() + avg_wattage.get_height() + peak_wattage.get_height() + 10

            dis_surf = pygame.Surface((200, total_height))
            dis_surf.blit(wattage, (0, 0))
            dis_surf.blit(avg_wattage, (5, wattage.get_height() + 5))
            dis_surf.blit(peak_wattage, (5, wattage.get_height() + avg_wattage.get_height() + 10))

            self.surface.blit(dis_surf,
                     (self.width - dis_surf.get_width() - 5, (self.height // 2) - (dis_surf.get_height() // 2)))


        if frame_count == 120:
            threading.Thread(target=self.__update_usage, daemon=True).start()

        return self.surface

    def display(self):
        new = pygame.Surface((self.width, self.height))
        new.blit(self.surface, (0, 0))

        wattage = self.font_large.render(f"{round(self.current_reading, 1)}W", True, (255, 255, 255))
        avg_wattage = self.font_small.render(f"avg: {round(self.summed_readings/(self.readings+0.01), 1)}W", True, (255, 255, 255))
        peak_wattage = self.font_small.render(f"max: {round(self.peak_reading, 1)}W", True, (255, 255, 255))

        total_height = wattage.get_height() + avg_wattage.get_height() + peak_wattage.get_height() + 10

        dis_surf = pygame.Surface((200, total_height))
        dis_surf.blit(wattage, (0, 0))
        dis_surf.blit(avg_wattage, (5, wattage.get_height() + 5))
        dis_surf.blit(peak_wattage, (5, wattage.get_height() + avg_wattage.get_height() + 10))

        new.blit(dis_surf, (self.width - dis_surf.get_width() - 5, (self.height // 2) - (dis_surf.get_height() // 2)))

        return new


class EpochProgress:
    def __init__(self, width, height, value_a: dict, value_b: dict, value_c: dict):
        self.width = width
        self.height = height

        self.value_a = value_a
        self.value_b = value_b
        self.value_c = value_c

        self.values = [0, 0, 0]

        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("monospace", 20)

    def __update_usage(self, func, index):
        while True:
            self.values[index] = func()
            time.sleep(0.05)

    def display_load(self, frame_count):
        section_A_progress = min(60, frame_count) / 60
        section_B_progress = min(60, frame_count - 60) / 60

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (self.width * section_A_progress, 0),
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (0, 0),
            (0, self.height * section_A_progress),
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (self.width, self.height - 2),
            (self.width * (1 - section_A_progress), (self.height - 2)),
            width=2,
        )

        pygame.draw.line(
            self.surface,
            (255, 255, 255),
            (self.width - 2, self.height),
            (self.width - 2, self.height * (1 - section_A_progress)),
            width=2,
        )

        if section_B_progress > 0:
            pygame.draw.rect(
                self.surface,
                (0, 0, 0),
                (2, 2, self.width-4, self.height-4),
            )

            section_heights = (self.height - 4) // 6

            pygame.draw.rect(
                self.surface,
                (255, 255, 255),
                (40, 10 + section_heights, (self.width - 80) * section_B_progress, 40),
                width=2,
            )

            pygame.draw.rect(
                self.surface,
                (255, 255, 255),
                (40, 10 + (section_heights*3), (self.width - 80) * section_B_progress, 40),
                width=2,
            )

            pygame.draw.rect(
                self.surface,
                (255, 255, 255),
                (40, 10 + (section_heights*5), (self.width - 80) * section_B_progress, 40),
                width=2,
            )

            colour = [round(255 * section_B_progress) for i in range(3)]
            text = self.font.render(self.value_a["name"], True, colour)
            self.surface.blit(text, ((self.width // 2) - (text.get_width() // 2), 2 + (section_heights * 0.5)))

            text = self.font.render(self.value_b["name"], True, colour)
            self.surface.blit(text, ((self.width // 2) - (text.get_width() // 2), 2 + (section_heights * 2.5)))

            text = self.font.render(self.value_c["name"], True, colour)
            self.surface.blit(text, ((self.width // 2) - (text.get_width() // 2), 2 + (section_heights * 4.5)))

        if frame_count == 120:
            threading.Thread(target=self.__update_usage, args=(self.value_a["func"], 0), daemon=True).start()
            threading.Thread(target=self.__update_usage, args=(self.value_b["func"], 1), daemon=True).start()
            threading.Thread(target=self.__update_usage, args=(self.value_c["func"], 2), daemon=True).start()

        return self.surface

    def display(self):
        new = pygame.Surface((self.width, self.height))
        new.blit(self.surface, (0, 0))

        section_heights = (self.height - 4) // 6

        pygame.draw.rect(
            new,
            (255, 255, 255),
            (40, 10 + section_heights, (self.width - 80) * (self.values[0]), 40),
        )

        pygame.draw.rect(
            new,
            (255, 255, 255),
            (40, 10 + (section_heights * 3), (self.width - 80) * (self.values[1]), 40),
        )

        pygame.draw.rect(
            new,
            (255, 255, 255),
            (40, 10 + (section_heights * 5), (self.width - 80) * (self.values[2]), 40),
        )
        return new


def Display_threaded(*args):
    threading.Thread(target=__display_threaded, args=args).start()

def __display_threaded(*args):
    Display(*args).run()

def return_0():
    time.sleep(1)
    return 0


class Display:
    def __init__(self, net=None):
        pygame.init()

        screen_size = pygame.display.get_desktop_sizes()[0]
        self.screen = pygame.display.set_mode(screen_size, pygame.FULLSCREEN)

        self.running = False
        self.display_loading = True
        self.clock = pygame.time.Clock()

        self.display_elements = [
            [BoxWindow(470, 470), (self.screen.get_width() - 510, 10)],
            [Utilisation("CPU", device.get_cpu_usage), (self.screen.get_width() - 500, 30)],
            [Utilisation("GPU", device.get_gpu_usage, softness_factor=0.01), (self.screen.get_width() - 250, 30)],
            [Utilisation("MEM", device.get_mem_usage, softness_factor=0.01), (self.screen.get_width() - 500, 270)],
            [Utilisation("SWP", device.get_swap_usage, softness_factor=0.01), (self.screen.get_width() - 250, 270)],

            [BoxWindow(470, 240), (self.screen.get_width() - 510, 10+500)],
            [Utilisation("C-TEMP", device.get_cpu_temperature), (self.screen.get_width() - 500, 30+500)],
            [Utilisation("G-TEMP", device.get_gpu_temperature), (self.screen.get_width() - 250, 30+500)],

            [Graph(500, 500, get_datapoints_test, "Epoch", "Error"), (self.screen.get_width() // 2 - 100, 20)],
            [TerminalOutput(1350, 400), (20, 580)],

            [PowerOutput(470, 150, "GPU", device.get_gpu_power), (self.screen.get_width() - 510, 30+500+300)],
            [EpochProgress(700, 400,
                           {"name": "Epoch", "func": (lambda: net.get_display_data()[0]) if net else return_0},
                           {"name": "Batch", "func": (lambda: net.get_display_data()[1]) if net else return_0},
                           {"name": "Progress", "func": (lambda: net.get_display_data()[2]) if net else return_0}),
             (20, 20)]
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

                    for element, pos in self.display_elements:
                        if hasattr(element, "on_key_press"):
                            element.on_key_press(event)


            if self.display_loading:
                for element, [x, y] in self.display_elements:
                    self.screen.blit(element.display_load(frame_counter), (x, y))
                frame_counter += 1

                if frame_counter == 121:
                    print("[DISPLAY] Loading Complete")
                    self.display_loading = False

            else:
                for element, [x, y] in self.display_elements:
                    self.screen.blit(element.display(), (x, y))

            pygame.display.flip()

            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    display = Display()
    display.run()