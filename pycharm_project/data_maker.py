import pyopencl as cl
import numpy as np
import random
import time
import os


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with open("data_maker.cl", "r") as f:
    cl_code = f.read()

program = cl.Program(ctx, cl_code).build()

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class rotate_180:
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        for x in range(width):
            for y in range(height // 2):
                sample.data[y][x], sample.data[height - y - 1][width - x - 1] = sample.data[height - y - 1][width - x - 1], sample.data[y][x]

    @staticmethod
    def preprocess(sample):
        sample_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sample.data)

        width = sample.data.shape[0]
        height = sample.data.shape[1]

        program.rotate_180(queue, (width, height//2), None,
                           sample_buffer, np.int32(width), np.int32(height))

        cl.enqueue_copy(queue, sample_buffer, sample.data).wait()


class blur:
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        for x in range(width):
            for y in range(height):
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        if 0 < dx + x < width and 0 < dy + y < height:
                            sample.data[y][x] += sample.data[y + dy][x + dx] * 0.1 * dx

    @staticmethod
    def preprocess(sample):
        sample_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sample.data)

        width = sample.data.shape[0]
        height = sample.data.shape[1]

        program.blur(queue, (width, height), None,
                           sample_buffer, np.int32(width), np.int32(height))

        cl.enqueue_copy(queue, sample_buffer, sample.data).wait()


class colour_shift:
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        shift = (np.random.random(sample.data.shape[2]) - 0.5) * 0.5

        for x in range(width):
            for y in range(height):
                sample.data[y][x] += shift

    @staticmethod
    def preprocess(sample):
        colour_shift.jit(sample)


class add_noise_specs:
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        area = width * height

        for i in range(round(area / 20)):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)

            sample.data[y][x] *= 0.05
            sample.data[y][x] += random.random() - 0.5

    @staticmethod
    def preprocess(sample):
        add_noise_specs.jit(sample)

class random_colour_shift:
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        for x in range(width):
            for y in range(height):
                shift = (np.random.random(sample.data.shape[2]) - 0.5) * 0.5
                sample.data[y][x] += shift

    @staticmethod
    def preprocess(sample):
        random_colour_shift.jit(sample)


def load_training_data(mutations_per_image=100, transformations_per_image=3, pre_load=False):
    trainingData = []
    transformations = (rotate_180, blur, colour_shift, add_noise_specs, random_colour_shift)

    with open("trainingPoints.txt", "r") as f:
        unprocessedTrainingData = eval(f.read())

    for segment in unprocessedTrainingData:
        slide = openslide.open_slide(segment["tiffPath"])
        for point in segment["points"]:
            for i in range(mutations_per_image):
                sample_transformations = random.choices(transformations, k=transformations_per_image)

                sample = TrainingSample(slide, point[0], point[1], sample_transformations)

                if pre_load:
                    sample.load()

                trainingData.append(sample)

    return trainingData


class TrainingSample:
    def __init__(self, slide, location, output, transformations: list | tuple):
        self.slide = slide
        self.pos = location
        self.output = np.array(output, dtype=np.float32)

        self.data = None
        self.transformations = transformations

    def load(self, jit=False):
        if not self.data:
            image = self.slide.read_region(self.pos, 0, (100, 100)).convert("RGB")
            self.data = np.array(image.getdata(), dtype=np.uint8)

            for transformer in self.transformations:
                print(self.data.shape, transformer)
                if jit:
                    transformer.jit(self)
                else:
                    transformer.preprocess(self)

            self.data = self.data.astype(dtype=np.float32) / 255


    def __array__(self):
        if not self.data:  # JIT, but it's not preemptive at all
            self.load(jit=True)

        return [self.data, np.array([self.output[0], 0 if self.output[0] == 1 else 1], dtype=np.float32)]



if __name__ == "__main__":
    start = time.time()
    data = load_training_data(100, pre_load=True)

    elapsed = time.time() - start
    print(f"Completed Loading of {len(data)} samples, Taking {round(elapsed*1000)}ms.")

