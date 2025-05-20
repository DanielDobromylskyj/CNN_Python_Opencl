import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import lz4.frame
import hashlib

from platformdirs import user_cache_dir
import importlib.resources


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with importlib.resources.files("sampler").joinpath(f"data_maker.cl").open("r") as f:
    cl_code = f.read()

program = cl.Program(ctx, cl_code).build()

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def str_hash(str_data: str) -> bytes:
    return hashlib.sha256(str_data.encode()).digest()


def get_cache_dir() -> str:
    dir = os.path.join(user_cache_dir("myconet"), "samples")
    os.makedirs(dir, exist_ok=True)
    return dir


class rotate_180:  # fixme - Broken
    @staticmethod
    def jit(sample):
        width = sample.data.shape[0]
        height = sample.data.shape[1]

        for x in range(width // 2):
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

            for i in range(3):
                sample.data[y][x][i] *= 0.9

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


def load_training_data(path="trainingPoints.txt", mutations_per_image=20, transformations_per_image=3, pre_load=False, cache=True):
    trainingData = []
    transformations = (rotate_180, blur, colour_shift, add_noise_specs, random_colour_shift)

    if not os.path.exists(f"{get_cache_dir()}/training_data_hash.txt"):
        open(f"{get_cache_dir()}/training_data_hash.txt", 'w').close()


    with open(f"{get_cache_dir()}/training_data_hash.txt", "rb") as f:
        old_hash = f.read()

    with open(path, "r") as f:
        data = f.read()

        file_hash = str_hash(data)

        if cache and file_hash == old_hash and os.path.exists(f"{get_cache_dir()}/training_data_cache.bin"):
            return read_from_cache(f"{get_cache_dir()}/training_data_cache.bin")

        unprocessedTrainingData = eval(data)


    start = time.time()
    last_update = start
    sample_count = sum([len(segment["points"]) for segment in unprocessedTrainingData]) * mutations_per_image
    sample_index = 0
    for segment in unprocessedTrainingData:
        slide = openslide.open_slide(segment["tiffPath"])
        for point in segment["points"]:
            for i in range(mutations_per_image):
                sample_transformations = random.choices(transformations, k=transformations_per_image)

                sample = TrainingSample(slide, point[0], point[1], sample_transformations)

                if pre_load:
                    sample.load()

                trainingData.append(sample)

                sample_index += 1
                if pre_load and time.time() - last_update > 0.5:
                    last_update = time.time()

                    print(
                        f"\r[PreLoading] Loading {sample_index}/{sample_count} ({round(sample_index / sample_count * 100, 1)}%)",
                        end='')

    if pre_load:
        print(f"\r[PreLoading] Loaded {sample_index} samples")

        if cache and (file_hash != old_hash or not os.path.exists(f"{get_cache_dir()}/training_data_cache.bin")):
            print("[PreLoading] Writing loaded data to cache...")
            write_to_cache(f"{get_cache_dir()}/training_data_cache.bin", trainingData)

            with open(f"{get_cache_dir()}/training_data_hash.txt", "wb") as fb:
                fb.write(file_hash)

            print("[PreLoading] Cache Write Complete")


    return trainingData


class LoadedSample:
    def __init__(self, data, output):
        self.data = data
        self.output = output

    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)


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
            self.data = np.array(image.getdata(), dtype=np.float32).reshape(100, 100, 3)

            for transformer in self.transformations:
                if jit:
                    transformer.jit(self)
                else:
                    transformer.preprocess(self)

            self.data = self.data.astype(dtype=np.float32) / 255

    def __array__(self, dtype=None):
        if not self.data:  # Panic cos we haven't loaded it yet :)
            self.load(jit=True)

        return np.array(self.data, dtype=dtype)


def write_to_cache(path, data):
    with lz4.frame.open(path, mode='wb') as f:
        for sample in data:
            for point in [sample.data, sample.output]:
                point_bytes = point.tobytes()
                length_bytes = len(point_bytes).to_bytes(32, "little")

                f.write(length_bytes)
                f.write(point_bytes)


def read_from_cache(path):
    data = []
    with lz4.frame.open(path, mode='rb') as f:
        while True:
            length_bytes = f.read(32)
            if not length_bytes:
                break

            length = int.from_bytes(length_bytes, "little")
            data_bytes = f.read(length)

            length_bytes = f.read(32)
            length = int.from_bytes(length_bytes, "little")
            output_bytes = f.read(length)

            data.append(LoadedSample(
                np.frombuffer(data_bytes, dtype=np.float32).reshape((100, 100, 3)),
                np.frombuffer(output_bytes, dtype=np.float32)
            ))

    return data



def view_training_data(sample):
    if sample.data is None:
        sample.load(True)

    plt.imshow(sample.data)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    start = time.time()
    data = load_training_data(100, pre_load=True, cache=True)  # Ideal is 100 mutations per
    elapsed = time.time() - start
    print(f"Completed Loading of {len(data)} samples, Taking {round(elapsed*1000)}ms.")



