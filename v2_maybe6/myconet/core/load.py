import importlib.resources
import pyopencl as cl


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def load_kernel(path):
    with importlib.resources.files("myconet.core.standard").joinpath(f"{path}.cl").open("r") as f:
        return cl.Program(ctx, f.read()).build()


def load_training_kernel(path):
    with importlib.resources.files("myconet.core.train").joinpath(f"{path}.cl").open("r") as f:
        return cl.Program(ctx, f.read()).build()