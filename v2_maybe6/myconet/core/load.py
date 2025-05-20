import importlib.resources
import pyopencl as pycl

mf = pycl.mem_flags


def load_kernel(cl_instance, path):
    with importlib.resources.files("myconet.core.standard").joinpath(f"{path}.cl").open("r") as f:
        return pycl.Program(cl_instance.ctx, f.read()).build()


def load_training_kernel(cl_instance, path):
    with importlib.resources.files("myconet.core.train").joinpath(f"{path}.cl").open("r") as f:
        return pycl.Program(cl_instance.ctx, f.read()).build()
