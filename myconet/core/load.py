import importlib.resources
import pyopencl as pycl
import re


mf = pycl.mem_flags

class bcolours:
    HEADER = '\033[95m'
    ERROR_RED = '\033[91m'
    ENDC = '\033[0m'


def _load_kernel_from_file_object(cl_instance, file):
    with file:
        try:
            return pycl.Program(cl_instance.ctx, file.read()).build()
        except pycl.RuntimeError as e:
            error_text = '\n'.join(str(e).split('\n')[3:-3])
            line_error = int(re.search(r'(\d+)\s*\|', error_text).group(1))

            print(f'{bcolours.ERROR_RED}{error_text}\nSource: {file.name.replace("\\", "/")}:{line_error}{bcolours.ENDC}')
            exit(1)



def load_kernel(cl_instance, path):
    return _load_kernel_from_file_object(
        cl_instance,
        importlib.resources.files("myconet.core.standard").joinpath(f"{path}.cl").open("r")
    )


def load_training_kernel(cl_instance, path):
    return _load_kernel_from_file_object(
        cl_instance,
        importlib.resources.files("myconet.core.train").joinpath(f"{path}.cl").open("r")
    )
