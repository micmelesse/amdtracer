import os
import socket
import argparse
import shutil
import collections
from natsort import natsorted
import torch

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False


def print_var_name(variable):
    for name in globals():
        if eval(name) == variable:
            return name


def check_hostdir_exists():
    host_name = socket.gethostname()

    if os.path.exists(host_name):
        # print(host_name, "exists")
        return True
    else:
        # print(host_name, "doesnot exist")
        return False


def init_hostdir():
    if not use_tracer():
        return

    host_name = socket.gethostname()
    os.mkdir(host_name)


def use_tracer():
    USE_TRACER = os.environ.get('USE_TRACER')
    if USE_TRACER is not None:
        # print("TRACER is enabled")
        return bool(int(USE_TRACER))
    else:
        # print("TRACER is not enabled")
        return False


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def save_tensor(tensor_to_save, name=None):
    if not use_tracer():
        return

    if not check_hostdir_exists():
        init_hostdir()

    host_name = socket.gethostname()
    if name == None:
        name = print_var_name(tensor_to_save)

    # device_id = tensor_to_save.get_device()
    device_id = 0

    save_path = os.path.join(
        host_name, str(name) + "_" + str(device_id) + '.pt')

    save_path = uniquify(save_path)

    torch.save(tensor_to_save, save_path)


def check_tensor_finite(tensor, tensor_name=None):
    if tensor_name is None:
        tensor_name = tensor.name

    if tensor is None:
        print("Tensor", tensor_name, "is None")
        return False

    if torch.isfinite(tensor).all():
        print("Tensor", tensor_name, "is finite")
        # print(tensor)
        return True
    else:
        print("Tensor", tensor_name, "is not finite")
        # print(tensor)
        return False


def check_grad(name):
    def hook(grad):
        check_tensor_finite(grad, name + " grad")
    return hook


def exit_model():
    if not use_tracer():
        return

    exit()


def get_tensors(dump_dir):
    tensors = {}
    for subdir, _, files in os.walk(dump_dir):
        for file in files:
            try:
                tensors[file] = torch.load(os.path.join(
                    subdir, file), map_location=torch.device('cpu'))
            except:
                print("Failed to load", subdir, file)
                pass

    return tensors


def compare(amd_dump_path, nv_dump_path):
    amd_dump = get_tensors(amd_dump_path)
    nv_dump = get_tensors(nv_dump_path)
    print("AMD tensor dump path:", amd_dump_path)
    print("NV tensor dump path:", nv_dump_path)

    for tensor_name in natsorted(amd_dump):
        print(tensor_name + ": ", end="")
        amd_tensor = amd_dump[tensor_name]

        if tensor_name in nv_dump:
            nv_tensor = nv_dump[tensor_name]
        else:
            print(tensor_name, "doesnot exist on NV side")
            continue

        if type(amd_tensor) == tuple:
            amd_tensor = amd_tensor[0]

        if type(nv_tensor) == tuple:
            nv_tensor = nv_tensor[0]

        diff = torch.dist(
            amd_tensor.float(), nv_tensor.float()).item()
        print("diff of", diff)


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('amd_dump')
    parser.add_argument('nv_dump')
    args = parser.parse_args()

    compare(args.amd_dump, args.nv_dump)
