from tensorflow.python.client import device_lib


def gpu_enable():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
    if len(gpus) > 0:
        return True
    else:
        return False
