import torch

from logger import get_logger

LOGGER = get_logger(__file__)

def get_device(force_cpu):
    # Check if CUDA can be used
    LOGGER.info(f"{torch.cuda.is_available()=}, {force_cpu=}")
    if torch.cuda.is_available() and not force_cpu:
        LOGGER.info("CUDA detected. Running with GPU acceleration.")
        device = torch.device("cuda")
    elif force_cpu:
        LOGGER.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        LOGGER.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device
