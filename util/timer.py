import atexit
import time

import numpy as np
import torch

cuda_timers = {}


class CudaTimer:
    def __init__(self, device: torch.device, timer_name: str = ""):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.device = device
        self.start = None
        self.end = None

    def __enter__(self):
        torch.cuda.synchronize(device=self.device)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        assert self.start is not None
        torch.cuda.synchronize(device=self.device)
        end = time.time()
        cuda_timers[self.timer_name].append(end - self.start)