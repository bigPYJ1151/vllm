import os
import time
import torch

from typing import List


def dump_tensors(path: str, *tensors: torch.Tensor):
  torch.save([tensor.cpu() for tensor in tensors], path)

def load_tensors(path: str) -> List[torch.Tensor]:
  return torch.load(path)

class VLLMTensorDumper:
  def __init__(self, path: str, device: torch.device, current_pos: int) -> None:
    self.device = device.type
    self.current_pos = str(current_pos)
    self.time_stamp = str(time.time_ns())
    self.path = os.path.join(path, "_".join([self.device, self.time_stamp, self.current_pos, "{}"]))

  def dump(self, tag: str, tensor: torch.Tensor):
    dump_tensors(self.path.format(tag), tensor)