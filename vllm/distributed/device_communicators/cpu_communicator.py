import os
import pickle
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        instance_identifier = os.environ["VLLM_DIST_IDENT"]

        group_ranks = [
            str(rank) for rank in self.ranks
        ]
        shm_group_identifier = f"[{'-'.join(group_ranks)}]"
        self.group_name = f"{instance_identifier}-{shm_group_identifier}-cpushm"

        self.handle = self._init_cpu_shm()

    def _init_cpu_shm(self) -> int:
        handle = torch.ops._C.init_shm_manager(
            self.group_name,
            self.world_size,
            self.rank,
        )
        torch.distributed.barrier(self.device_group)
        torch.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        torch.distributed.barrier(self.device_group)

        return handle

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        torch.ops._C.shm_allreduce(self.handle, input_)
        # torch.distributed.all_reduce(cpy_input, group=self.device_group)
        return input_

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        # Gather.
        # Note: different from the torch gather, here we use local dst rank.
        torch.ops._C.shm_gather(self.handle, input_, gather_list, dst)

        # torch.distributed.gather(input_,
        #                     gather_list,
        #                     dst=self.ranks[dst],
        #                     group=self.device_group)

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        torch.ops._C.shm_all_gather(self.handle, input_, output_tensor)
        # dist.all_gather_into_tensor(cp_output,
        #                             input_,
        #                             group=self.device_group)

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: int,
    ) -> None:
        key_list = list(tensor_dict.keys())
        value_list = list(tensor_dict.values())
        size_list = []
        for v in value_list:
            if not isinstance(v, torch.Tensor):
                raise RuntimeError(
                    "CpuCommunicator only supports sending tensors.")
            size_list.append(v.size())
        key_size_tensor = torch.frombuffer(pickle.dumps([key_list, size_list]),
                                           dtype=torch.uint8)
        value_list.append(key_size_tensor)

        torch.ops._C.shm_send_tensor_list(self.handle, value_list, dst)

        return None

    def recv_tensor_dict(
        self,
        src: int,
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        tensor_list = torch.ops._C.shm_recv_tensor_list(self.handle, src)

        value_list: List[torch.Tensor] = tensor_list[:-1]
        key_size_tensor = tensor_list[-1]

        key_size = pickle.loads(key_size_tensor.numpy().tobytes())
        key_list = key_size[0]
        size_list = key_size[1]
        assert len(key_list) == len(size_list)
        assert len(key_list) == len(value_list)

        tensor_dict: Dict[str, torch.Tensor] = {}
        for key, size, t in zip(key_list, size_list, value_list):
            tensor_dict[key] = t.view(size)
        return tensor_dict
