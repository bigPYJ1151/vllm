import os
import pickle
from typing import Any, Dict, List, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform


class CpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_cpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        instance_identifier = os.environ["VLLM_DIST_IDENT"]

        group_ranks = [
            str(rank) for rank in dist.get_process_group_ranks(self.group)
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
        torch.distributed.barrier(self.group)
        torch.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        torch.distributed.barrier(self.group)

        return handle

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # torch.ops._C.shm_allreduce(self.handle, input_)
        torch.distributed.all_reduce(input_, group=self.group)
        return input_

    def gather(self,
               input_: torch.Tensor,
               rank_in_group: int,
               ranks: List[int],
               dst: int = 0,
               dim: int = -1):
        # Allocate output tensor.
        if rank_in_group == dst:
            gather_list = [
                torch.empty_like(input_) for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        # Gather.
        # Note: different from the torch gather, here we use local dst rank.
        torch.ops._C.shm_gather(self.handle, input_, gather_list, dst)

        if rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None

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
