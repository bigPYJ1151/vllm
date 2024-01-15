from threadpoolctl import threadpool_info
from pprint import pprint

import torch
from benchmark import KernelBenchmark

class RMSNormBench(KernelBenchmark):

    def __init__(self, loop_time, token_num, hidden_size, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__(loop_time)
        self.x = torch.empty(token_num,
                             hidden_size,
                             dtype=dtype,
                             device=device)
        self.out = torch.empty_like(self.x)
        self.weight = torch.empty(hidden_size,
                                  hidden_size,
                                dtype=dtype,
                                device=device)

    def _run(self):
        self.out = self.x @ self.weight


bench = RMSNormBench(10, 256, 4096, torch.float32, torch.device("cpu"))
bench.execute()

bench = RMSNormBench(10, 256, 4096, torch.int8, torch.device("cpu"))
bench.execute()
pprint(threadpool_info())

# RMSNormBench(10, 4096, 4096, torch.float32, torch.device("cpu"))
# Scalar: 282420151.5 ns
# token parallel: 36635991.875 ns 7.7x
# FMA: 36517116.125 ns
