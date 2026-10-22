# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.one_batch import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkOneBatchSubcommand(BenchmarkSubcommandBase):
    """The `one-batch` subcommand for `vllm bench`."""

    name = "one-batch"
    help = "Benchmark prefill and per-step decode latency of one static batch."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
