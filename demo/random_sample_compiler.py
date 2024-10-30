import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

import argparse
import datetime

import numpy as np
from csstuning.compiler.compiler_benchmark import CompilerBenchmarkBase

from transopt.Benchmark import construct_test_suits
from transopt.KnowledgeBase.kb_builder import construct_knowledgebase
from transopt.KnowledgeBase.TaskDataHandler import OptTaskDataHandler
from optimizer.construct_optimizer import get_optimizer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_experiments(tasks, args):
    kb = construct_knowledgebase(args)
    testsuits = construct_test_suits(tasks, args.seed)
    optimizer = get_optimizer(args)
    data_handler = OptTaskDataHandler(kb, args)
    optimizer.optimize(testsuits, data_handler)


def split_into_segments(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples_num",
        type=int,
        help="Number of samples to be collected for each workload",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        help="Index for splitting the workload segments",
        default=0,
    )
    args = parser.parse_args()
    split_index = args.split_index
    samples_num = args.samples_num

    available_workloads = CompilerBenchmarkBase.AVAILABLE_WORKLOADS
    collected_workloads = [
        "cbench-automotive-susan-c",
        "cbench-automotive-bitcount",
        "cbench-security-rijndael",
        "cbench-consumer-tiff2rgba",
        "cbench-telecom-adpcm-d",
        "cbench-consumer-tiff2bw",
        "cbench-telecom-adpcm-c",
        "cbench-consumer-tiff2dither",
        "cbench-telecom-gsm",
        "cbench-automotive-susan-e",
        "cbench-security-sha",
        "cbench-network-patricia",
        "cbench-telecom-crc32",
        "cbench-security-pgp",
        "cbench-consumer-mad",
        "cbench-automotive-qsort1",
        "polybench-cholesky",
        "polybench-trisolv",
        "polybench-adi",
        "polybench-symm",
        "polybench-gesummv",
        "polybench-gemver",
        "polybench-durbin",
        "polybench-atax",
        "polybench-fdtd-apml",
        "polybench-jacobi-1d-imper",
        "polybench-bicg",
        "polybench-syr2k",
        "polybench-mvt",
        "polybench-lu",
        "polybench-3mm",
    ]
    available_workloads = list(set(available_workloads) - set(collected_workloads))

    split_workloads = split_into_segments(available_workloads, 10)

    if split_index >= len(split_workloads):
        raise IndexError("split index out of range")

    workloads = split_workloads[split_index]

    tasks = {
        "GCC": {"budget": samples_num, "workloads": workloads},
        # "LLVM": {"budget": samples_num, "workloads": workloads},
    }

    # Get date and set exp name
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"sampling_compiler_{date}"

    args = argparse.Namespace(
        seed=0,
        optimizer="ParEGO",
        init_number=2,
        init_method="random",
        exp_path=f"{package_dir}/../experiment_results",
        exp_name=exp_name,
        verbose=True,
        normalize="norm",
        source_num=2,
        selector="None",
        save_mode=1,
        load_mode=False,
        acquisition_func="LCB",
    )

    run_experiments(tasks, args)
