import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def benchmark_module(
    module: Callable,
    *args: Any,
    warmup_runs: int = 10,
    num_runs: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmarks a PyTorch module on GPU, returning performance metrics.
    Handles moving the module and inputs to the GPU, performs warm-up runs,
    and measures performance using CUDA events for accurate timing.

    Args:
        module: The PyTorch module or compiled function to benchmark.
        *args: Positional arguments for the module's forward pass.
        warmup_runs: Number of warm-up iterations before benchmarking.
        num_runs: Number of iterations to run the benchmark.
        **kwargs: Keyword arguments for the module's forward pass.

    Returns:
        A dictionary containing benchmark results, including:
        - "device": The torch.device used for the benchmark.
        - "avg_time_ms": Average forward pass time in milliseconds.
        - "throughput": Throughput in samples per second.
        - "output_shape": The shape of the module's output, if it's a Tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def move_to_device(data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return type(data)(move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: move_to_device(value) for key, value in data.items()}
        return data

    if isinstance(module, torch.nn.Module):
        module = module.to(device)
    args = move_to_device(args)
    kwargs = move_to_device(kwargs)

    if device.type == "cpu":
        logger.warning("Benchmarking on CPU. GPU is not available.")
        import time

        # Warm-up runs
        for _ in range(warmup_runs):
            _ = module(*args, **kwargs)

        start_time = time.perf_counter()
        output = None
        for _ in range(num_runs):
            output = module(*args, **kwargs)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_runs
    else:
        # GPU benchmarking with CUDA events
        for _ in range(warmup_runs):
            _ = module(*args, **kwargs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        output = None
        start_event.record()
        for _ in range(num_runs):
            output = module(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / (1000 * num_runs)

    if output is None:
        raise ValueError("Benchmarking failed; output was not generated.")

    def find_first_tensor(*args: Any, **kwargs: Any) -> torch.Tensor | None:
        items = list(args) + list(kwargs.values())
        for item in items:
            if isinstance(item, torch.Tensor):
                return item
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        return sub_item
        return None

    first_tensor = find_first_tensor(*args, **kwargs)
    batch_size = first_tensor.shape[0] if first_tensor is not None else 1

    return {
        "device": device,
        "avg_time_ms": avg_time * 1000,
        "throughput": batch_size / avg_time,
        "output_shape": output.shape if isinstance(output, torch.Tensor) else None,
    }
