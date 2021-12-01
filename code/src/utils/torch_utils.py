"""Common utility functions.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tensorly.decomposition import partial_tucker
from torch import nn
from torch.utils.data import Subset


def convert_model_to_torchscript(
        model: nn.Module, path: Optional[str] = None
) -> torch.jit.ScriptModule:
    """Convert PyTorch Module to TorchScript.

    Args:
        model: PyTorch Module.

    Return:
        TorchScript module.
    """
    model.eval()
    jit_model = torch.jit.script(model)

    if path:
        jit_model.save(path)

    return jit_model


def split_dataset_index(
        train_dataset: torch.utils.data.Dataset, n_data: int, split_ratio: float = 0.1
) -> Tuple[Subset, Subset]:
    """Split dataset indices with split_ratio.

    Args:
        n_data: number of total data
        split_ratio: split ratio (0.0 ~ 1.0)

    Returns:
        SubsetRandomSampler ({split_ratio} ~ 1.0)
        SubsetRandomSampler (0 ~ {split_ratio})
    """
    indices = np.arange(n_data)
    split = int(split_ratio * indices.shape[0])

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

    return train_subset, valid_subset


def save_model(model, path, data, device):
    """save model to torch script, onnx."""
    try:
        torch.save(model.state_dict(), f=path)
        ts_path = os.path.splitext(path)[:-1][0] + ".ts"
        convert_model_to_torchscript(model, ts_path)
    except Exception:
        print("Failed to save torch")


def model_info(model, verbose=False):
    """Print out model info."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    print(
        f"Model Summary: {len(list(model.modules()))} layers, "
        f"{n_p:,d} parameters, {n_g:,d} gradients"
    )


@torch.no_grad()
def check_runtime(
        model: nn.Module, img_size: List[int], device: torch.device, repeat: int = 100
) -> float:
    repeat = min(repeat, 20)
    img_tensor = torch.rand([1, *img_size]).to(device)
    measure = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.eval()
    for _ in range(repeat):
        start.record()
        _ = model(img_tensor)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        measure.append(start.elapsed_time(end))

    measure.sort()
    n = len(measure)
    k = int(round(n * (0.2) / 2))
    trimmed_measure = measure[k + 1: n - k]

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        _ = model(img_tensor)
    print(prof)
    print("measured time(ms)", np.mean(trimmed_measure))
    model.train()
    return np.mean(trimmed_measure)


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def autopad(
        kernel_size: Union[int, List[int]], padding: Union[int, None] = None
) -> Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    # Pad to 'same'
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    return padding or [x // 2 for x in kernel_size]


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.type = act_type
        self.args = [1] if self.type == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.type is None:
            return nn.Identity()
        elif hasattr(nn, self.type):
            return getattr(nn, self.type)(*self.args)
        else:
            return getattr(
                __import__("src.modules.activations", fromlist=[""]), self.type
            )()


def tucker_decomposition_conv_layer(
        layer: nn.Module,
        normed_rank: List[int] = [0.5, 0.5],
) -> nn.Module:
    """Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    The ranks are estimated with a Python implementation of VBMF
    https://github.com/CasvandenBogaard/VBMF
    """
    if hasattr(layer, "rank"):
        normed_rank = getattr(layer, "rank")
    rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)]  # output channel * normalized rank
    rank = [max(r, 2) for r in rank]

    core, [last, first] = partial_tucker(
        layer.weight.data,
        modes=[0, 1],
        n_iter_max=2000000,
        rank=rank,
        init="svd",
    )

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = (
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    )
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def decompose(module: nn.Module):
    """Iterate model layers and decompose"""
    model_layers = list(module.children())
    if not model_layers:
        return None
    for i in range(len(model_layers)):
        if type(model_layers[i]) == nn.Sequential:
            decomposed_module = decompose(model_layers[i])
            if decomposed_module:
                model_layers[i] = decomposed_module
        if type(model_layers[i]) == nn.Conv2d:
            model_layers[i] = tucker_decomposition_conv_layer(model_layers[i])
    return nn.Sequential(*model_layers)


if __name__ == "__main__":
    # test
    check_runtime(None, [32, 32] + [3])
