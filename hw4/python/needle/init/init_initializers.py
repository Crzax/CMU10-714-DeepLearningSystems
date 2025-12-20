import math
from .init_basic import *
from typing import Any

def _calculate_fans(shape):
  fan_in, fan_out = shape[-2], shape[-1]
  rec = 1
  for dim in shape[:-2]:
    rec *= dim
  fan_in *= rec
  fan_out *= rec
  return (fan_in, fan_out)

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2/(fan_in+fan_out))
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in: int, fan_out: int, shape:tuple = None, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
      fan_in, fan_out = _calculate_fans(shape)
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3/fan_in)
    if shape is None:
      return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
      return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, shape:tuple = None, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
      fan_in, fan_out = _calculate_fans(shape)
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    if shape is None:
      return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    else:
      return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION