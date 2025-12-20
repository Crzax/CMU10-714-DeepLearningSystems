from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return Z - array_api.broadcast_to(logsumexp(Tensor(Z),axes = (1,)).realize_cached_data().reshape((Z.shape[0],1)),Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=1,keepdims=True)
        exp_z = exp(z-max_z)

        before_broad_shape = list(z.shape)
        before_broad_shape[1] = 1
        softmax_z = exp_z / summation(exp_z, axes=1).reshape(tuple(before_broad_shape)).broadcast_to(z.shape)
        out_grad_sum = summation(out_grad, axes=1).reshape((z.shape[0],1)).broadcast_to(z.shape)
        return (out_grad + (-softmax_z) * out_grad_sum,)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        maxzplus = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z-maxz),axis=self.axes)) + maxzplus
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=self.axes,keepdims=True)
        exp_z = exp(z-max_z)

        before_broad_shape = list(z.shape)
        axes = range(len(before_broad_shape)) if self.axes == None else self.axes
        for axis in axes:
          before_broad_shape[axis] = 1
        partial_z = exp_z / summation(exp_z, axes=self.axes).reshape(tuple(before_broad_shape)).broadcast_to(z.shape)
        return (partial_z * out_grad.reshape(tuple(before_broad_shape)).broadcast_to(z.shape),)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)