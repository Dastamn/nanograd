import random
from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np

from scalar import Scalar


class SimpleModule(ABC):

    @abstractmethod
    def parameters(self) -> List[Scalar]:
        pass


class ScalarNeuron(SimpleModule):

    def __init__(self, nin: int, low=-1, high=1):
        assert low < high
        self.w = [Scalar(random.uniform(low, high)) for _ in range(nin)]
        self.b = Scalar(random.uniform(low, high))

    def __call__(self, x: List[Scalar], act: Literal['tanh', 'sigmoid', 'relu'] = 'tanh'):
        assert isinstance(x, (list, np.ndarray))
        if not len(self.w) == len(x):
            raise ValueError(
                f'Input size mismatch, expected: ({len(self.w)}) got ({len(x)}).')
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if act == 'tanh':
            out = out.tanh()
        elif act == 'sigmoid':
            out = out.sigmoid()
        elif act == 'relu':
            out = out.relu()
        else:
            raise ValueError(f'Unknown activation function `{act}`.')
        return out

    def parameters(self):
        return self.w + [self.b]


class ScalarLayer(SimpleModule):

    def __init__(self, nin: int, nout: int):
        self.neurons = [ScalarNeuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Scalar]):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class ScalarMLP(SimpleModule):

    def __init__(self, nin: int, nouts: List[int]):
        self.layers = [ScalarLayer(*pair)
                       for pair in zip([nin] + nouts, nouts)]

    def __call__(self, x: List[Scalar]):
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
