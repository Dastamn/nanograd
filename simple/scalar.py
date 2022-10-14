from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import numpy as np
from graphviz import Digraph

_SUPPORTED_TYPES = (int, float, np.int32, np.float32)


class Scalar:
    def __init__(self, data: float, _label='', _children: Tuple[Scalar, Scalar] = (), _op=''):
        assert isinstance(
            data, _SUPPORTED_TYPES), f'Cannot create `Scalar` from `{type(data)}`, expected {", ".join(_SUPPORTED_TYPES)}'
        self.data = float(data)
        self.grad = 0.
        self._label = _label
        self._op = _op
        self._children = set(_children)
        self._backward = lambda: None

    def graph(self):
        def _graph(node: Scalar, nodes: Set[Scalar], edges: List[Tuple[Scalar, Scalar]]):
            if node not in nodes:
                nodes.add(node)
                for child in node._children:
                    edges.add((child, node))
                    _graph(child, nodes, edges)
            return nodes, edges
        return _graph(self, set(), set())

    def toposort(self):
        def _toposort(curr: Scalar, visited: Set[Scalar], nodes: List[Scalar]):
            if curr not in visited:
                visited.add(curr)
                [_toposort(i, visited, nodes) for i in curr._children]
                nodes.append(curr)
            return nodes
        return _toposort(self, set(), [])

    def backward(self):
        self.grad = 1.
        [node._backward() for node in reversed(self.toposort())]

    def draw(self):
        G = Digraph(f'{self._label}_graph', format='jpg',
                    graph_attr={'rankdir': 'LR'})
        nodes, edges = self.graph()
        for n in nodes:
            uid = str(id(n))
            label = f'{n._label} |' if n._label else ''
            grad = f' | grad={n.grad:.3f}' if n.grad else ''
            G.node(name=uid, label='{ %s data=%.3f%s}' %
                   (label, n.data, grad), shape='record')
            if n._op:
                G.node(name=uid+n._op, label=n._op)
                G.edge(uid+n._op, uid)
        for n1, n2 in edges:
            G.edge(str(id(n1)), str(id(n2)) + n2._op)
        return G

    def sigmoid(self, label=''):
        x = self.data
        e = math.exp(-x)
        out = Scalar(1/(1+e), label, (self,), 'sigmoid')

        def _backward():
            self.grad += e*(1+e)**-2
        self._backward = _backward
        return out

    def tanh(self, _label=''):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Scalar(t, _label, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self, _label=''):
        out = Scalar(max(self.data, 0), _label, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self, _label=''):
        x = self.data
        out = Scalar(math.exp(x), _label, (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def add(self, x: Scalar, _label=''):
        _x = self + x
        _x._label = _label
        return _x

    def sub(self, x: Scalar, _label=''):
        _x = self - x
        _x._label = _label
        return _x

    def mult(self, x: Scalar, _label=''):
        _x = self * x
        _x._label = _label
        return _x

    def div(self, x: Scalar, _label=''):
        _x: Scalar = self / x
        _x._label = _label
        return _x

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)), f"Expected exponent to be 'int' or 'float', got '{type(other)}'."
        out = Scalar(self.data**other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += (other*self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    def __repr__(self) -> str:
        label = self._label if self._label else 'Scalar'
        grad = f', grad={self.grad:.3f}' if self.grad else ''
        return f'{label}({self.data}{grad})'
