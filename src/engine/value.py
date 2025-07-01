import math
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple, TypeAlias, Union


class OperationEnum(str, Enum):
    """
    Enum representing possible operations that can be performed on Nodes.
    """

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "^"
    SIGMOID = "sigmoid"
    ReLU = "relu"


ParentValues: TypeAlias = Union[Tuple["Value"], Tuple["Value", "Value"]]
OperationType: TypeAlias = Union[OperationEnum, Tuple[OperationEnum, str]]


class Value:
    """
    A class representing a node in a computational graph for automatic differentiation.

    Attributes
        data `float`: The scalar value this node holds.
        grad `float`: The gradient of this node, used during backpropagation
        _previous `Set[Node]`: The parent nodes that this node originates from
        _operation `Optional[OperationType]`: The operation that produced this node
        _label `Optional[str]`: An optional label for the node
        _backward `Callable[[], None]`: A function that defines how the gradient should be propagated
    """

    def __init__(
        self,
        data: Union[int, float],
        previous: Optional[ParentValues] = None,
        operation: Optional[OperationType] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Initializes a new node in the computational graph

        Args
            data `Union[int, float]`: The value held by the node
            previous `Optional[ParentNodes]`: Parent nodes this node depends on
            operation `Optional[OperationType]`: Operation used to compute this node
            label `Optional[str]`: Optional label for the node
        """
        self.data = float(data)
        self.grad = 0.0

        self._previous: Set[Value] = set(previous) if previous is not None else set()
        self._operation = operation
        self._label = label

        self._backward: Callable[[], None] = lambda: None

    def backward(self) -> None:
        """
        Performs backpropagation through the computational graph to compute gradients
        for all nodes that affect the current node.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._previous:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def sigmoid(self) -> "Value":
        """
        Applies the sigmoid activation function to the node

        Returns
            `Node`: A new node representing the sigmoid of this node
        """
        out = Value(1 / (1 + math.exp(-self.data)), (self,), OperationEnum.SIGMOID)

        def _backward() -> None:
            self.grad = out.data * (1 - out.data) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        """
        Applies the ReLU activation function to the node

        Returns
            `Node`: A new node representing the ReLU of this node
        """
        out = Value(0 if self.data <= 0 else self.data, (self,), OperationEnum.ReLU)

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other: Union["Value", int, float]) -> "Value":
        """
        Adds this node to another node or number

        Returns
            `Node`: Resulting node after addition
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), OperationEnum.ADD)

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union["Value", int, float]) -> "Value":
        """
        Multiplies this node with another node or number

        Returns
            `Node`: Resulting node after addition
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), OperationEnum.MULTIPLY)

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        """
        Raises this node to the power of a number.

        Args
            other `Union[int, float]`: The exponent.

        Returns
            `Node`: Resulting node after exponentiation.
        """
        out = Value(self.data**other, (self,), (OperationEnum.POWER, f"{other}"))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> "Value":
        """
        -self
        """
        return self * -1

    def __radd__(self, other: Union["Value", int, float]) -> "Value":
        """
        other + self
        """
        return self + other

    def __sub__(self, other: Union["Value", int, float]) -> "Value":
        """
        self - other
        """
        return self + (-other)

    def __rsub__(self, other: Union["Value", int, float]) -> "Value":
        """
        other - self
        """
        return other + (-self)

    def __rmul__(self, other: Union["Value", int, float]) -> "Value":
        """
        other * self
        """
        return self * other

    def __truediv__(self, other: Union["Value", int, float]) -> "Value":
        """
        self / other
        """
        return self * other**-1

    def __rtruediv__(self, other: Union["Value", int, float]) -> "Value":
        """
        other / self
        """
        return other * self**-1

    def __repr__(self) -> str:
        """
        Returns a string representation of the node, showing its data and gradient.

        Returns
            `str`: The string representation.
        """
        return f"Value(data={self.data}, grad={self.grad})"
