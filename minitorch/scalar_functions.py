from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given scalar values, storing history for backpropagation.

        Args:
        ----
        *vals: Input values, which can be either Scalar objects or raw numbers.

        Returns:
        -------
        A Scalar object with the result of the function applied, with history for backpropagation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass of the addition function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: First scalar value.
        b: Second scalar value.

        Returns:
        -------
        The result of adding a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Perform backward pass of the addition function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        A tuple containing the gradients of a and b with respect to the output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the logarithmic function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: Scalar value.

        Returns:
        -------
        The result of applying the logarithm to a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the logarithmic function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of a with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass of the multiplication function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: First scalar value.
        b: Second scalar value.

        Returns:
        -------
        The result of multiplying a and b.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass of the multiplication function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        A tuple containing the gradients of a and b with respect to the output.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the inverse function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: Scalar value.

        Returns:
        -------
        The result of 1/a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the inverse function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of the input a with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the negation function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        a: Scalar value.

        Returns:
        -------
        The result of -a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the negation function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of the input a with respect to the output.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the sigmoid function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: Scalar value.

        Returns:
        -------
        The result of the sigmoid function applied to a.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the sigmoid function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of the input a with respect to the output.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the ReLU function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: Scalar value.

        Returns:
        -------
        The result of the ReLU function applied to a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the ReLU function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of the input a with respect to the output.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass of the exponential function.

        Args:
        ----
        ctx: Context object used to store information for backpropagation.
        a: Scalar value.

        Returns:
        -------
        The result of exp(a).

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass of the exponential function.

        Args:
        ----
        ctx: Context object storing the values saved during the forward pass.
        d_output: The gradient of the output.

        Returns:
        -------
        The gradient of the input a with respect to the output.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1$ if $x < y$ else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass of the less-than function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        a: First scalar value.
        b: Second scalar value.

        Returns:
        -------
        1.0 if a < b, otherwise 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass of the less-than function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        d_output: The gradient of the output.

        Returns:
        -------
        A tuple of gradients, which is (0.0, 0.0) since no gradient is needed for comparison.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1$ if $x == y$ else $0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass of the equality function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        a: First scalar value.
        b: Second scalar value.

        Returns:
        -------
        1.0 if a == b, otherwise 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass of the equality function.

        Args:
        ----
        ctx: Context object (not used in this operation).
        d_output: The gradient of the output.

        Returns:
        -------
        A tuple of gradients, which is (0.0, 0.0) since no gradient is needed for comparison.

        """
        return 0.0, 0.0
