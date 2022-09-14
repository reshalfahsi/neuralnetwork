# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


import numpy as np
from .base import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=np.float64):
        super(Linear, self).__init__()
        self.A = np.random.randn(1, in_features, out_features).astype(dtype)
        self.bias = bias
        if self.bias:
            self.b = np.random.randn(1, 1, out_features).astype(dtype)

    def forward(self, x):
        self.x = x
        if self.bias:
            return np.matmul(self.x, self.A) + self.b
        return np.matmul(self.x, self.A)

    def grad(self, gradient, order="jacobian"):
        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"

        name_gradient_A = None
        if self.bias:
            name_gradient_b = None
        gradient_A = None
        gradient_x = None
        if self.bias:
            gradient_b = None

        if order == "jacobian":
            name_gradient_A = "grad_A"
            if self.bias:
                name_gradient_b = "grad_b"

            gradient_A = np.matmul(np.transpose(self.x, (0, 2, 1)), gradient)
            if self.bias:
                gradient_b = gradient.mean(axis=0, keepdims=True)
            gradient_x = np.matmul(gradient, np.transpose(self.A, (0, 2, 1)))

        else:
            name_gradient_A = "grad2_A"
            if self.bias:
                name_gradient_b = "grad2_b"

            _error_first = gradient["error_first"]
            _error_second = gradient["error_second"]
            _nonlinearity_first = gradient["nonlinearity_first"]
            _nonlinearity_second = gradient["nonlinearity_second"]

            gradient_A = np.matmul(
                np.transpose(self.x, (0, 2, 1)), _nonlinearity_first * _error_second
            ) * np.matmul(np.transpose(self.x, (0, 2, 1)), _nonlinearity_first)
            gradient_A = gradient_A + np.matmul(
                np.transpose((self.x * self.x), (0, 2, 1)),
                _error_first * _nonlinearity_second,
            )
            if self.bias:
                gradient_b = (
                    _error_second * np.square(_nonlinearity_first)
                    + _error_first * _nonlinearity_second
                ).mean(axis=0, keepdims=True)
            gradient_x = np.matmul(
                (_error_second * _nonlinearity_first), np.transpose(self.A, (0, 2, 1))
            ) * np.matmul(_nonlinearity_first, np.transpose(self.A, (0, 2, 1)))
            gradient_x = gradient_x + np.matmul(
                (_error_first * _nonlinearity_second),
                np.transpose(np.square(self.A), (0, 2, 1)),
            )

        setattr(self, name_gradient_A, gradient_A)
        if self.bias:
            setattr(self, name_gradient_b, gradient_b)

        return gradient_x

    def update(self, lr=1e-3, method=None):
        denum_A = 1.0
        if self.bias:
            denum_b = 1.0
        epsilon = 0

        if isinstance(method, str):
            assert (
                method in self._valid_optimizer
            ), f"Invalid optimizer: {method}, expected: 'gradient_descent' or 'newton'"

        if method == "newton":
            assert (
                "grad2_A" in self.__dict__
            ), "Second order derivative has not been calculated yet."
            denum_A = self.grad2_A.mean(axis=0, keepdims=True)
            if self.bias:
                denum_b = self.grad2_b.mean(axis=0, keepdims=True)
            lr = 1.0
            epsilon = 1e-8

        self.A -= lr * self.grad_A.mean(axis=0, keepdims=True) / (denum_A + epsilon)
        if self.bias:
            self.b -= lr * self.grad_b.mean(axis=0, keepdims=True) / (denum_b + epsilon)
