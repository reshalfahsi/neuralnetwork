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


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def grad(self, x, order="jacobian"):
        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"
        return (
            self(x) * (1.0 - self(x))
            if order == "jacobian"
            else self(x) * (1.0 - self(x)) * (1.0 - 2.0 * self(x))
        )


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        return np.tanh(input)

    def grad(self, x, order="jacobian"):
        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"
        return (
            1.0 - (self(x) * self(x))
            if order == "jacobian"
            else -2.0 * self(x) * (1.0 - (self(x) * self(x)))
        )
