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
        self.epsilon = 1.4012985e-45  # for numeric stability

    def forward(self, input):
        return 1.0 / np.clip(1.0 + np.exp(-input), self.epsilon, 1.)

    def grad(self, x, order="jacobian"):
        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"
        return (
            self(x) * (1.0 - self(x))
            if order == "jacobian"
            else self(x) * (1.0 - self(x)) * (1.0 - 2.0 * self(x))
        )
