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


from .base import Module
import numpy as np


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape
        self.input = input
        self.target = target
        res = (target - input) ** 2
        self.loss = res.mean()
        return self.loss

    def grad(self, order="jacobian"):
        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"
        return -2 * (self.target - self.input) if order == "jacobian" else 2.0


class BCELoss(Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.epsilon = 1.4012985e-45  # for numeric stability

    def forward(self, input, target):
        self.input = input
        self.target = target
        self.m = input.shape[-1]
        res = (1 / self.m) * np.sum(
            -target * np.log(input + self.epsilon)
            - (1 - target) * np.log(1 - (input + self.epsilon))
        )
        self.loss = res.mean()
        return self.loss

    def grad(self, order="jacobian"):

        assert (
            order in self._valid_order
        ), f"Invalid order: {order}, expected 'jacobian' or 'hessian'"
        return (
            (1 / self.m)
            * (
                -(self.target / (self.input + self.epsilon))
                + ((1.0 - self.target) / ((1.0 - self.input) + self.epsilon))
            )
            if order == "jacobian"
            else (1 / self.m)
            * (self.input * self.input - 2.0 * self.input * self.target + self.target)
            / (
                ((1 - self.input) * (1 - self.input) * self.input * self.input)
                + self.epsilon
            )
        )
