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
        self.A = np.random.randn(in_features, out_features).astype(dtype)
        self.bias = bias
        if self.bias:
            self.b = np.random.randn(1, out_features).astype(dtype)

    def forward(self, x):
        self.x = x
        if self.bias:
            return self.x.dot(self.A) + self.b
        return self.x.dot(self.A)

    def grad(self, gradient):

        name_gradient_A = "grad_A"
        name_gradient_b = "grad_b"

        gradient_A = self.x.T.dot(gradient)
        if self.bias:
            gradient_b = gradient.mean(axis=0, keepdims=True)
        gradient_x = gradient.dot(self.A.T)

        setattr(self, name_gradient_A, gradient_A)
        if self.bias:
            setattr(self, name_gradient_b, gradient_b)
        return gradient_x

    def update(self, lr):
        self.A -= lr * self.grad_A
        if self.bias:
            self.b -= lr * self.grad_b
