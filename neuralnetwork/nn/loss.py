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


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        assert len(input.shape) == len(target.shape)
        self.input = input
        self.target = target
        res = (target - input) ** 2
        self.loss = res.mean()
        return self.loss

    def grad(self, orde="jacobian"):
        assert orde in self._valid_orde, f"Invalid orde: {orde}, expected 'jacobian' or 'hessian'"
        return -2 * (self.target - self.input) if orde == 'jacobian' else 2.
