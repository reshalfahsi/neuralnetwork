# MIT License
#
# Copyright (c) 2023 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ==============================================================================


import pandas as pd
import numpy as np
import os


class MedicalInsuranceCost:
    def __init__(self, split=None):
        assert split != None, "Please specify the split: 'train' or 'test'"
        DATASET_FILENAME = "train.csv" if split == "train" else "test.csv"
        self.dataset = pd.read_csv(os.path.join(str(os.getcwd()), DATASET_FILENAME))

        self.dataset["smoker"] = list(
            map(lambda x: 1 if x == "yes" else 0, df["smoker"])
        )

    def __len__(self):
        return len(self.dataset)

    def head(self):
        self.dataset.head()

    def __getitem__(self, index):
        """
        Why these features? `Check the EDA on this Kaggle notebook
        <https://www.kaggle.com/code/deepgupta4023/eda-and-simple-gradient-boosting>`__ .
        """
        X = self.dataset[["age", "bmi", "smoker"]].iloc[index].values.tolist()
        Y = self.dataset["charges"].iloc[index].values.tolist()

        return X, Y

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(np.array(x))
            ys.append(y)
        return np.array(xs), np.array(ys)
