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
import warnings



class MedicalInsuranceCost:
    def __init__(self, split=None):
        assert split != None, "Please specify the split: 'train' or 'test'"

        self.dataset = pd.read_csv(
            os.path.join(str(os.path.dirname(os.path.realpath(__file__))), "dataset.csv")
        )

        self.dataset["smoker"] = list(
            map(lambda x: 1 if x == "yes" else 0, self.dataset["smoker"])
        )

        warnings.filterwarnings("ignore")

        mean_age = self.dataset.mean()["age"]
        mean_bmi = self.dataset.mean()["bmi"]
        self.mean_charges = self.dataset.mean()["charges"]

        std_age = self.dataset.std()["age"]
        std_bmi = self.dataset.std()["bmi"]
        self.std_charges = self.dataset.std()["charges"]

        self.dataset = self.dataset[:2904] if split == "train" else self.dataset[2904:]

        self.dataset["age"] = list(
            map(lambda x: (x - mean_age) / std_age, self.dataset["age"])
        )
        self.dataset["bmi"] = list(
            map(lambda x: (x - mean_bmi) / std_bmi, self.dataset["bmi"])
        )
        self.dataset["charges"] = list(
            map(
                lambda x: (x - self.mean_charges) / self.std_charges,
                self.dataset["charges"],
            )
        )

    def final_pred(self, x):
        """Get the the real value of prediction."""
        return (x * self.std_charges) + self.mean_charges

    def __len__(self):
        return len(self.dataset)

    def head(self, n=5):
        print(self.dataset.head(n))

    def __getitem__(self, index):
        """
        Why these features? `Check the EDA on this Kaggle notebook
        <https://www.kaggle.com/code/deepgupta4023/eda-and-simple-gradient-boosting>`__ .
        """
        X = self.dataset[["age", "bmi", "smoker"]].iloc[index].tolist()
        Y = self.dataset["charges"].iloc[index].tolist()

        return X, Y

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(np.array(x))
            ys.append(y)
        return np.array(xs), np.array(ys)
