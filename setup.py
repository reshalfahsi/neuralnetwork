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


import sys
import platform
from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# read version
version = {}
version_file_contents = (this_directory / "neuralnetwork" / "version.py").read_text()
exec(version_file_contents, version)


setup(
    name="neuralnetwork",
    version=version["__version__"],
    author="Resha Dwika Hefni Al-Fahsi",
    author_email="resha.alfahsi@gmail.com",
    description="Implementation of Artificial Neural Network Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reshalfahsi/neuralnetwork",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "medmnist",
        "Pillow",
    ],
    package_data={
        "neuralnetwork": [
            "ds/medical_insurance_cost/*.csv",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
    ],
    keywords="machine learning",
)
