from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="analysis_functions",
        sources=["analysis_functions.pyx"],
    )
]

setup(
    ext_modules=cythonize("analysis_functions.pyx"),
    include_dirs=[np.get_include()]
)