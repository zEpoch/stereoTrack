source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
source /home/HPCBase/tools/module-5.2.0/init/profile.sh 
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.26_gcc9.3.0
conda create -n stereoTrack python=3.10 -y
conda activate stereoTrack
pip install /home/HPCBase/PACKAGE/whl/pytorch/2.4.1+cu118_cp10/torch_scatter-2.1.2-cp310-cp310-linux_aarch64.whl
pip install /home/HPCBase/PACKAGE/whl/pytorch/2.4.1+cu118_cp10/torch_sparse-0.6.18-cp310-cp310-linux_aarch64.whl
pip install /home/HPCBase/PACKAGE/whl/pytorch/2.4.1+cu118_cp10/torch-2.4.1+cuda118-cp310-cp310-linux_aarch64.whl
pip install /home/HPCBase/PACKAGE/whl/pytorch/2.4.1+cu118_cp10/torchaudio-2.4.1+cuda118-cp310-cp310-linux_aarch64.whl
pip install /home/HPCBase/PACKAGE/whl/pytorch/2.4.1+cu118_cp10/torchvision-0.19.1+cuda118-cp310-cp310-linux_aarch64.whl
pip install geomloss
# pip install faiss-gpu-cu11 # error
pip install pot
pip install scanpy
pip install plotly
pip install ipywidgets
pip install pygam
pip install /home/HPCBase/PACKAGE/whl/faiss/faiss-1.8.0-py3-none-any.whl
pip install networkx
pip install numpy
pip install scipy
pip install pandas
pip install nbformat
pip install pysal
# module load libs/faiss/1.8.0
module add  libs/openblas/0.3.26_gcc9.3.0   compilers/cuda/11.8.0  tools/cmake/4.1.0  compilers/gcc/11.3.0   tools/swig/4.4.1
python -c """import os
import ot
import sys
import faiss
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from scipy import stats
import plotly.offline as py
from anndata import AnnData
from ipywidgets import VBox
import multiprocessing as mp
import plotly.graph_objs as go
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from IPython.display import display
from numpy.random import RandomState
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact, norm
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances

sc.settings.verbosity = 0

"""