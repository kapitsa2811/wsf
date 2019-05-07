import os
import h5py
import numpy as np
import argparse

from extract_cnn_vgg16_keras import VGGNet

cwd=os.getcwd()+"//"
output = cwd + '1'+"//"  # args["index"]

h5f = h5py.File(output, 'w')