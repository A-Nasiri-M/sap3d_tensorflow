# coding = utf-8

import tensorflow as tf
import sys
import os

from network import *
from datetime import datetime
import cv2
import math
import datetime

from tensorflow.python import pywrap_tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

model_dir = './model/P3D_SA_CONCAT_DECODER_2_0.0001__with_gn_2019-04-11/'


checkpoint_path = os.path.join(model_dir, "p3d_with_gn_svsd_model_76000.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    if 'conv' in key:
        print  key, reader.get_tensor(key).shape