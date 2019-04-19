# coding = utf-8
import tensorflow as tf
import sys
import os

import p3d
from network import *
from datetime import datetime
import cv2
import math
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import time
from SDataset import VideoDataset
import numpy as np
from metrics import CC, SIM, AUC_Judd, KLdiv, NSS, AUC_Borji


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
CROP_SIZE = 112

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='P3D', help="model")
    parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--plotiter', type=int, default=500, help='training mini batch')
    parser.add_argument('--validiter', type=int, default=4000, help='training mini batch')
    parser.add_argument('--model', type=str, default="", help='saved model name')

    parser.add_argument('--trainingexampleprops',type=float, default=0.9, help='training dataset.')
    parser.add_argument('--trainingbase',type=str, default='svsd', help='training dataset.')
    parser.add_argument('--videolength',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=15, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=2, help='length of video')
    parser.add_argument('--imagesize', type=tuple, default=(112,112))
    parser.add_argument('--gpu', type=str, default="0")
    
    parser.add_argument('--extramodinfo', type=str, default='', help="add extra model information")
    return parser.parse_args()

def generate_saliency(pred_patch_smap):
    # pred_patch_l shape: [patch_no, 224, 224, 1]
    boxsize=224
    patches=pred_patch_smap.shape[0]

    pa1 = int(math.ceil(1080 / boxsize)+1)
    pa2 = int(math.ceil(1920 / boxsize)+1)
    final_smap=np.zeros([1080,1920])
    temp_smap=np.zeros([pa1*boxsize,pa2*boxsize])

    patch_no=0
    for i in range(pa1):
        for j in range(pa2):
            temp_patch=pred_patch_smap[patch_no, :, :]
            temp_smap[boxsize*i:boxsize*(i+1), boxsize*j:boxsize*(j+1)]=temp_patch
            final_smap[:,:]=temp_smap[0:1080, 0:1920]
            patch_no=patch_no+1
    return final_smap

def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print ("Loading data...")
LeftPath = '../svsd/test/left_view_svsd/'
RightPath = '../svsd/test/right_view_svsd/'
GtPath = '../svsd/test/view_svsd_density/'

batch_size=args.batch


tranining_dataset = VideoDataset(LeftPath, GtPath, img_size=(CROP_SIZE, CROP_SIZE), video_length=16,  bgr_mean_list=[98,102,90], sort='rgb')
tranining_dataset.setup_video_dataset_c3d(overlap=8, training_example_props=0)

print "data loaded"
net = args.net

frames=16



t = datetime.datetime.now().isoformat()[:-9]
dir_name = net +'_' + t + '/'

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [batch_size, frames, CROP_SIZE, CROP_SIZE, 3])
    y = tf.placeholder(tf.float32, [batch_size, 1, CROP_SIZE, CROP_SIZE])
    training = tf.placeholder(tf.bool)
    _dropout = tf.placeholder(tf.float32)


if net == 'P3D':
    pred=p3d.inference_p3d(x, _dropout, batch_size, training)
elif net == 'P3D_SA': 
    pred=p3d.inference_p3d_sa(x, _dropout, batch_size, training)



with tf.Session(config=config) as sess:
    if int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        merged = tf.merge_all_summaries()
    else:  # tensorflow version >= 0.12
        merged = tf.summary.merge_all()
    logs_dir = './logs/' + dir_name
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter(logs_dir, sess.graph)
    else:  # tensorflow version >= 0.123
        writer = tf.summary.FileWriter(logs_dir, sess.graph)

    if int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    print 'Init variable'
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state("./model/"+args.model)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
    tmp_cc = []; tmp_sim = []; tmp_auc = []; tmp_nss = []; tmp_aucborji = [];
    index = 0
    data_tuple = tranining_dataset.get_frame_c3d_tf(mini_batch=batch_size, phase='validation', density_length='one')
    index = 0
    while data_tuple is not None:
        valid_batch_xs, valid_batch_ys = data_tuple
        index += 1
        image = sess.run(pred, feed_dict={ x: valid_batch_xs, _dropout: 0,  training: False})
        if index % 10 == 0:
            print " Step: %d, Metrics: CC: %.3f  SIM: %.3f   NSS: %.3f  AUC_Judd: %.3f   AUC_Borji: %.3f" \
                % (index, np.mean(tmp_cc), np.mean(tmp_sim),  np.mean(tmp_nss), np.mean(tmp_auc), np.mean(tmp_aucborji))
        for (prediction, ground_truth) in zip(image, valid_batch_ys):
            prediction = np.transpose(np.array(prediction[-1]), (2, 0, 1))
            ground_truth = np.array(ground_truth)
            print np.array(prediction).shape, np.sum(prediction), np.array(ground_truth).shape, np.sum(ground_truth)
            for (preds, gt) in zip(prediction, ground_truth):
                tmp_cc.append(CC(preds, gt))
                tmp_sim.append(SIM(preds, gt))
                tmp_auc.append(AUC_Judd(preds, gt))  
                tmp_nss.append(NSS(preds, gt))
                tmp_aucborji.append(AUC_Borji(preds, gt))
        data_tuple = tranining_dataset.get_frame_c3d_tf(mini_batch=batch_size, phase='validation', density_length='one')
    tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
    tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
    tmp_auc = np.array(tmp_auc)[~np.isnan(tmp_auc)]
    tmp_nss = np.array(tmp_nss)[~np.isnan(tmp_nss)]
    tmp_aucborji = np.array(tmp_aucborji)[~np.isnan(tmp_aucborji)]
    print " Step: %d, Metrics: CC: %.3f  SIM: %.3f   NSS: %.3f  AUC_Judd: %.3f   AUC_Borji: %.3f" \
        % (index, np.mean(tmp_cc), np.mean(tmp_sim),  np.mean(tmp_nss), np.mean(tmp_auc), np.mean(tmp_aucborji))

    print 'Testing Finished!'


