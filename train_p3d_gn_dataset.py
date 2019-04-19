# -*- coding:utf-8 -*-
import tensorflow as tf
import sys
import os
import p3d_gn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network import *
import argparse
# from datetime import datetime
import cv2
import datetime
import time
from tensorpack.dataflow import *
from tensorpack.dataflow.imgaug import *
from dataflow import VideoDataset, ImageFromFile, mapf
import numpy as np
from metrics import CC, SIM, AUC_Judd
from tensorpack.input_source import QueueInput, BatchQueueInput, StagingInput
# from tensorflow.python.client import timeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
CROP_SIZE = 112

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='P3D', help="P3D, P3D_SA, P3D_RSA, P3D_CONCAT, P3D_SA_CONCAT, P3D_SA_CONCAT_DECODER")
    parser.add_argument('--plotiter', type=int, default=10000, help='training mini batch')
    parser.add_argument('--validiter', type=int, default=4000, help='training mini batch')
    parser.add_argument('--savemodeliter', type=int, default=10000, help='training mini batch')
    parser.add_argument('--pretrain', type=str, default=None, help='finetune using SGD')

    parser.add_argument('--trainingexampleprops',type=float, default=0.9, help='training dataset.')
    parser.add_argument('--trainingbase',type=str, default='svsd', help='training dataset.')
    parser.add_argument('--videolength',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=15, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=2, help='length of video')
    parser.add_argument('--imagesize', type=tuple, default=(112,112))
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--info', type=str, default='_with_gn', help="add extra model information")
    return parser.parse_args()

def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)

def output_message(args):
    print "############################################################"
    print "  Output the Training Config  "
    print "############################################################"

    print "Net                   : ", args.net
    print "Batch Size            : ", args.batch
    print "Input Tensor          :  (%d, %d, %d, %d, 3)" % (args.batch, args.videolength, args.imagesize[0], args.imagesize[1])
    print "Initial Learning Rate : ", args.lr
    print "Loss                  :  Smooth L1 Loss + L2 Reg"
    print "Optimizer             :  Momentum" 
    print "Training datasets     : ", args.trainingbase
    print "Training Example Prop : ", args.trainingexampleprops
    print "Valid Iter            : ", args.validiter
    print "Save Iter             : ", args.savemodeliter
    print "GPU                   : ", args.gpu
    print ""
    print "############################################################"

print "Parsing arguments..."
args = get_arguments()
output_message(args)




os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print ("Loading data...")
LeftPath = '../svsd/train/left_view_svsd/'
RightPath = '../svsd/train/right_view_svsd/'
GtPath = '../svsd/train/left_density_svsd/'

# train_frame_basedir = '../svsd/test/left_view_svsd'
# train_density_basedir = '../svsd/test/view_svsd_density/'
videodataset = VideoDataset(LeftPath,GtPath, video_length=16, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
videodataset.setup_video_dataset_p3d(overlap=args.overlap, training_example_props=0.9)
videodataset.get_frame_p3d_tf()
df = ImageFromFile(videodataset.final_train_list)
df = MultiThreadMapData(
    df, nr_thread=16,
    map_func=mapf,
    buffer_size=1000,
    strict=True)
df = BatchData(df, args.batch, remainder=False, use_list=True)
df = PrefetchDataZMQ(df, nr_proc=1)
df = RepeatedData(df, -1)
df.reset_state()

# valid
gt_df = ImageFromFile(videodataset.final_valid_list)
gt_df = MultiThreadMapData(
    gt_df, nr_thread=16,
    map_func=mapf,
    buffer_size=1000,
    strict=True)
gt_df = BatchData(gt_df, args.batch, remainder=False, use_list=True)
gt_df = PrefetchDataZMQ(gt_df, nr_proc=1)
gt_df.reset_state()

# queue
# queue = tf.FIFOQueue(100, dtypes=tf.float32)
# queue_i = QueueInput(df)

# print queue_i._set
# 78770 19693
print "data and lable loaded"
net = args.net
validation_iter=args.validiter
plot_iter = args.plotiter
plot_dict = {
    'x':[], 
    'x_valid':[], 
    'y_loss':[], 
    'y_cc':[], 
    'y_sim':[], 
    'y_auc':[]
}

plt.subplot(4, 1, 1)
plt.plot(plot_dict['x'], plot_dict['y_loss'])
plt.ylabel('loss')
plt.subplot(4, 1, 2)
plt.plot(plot_dict['x_valid'], plot_dict['y_cc'])
plt.ylabel('cc metric')
plt.subplot(4, 1, 3)
plt.plot(plot_dict['x_valid'], plot_dict['y_sim'])
plt.ylabel('sim metric')
plt.subplot(4, 1, 4)
plt.plot(plot_dict['x_valid'], plot_dict['y_auc'])
plt.xlabel('iter')
plt.ylabel('auc metric')

t = datetime.datetime.now().isoformat()[:-16]
dir_name = net + '_' + str(args.batch) + '_' + str(args.lr) + '_' + args.info +'_'+ t + '/'
plot_figure_dir = './figure/' + dir_name
model_save_dir = './model/' + dir_name
logs_dir = './logs/' + dir_name
mkDir(logs_dir)
mkDir(plot_figure_dir)
mkDir(model_save_dir)



def train_STSM():

    batch_size = args.batch  # train images=batch_size*frames
    frames = 16  # 3 frames
    Dataset='svsd'
    with tf.device('/cpu:0'):
        # global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [batch_size, frames, CROP_SIZE, CROP_SIZE, 3])
            y = tf.placeholder(tf.float32, [batch_size, 16, CROP_SIZE, CROP_SIZE])
            y_valid = tf.placeholder(tf.float32, [batch_size, 1, CROP_SIZE, CROP_SIZE])
            training = tf.placeholder(tf.bool)
            _dropout = tf.placeholder(tf.float32)
    
    
    if net == 'P3D':
        pred=p3d_gn.inference_p3d(x, _dropout, batch_size, training)
    elif net == 'P3D_CONCAT':
        pred=p3d_gn.inference_p3d_concat(x, _dropout, batch_size, training)
    elif net == 'P3D_SA_CONCAT':
        pred=p3d_gn.inference_p3d_sa_concat(x, _dropout, batch_size, training)
    elif net == 'P3D_SA_CONCAT_2':
        pred=p3d_gn.inference_p3d_sa_concat_2(x, _dropout, batch_size, training)
    elif net == 'P3D_DECODER':
        pred=p3d_gn.inference_p3d_decoder_block(x, _dropout, batch_size, training)
    elif net == 'P3D_SA_DECODER':
        pred=p3d_gn.inference_p3d_sa_decoder_block(x, _dropout, batch_size, training)
    print "built model."

    pred_reshape = tf.reshape(pred, [batch_size, 16, CROP_SIZE, CROP_SIZE])

    with tf.name_scope('loss'):
        loss1 = smooth_l1_loss(pred_reshape, y, 1, 1, sigma=1.0, dim=[-1])
        # wd_loss = 0
        # wd_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))
        # l2_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'P3D'))
        total_loss = loss1
        tf.summary.scalar('loss', total_loss)

    print "built loss"
    with tf.name_scope('train'):
        lr = args.lr
        optimizer=tf.train.AdamOptimizer(lr).minimize(total_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op=tf.group(optimizer)
            
    print "built ops"
    with tf.Session(config=config) as sess:
        if int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
            merged = tf.merge_all_summaries()
        else:  # tensorflow version >= 0.12
            merged = tf.summary.merge_all()
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
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        print 'Init variable'
        sess.run(init)

        # fine-tuning
        if args.pretrain:
            print args.pretrain, "Using this model to retrain..."
            ckpt = tf.train.get_checkpoint_state('./model/'+args.pretrain+'/')
            if ckpt and ckpt.model_checkpoint_path:
                print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("load complete!")
                # exit()

        print 'Start training'
        step = 0
        epoch = 4
        training_iters = epoch * 78880 / batch_size
        # training_iters = 30
        data_time = 0
        for data in df:
            step = step + 1
            if step >= training_iters:
                break
            train_batch_xs, train_batch_ys = data
            _, train_loss= sess.run([train_op, total_loss],
                     feed_dict={x: train_batch_xs, y: train_batch_ys, _dropout: 0.5, training: True})
            plot_dict['x'].append(step)
            plot_dict['y_loss'].append(train_loss)
            if step<50 or step % 1000 == 0:
                # validation
                image = sess.run(pred,
                                feed_dict={ x: train_batch_xs, _dropout: 0, training: False})
                image1 = np.reshape(image, [-1, 16, CROP_SIZE, CROP_SIZE]) 

                save_image1 = np.zeros([CROP_SIZE, CROP_SIZE])
                save_image2 = np.zeros([CROP_SIZE, CROP_SIZE])
                save_image1[:, :] = image1[0, -1, :, :]*255.
                save_image2 = train_batch_ys[0][-1]*255.
                final_save_image1 = save_image1
                final_save_image2 = save_image2
                smap_save_path = os.path.join(plot_figure_dir, 'smap_Result')
                mkDir(smap_save_path)
                smap_save_name = smap_save_path + '/' +  'step_' + str(step) + '_pred.jpg'
                cv2.imwrite(smap_save_name, final_save_image1)
                smap_save_name2 = smap_save_path + '/' +  'step_' + str(step) + '_gt.jpg'
                cv2.imwrite(smap_save_name2, final_save_image2)
    
                print 'Datetime', datetime.datetime.now().isoformat()[:-7],'Training step:', step, np.sum(final_save_image1), np.sum(final_save_image2), 'Training Loss', train_loss

            if (step % validation_iter==0):
                print "Doing validation..."
                tmp_cc = []; tmp_sim = []; tmp_auc = []
                index = 0
                for valid_data in gt_df:
                    valid_batch_xs, valid_batch_ys = valid_data
                    index += 1
                    image0 = sess.run(pred, feed_dict={ x: valid_batch_xs, _dropout: 0, training: False})
                    image0 = np.reshape(image0, [-1, 16, CROP_SIZE, CROP_SIZE])
                    for (prediction, ground_truth) in zip(image0, valid_batch_ys):
                        prediction = np.array(prediction[-1])
                        ground_truth = np.array(ground_truth[-1])
                        if index % 1000 == 0:
                            print datetime.datetime.now(), ' Index', index, 'pred:', np.sum(prediction), 'gt:', np.sum(ground_truth)
                        tmp_cc.append(CC(prediction, ground_truth))
                        tmp_sim.append(SIM(prediction, ground_truth))
                        tmp_auc.append(AUC_Judd(prediction, ground_truth))  
                tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
                tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
                tmp_auc = np.array(tmp_auc)[~np.isnan(tmp_auc)]
                print datetime.datetime.now().isoformat()[:-7], " Step:", step, " Metrics:", np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_auc)
                plot_dict['x_valid'].append(step)
                plot_dict['y_cc'].append(np.mean(tmp_cc))
                plot_dict['y_sim'].append(np.mean(tmp_sim))
                plot_dict['y_auc'].append(np.mean(tmp_auc))
            
            if step%plot_iter==0:
                    plot_xlength=500
                    plt.subplot(4, 1, 1)
                    plt.plot(plot_dict['x'][-plot_xlength:], plot_dict['y_loss'][-plot_xlength:])
                    plt.ylabel('loss')
                    plt.subplot(4, 1, 2)
                    plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_cc'][-plot_xlength:])
                    plt.ylabel('cc metric')
                    plt.subplot(4, 1, 3)
                    plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_sim'][-plot_xlength:])
                    plt.ylabel('sim metric')
                    plt.subplot(4, 1, 4)
                    plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_auc'][-plot_xlength:])
                    plt.xlabel('iter')
                    plt.ylabel('auc metric')
                    plt.savefig(os.path.join(plot_figure_dir, "plot"+str(step)+".png"))
                    plt.clf()


                # print training step
            if step % 4000 == 0:
                saver.save(sess, os.path.join(model_save_dir, 'p3d_with_gn_svsd_model_' + str(step) + '.ckpt'))
        print 'Training Finished!'

        
        print 'Training Finished!'

train_STSM()
