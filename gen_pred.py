 #   _____    _____   _____   _   _   ______   ______   _____  
 #  / ____|  / ____| |_   _| | \ | | |  ____| |  ____| |  __ \ 
 # | (___   | (___     | |   |  \| | | |__    | |__    | |__) |
 #  \___ \   \___ \    | |   | . ` | |  __|   |  __|   |  _  / 
 #  ____) |  ____) |  _| |_  | |\  | | |      | |____  | | \ \ 
 # |_____/  |_____/  |_____| |_| \_| |_|      |______| |_|  \_\
                                                             
import imghdr, imageio
from math import floor
import argparse, glob, cv2, os, numpy as np, sys
import tensorflow as tf
import sys
import os
from Deep3DSaliency_model import Model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def mkDir(dirpath):
    if os.path.exists(dirpath)==0:
        os.mkdir(dirpath)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    return parser.parse_args()

args = get_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_path ='/data/lishikai/3DV/model/final/'
left_data_path = '../svsd/train/left_view_svsd/'
save_path_all = '../svsd/p3d_results_gn/'
mkDir(save_path_all)

model_file = glob.glob(model_path+"*.data-00000-of-00001")

index = 3
MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)

x = tf.placeholder(tf.float32, [1, 3, 224, 224, 3])
sal = Model.Deep3DSaliencySTSM_SA_2(x)

with tf.Session(config=config) as sess:
    for j in range(len(model_file)):
        #print j

        model = model_file[j]
        model_name = os.path.basename(model)
        model_name_other = model_name.split('.')[0]
        model_name_path = model_name_other+'.ckpt'
        # restore model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("load complete!")
        save_path = save_path_all + model_name_other + '_test/'
        mkDir(save_path)
        save_fea_path = save_path_all + model_name_other + '_feature/'
        mkDir(save_fea_path)

        left_folds_name_list = os.listdir(left_data_path)
        left_folds_name_list.sort()

        for index_video in range(len(left_folds_name_list)):
            left_fold_name = left_folds_name_list[index_video]
            print(left_fold_name)
            
            left_fold_path = os.path.join(left_data_path,left_fold_name)
            
            left_files_path = glob.glob(os.path.join(left_fold_path,'*.jpg'))
            save_file_path = save_path + left_fold_name
            save_feature_path = save_fea_path + left_fold_name
            mkDir(save_file_path)
            mkDir(save_feature_path)

            for i in range(len(left_files_path)):
                # for each video
                left = left_files_path[i]
                left_name = os.path.basename(left).split('.')[0]
                name_index = int(left_name.split('_')[1])
                frame_wildcard = "frame_%d.jpg"
                start_frame_index = name_index 
                end_frame_index = name_index + index
                current_frame_list = []
                if name_index < len(left_files_path)-2:
                    for j in range(start_frame_index,end_frame_index):
                        frame_index = j
                        frame_name = frame_wildcard % frame_index
                        frame_path = os.path.join(left_fold_path, frame_name)
                        
                        frame = cv2.imread(frame_path)
                        frame = frame[:, :, ::-1]
                        frame = frame- MEAN_VALUE
                        frame = cv2.resize(frame, (224,224))
                        frame = frame / 255.
                        current_frame_list.append(frame)
                    current_frame_list = np.array(current_frame_list)[None, :]
                    #print current_frame_list.shape
                    sal_map = sess.run(sal, feed_dict={x: current_frame_list})
                    # print np.sum(sal_map)
                    image = np.reshape(sal_map, [-1, 1, 224, 224]) 
                    save_image = np.zeros([224, 224])
                    save_image[:, :] = image[0, 0, :, :]*255.
                    # for m in range(1,16):
                    frame_name = frame_wildcard % start_frame_index
                    # sal_map = np.uint8(sal_map)
                    #sal_map = cv2.resize(sal_map, dsize=(1920, 1080))
                    save_name = os.path.join(save_file_path, frame_name)
                    cv2.imwrite(save_name, save_image)
            print "The %d video Done." % index_video
                    
                

