 #   _____    _____   _____   _   _   ______   ______   _____  
 #  / ____|  / ____| |_   _| | \ | | |  ____| |  ____| |  __ \ 
 # | (___   | (___     | |   |  \| | | |__    | |__    | |__) |
 #  \___ \   \___ \    | |   | . ` | |  __|   |  __|   |  _  / 
 #  ____) |  ____) |  _| |_  | |\  | | |      | |____  | | \ \ 
 # |_____/  |_____/  |_____| |_| \_| |_|      |______| |_|  \_\
                                                             
import imghdr, imageio
from math import floor
import argparse, glob, cv2, os, numpy as np, datetime
import tensorflow as tf
import sys
import os
import p3d
import queue
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

model_path ='/data/lishikai/P3D/model/final/'
left_data_path = '../svsd/train/left_view_svsd/'
save_path_all = '../svsd/p3d_results/'
mkDir(save_path_all)

model_file = glob.glob(model_path+"*.data-00000-of-00001")

index = 16

MEAN_VALUE = np.array([98,102,90], dtype=np.float32)
MEAN_VALUE = MEAN_VALUE[::-1]
# mean_value = mean_value[None, ...]
# MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)

x = tf.placeholder(tf.float32, [1, 16, 112, 112, 3])
sal = p3d.p3d_unetplusplus_ds(x, 0, 1, False)

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
        save_path = save_path_all + model_name_other + '_train/'
        mkDir(save_path)
        # save_fea_path = save_path_all + model_name_other + '_feature/'
        # mkDir(save_fea_path)

        left_folds_name_list = os.listdir(left_data_path)
        left_folds_name_list.sort()

        for index_video in range(len(left_folds_name_list)):

            left_fold_name = left_folds_name_list[index_video]
            print(left_fold_name)
            
            left_fold_path = os.path.join(left_data_path,left_fold_name)
            
            left_files_path = glob.glob(os.path.join(left_fold_path,'*.jpg'))
            save_file_path = save_path + left_fold_name
            
            if os.path.exists(save_file_path)==0:
                os.mkdir(save_file_path)
            else:
                continue
            # sort image
            left_files_path = sorted(left_files_path, key=lambda name: int(name.split('_')[-1][:-4]))
            current_frame_queue = queue.Queue(index)
            for i in range(len(left_files_path)):
                # for each video
                left = left_files_path[i]
                left_name = os.path.basename(left).split('.')[0]
                name_index = int(left_name.split('_')[1])
                frame_wildcard = "frame_%d.jpg"
                start_frame_index = name_index 
                end_frame_index = name_index + index
                current_frame_list = []
                #
                if name_index <= len(left_files_path)-15:
                    # print left_name, name_index
                    ''''using queue to save time'
                    if name_index == 1:
                        put(data(1, 17))
                    else:
                        get(oldest)
                        put(newest)
                    change type from queue to list
                    '''
                    if name_index == 1:
                        for j in range(start_frame_index,end_frame_index):
                            frame_index = j
                            frame_name = frame_wildcard % frame_index
                            # print frame_name
                            frame_path = os.path.join(left_fold_path, frame_name)
                            
                            frame = cv2.imread(frame_path)
                            frame = frame[:, :, ::-1]
                            frame = frame- MEAN_VALUE
                            frame = cv2.resize(frame, (112,112))
                            frame = frame / 255.
                            current_frame_queue.put(frame)
                    else:
                        # get start_frame_index-1 put end_frame_index
                        current_frame_queue.get()
                        frame_name = frame_wildcard % (end_frame_index-1)
                        # print frame_name
                        frame_path = os.path.join(left_fold_path, frame_name)
                        frame = cv2.imread(frame_path)
                        frame = frame[:, :, ::-1]
                        frame = frame- MEAN_VALUE
                        frame = cv2.resize(frame, (112,112))
                        frame = frame / 255.
                        current_frame_queue.put(frame)              
                    current_frame_list=list(current_frame_queue.queue)
                    # print np.shape(current_frame_list)

                    # for j in range(start_frame_index,end_frame_index):
                    #     frame_index = j
                    #     frame_name = frame_wildcard % frame_index
                    #     frame_path = os.path.join(left_fold_path, frame_name)
                        
                    #     frame = cv2.imread(frame_path)
                    #     frame = frame[:, :, ::-1]
                    #     frame = frame- MEAN_VALUE
                    #     frame = cv2.resize(frame, (112,112))
                    #     frame = frame / 255.
                    #     current_frame_list.append(frame)
                    current_frame_list = np.array(current_frame_list)[None, :]
                    #print current_frame_list.shape
                    sal_map = sess.run(sal, feed_dict={x: current_frame_list})
                    # print np.sum(sal_map)
                    image = np.reshape(sal_map, [-1, 1, 112, 112])
                    if start_frame_index == 1:
                        for k in range(index):
                            save_image = np.zeros([112, 112])
                            save_image[:, :] = image[k, 0, :, :]*255.
                            frame_name = frame_wildcard % (start_frame_index+k)
                            save_image = cv2.resize(save_image, dsize=(960, 1080))
                            save_name = os.path.join(save_file_path, frame_name)
                            cv2.imwrite(save_name, save_image)
                    else:
                        save_image = np.zeros([112, 112])
                        save_image[:, :] = image[-1, 0, :, :]*255.
                        frame_name = frame_wildcard % (start_frame_index+15)
                        save_image = cv2.resize(save_image, dsize=(960, 1080))
                        save_name = os.path.join(save_file_path, frame_name)
                        cv2.imwrite(save_name, save_image)            
            # exit()
            print 'Datetime', datetime.datetime.now().isoformat()[:-7], "The %d video Done." % index_video
                    
                

