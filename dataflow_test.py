# coding = utf-8
import os, glob, cv2, numpy as np
import random
from random import shuffle
import tensorflow as tf
import time as time
import multiprocessing
import tqdm
import datetime
from tensorpack.dataflow import *
from tensorpack.dataflow.imgaug import *
from tensorpack.input_source import QueueInput, StagingInput
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class VideoDataset():
    def __init__(self, frame_basedir, density_basedir, img_size=(480,288), video_length=16, stack=5, bgr_mean_list=[103.939, 116.779, 123.68],sort='bgr'):
        MEAN_VALUE = np.array(bgr_mean_list, dtype=np.float32)   # B G R/ use opensalicon's mean_value
        if sort=='rgb':
            MEAN_VALUE= MEAN_VALUE[::-1]
        self.MEAN_VALUE = MEAN_VALUE[None, ...]
        self.img_size = img_size
        self.dataset_dict={}
        self.video_length = video_length
        self.step = 1
        # self.stack = stack
        assert self.step < self.video_length
        self.frame_basedir = frame_basedir
        self.density_basedir = density_basedir
        self.video_dir_list = glob.glob(os.path.join(self.frame_basedir, "*"))

        # self.setup_video_dataset()

    def setup_video_dataset_p3d(self, overlap=2, training_example_props=0.8, skip_head=11): ## skip those bad data in the previous of a video
        # pass
        self.tuple_list = []
        assert overlap < self.video_length, "overlap should smaller than videolength."
        step = self.video_length - overlap
        for i in range(len(self.video_dir_list)):
            video_dir = self.video_dir_list[i]
            frame_list = glob.glob(os.path.join(video_dir,'*.*'))
            total_frame = len(frame_list)
            for j in range(skip_head, total_frame, step): ## div 2, so 1/2 of the video_length is overlapped
                if j + self.video_length > total_frame-30:

                    break
                tup = (i, j) # video index and first frame index
                self.tuple_list.append(tup)
            # print self.tuple_list;exit()
        self.num_examples = len(self.tuple_list)
        # shuffle
        # shuffle(self.tuple_list)
        self.num_training_examples = int(self.num_examples * training_example_props)
        # 20% for validation and 80% for training 
        self.training_tuple_list = self.tuple_list[:self.num_training_examples]
        self.validation_tuple_list = self.tuple_list[self.num_training_examples:]
        self.num_validation_examples = len(self.validation_tuple_list)
        print self.num_examples, "samples generated in total,",self.num_training_examples,"training samples,",self.num_validation_examples,"validation samples";#exit()

        self.num_epoch = 0
        self.index_in_training_epoch = 0
        self.index_in_validation_epoch = 0
        self.final_train_list = []
        self.final_valid_list = []

    def get_frame_p3d_tf(self, mini_batch=1, phase='training', density_length='full'):
        ## 
        frame_wildcard = "frame_%d.jpg"
        gt_wildcard = "frame_%d.jpg"
        ##  training
        tuple_list = self.training_tuple_list
        index_in_epoch = 0
        self.index_in_training_epoch += mini_batch
        num_examples = self.num_training_examples
        tup_list = []
        while index_in_epoch <= num_examples - mini_batch:
            tup_batch = tuple_list[index_in_epoch:index_in_epoch+mini_batch]
            for tup in tup_batch:
                current_frame_list = []
                current_density_list = []
                video_index, start_frame_index=tup
                end_frame_index = start_frame_index + self.video_length
                video_dir = self.video_dir_list[video_index]
                # print video_dir
                video_name = os.path.basename(video_dir)
                # print video_name
                density_dir = os.path.join(self.density_basedir, video_name)
                for i in range(start_frame_index, end_frame_index):
                    frame_index = i + 1
                    frame_name = frame_wildcard % frame_index
                    current_frame_list.append(glob.glob(os.path.join(video_dir, frame_name))[0])
                if density_length=='full':
                    for i in range(start_frame_index,end_frame_index):
                        frame_index = i + 1
                        frame_name = gt_wildcard % frame_index
                        current_density_list.append(glob.glob(os.path.join(density_dir, frame_name))[0])                
                elif density_length=='one':
                    frame_index = end_frame_index
                    frame_name = gt_wildcard % frame_index
                    current_density_list.append(glob.glob(os.path.join(density_dir, frame_name))[0])
            tup_list.append(current_frame_list)
            tup_list.append(current_density_list)
            index_in_epoch += mini_batch
            self.final_train_list.append(tup_list)
            tup_list = []
       
        ## validation
        tuple_list = self.validation_tuple_list
        index_in_epoch = 0
        num_examples = self.num_validation_examples
        tup_list = []
        while not  index_in_epoch >= num_examples - mini_batch:
            tup_batch = tuple_list[index_in_epoch:index_in_epoch+mini_batch]
            for tup in tup_batch:
                current_frame_list = []
                current_density_list = []
                video_index, start_frame_index=tup
                end_frame_index = start_frame_index + self.video_length
                video_dir = self.video_dir_list[video_index]
                video_name = os.path.basename(video_dir)
                density_dir = os.path.join(self.density_basedir, video_name)
                for i in range(start_frame_index, end_frame_index):
                    frame_index = i + 1
                    frame_name = frame_wildcard % frame_index
                    current_frame_list.append(glob.glob(os.path.join(video_dir, frame_name))[0])
                if density_length=='full':
                    for i in range(start_frame_index,end_frame_index):
                        frame_index = i + 1
                        frame_name = gt_wildcard % frame_index
                        current_density_list.append(glob.glob(os.path.join(density_dir, frame_name))[0])              
                elif density_length=='one':
                    frame_index = end_frame_index
                    frame_name = gt_wildcard % frame_index
                    current_density_list.append(glob.glob(os.path.join(density_dir, frame_name))[0])  
            tup_list.append(current_frame_list)
            tup_list.append(current_density_list)
            index_in_epoch += mini_batch
            self.final_valid_list.append(tup_list)
            tup_list = []
        


class ImageFromFile(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.shuffle = shuffle


    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for clip in self.files:
            yield clip

def rgb_augmentor():
    augmentors = [
        imgaug.Resize(112),
    ]
    return augmentors

def grey_augmentor():
    augmentors = [
        imgaug.Resize(112)
    ]
    return augmentors

mean_value = np.array([98,102,90], dtype=np.float32)
mean_value = mean_value[::-1]
mean_value = mean_value[None, ...]

def mapf(dp):
    frame_list, density_list = dp
    ret_frame = []
    ret_density = []
    for frame in frame_list:
        im = cv2.imread(frame, cv2.IMREAD_COLOR)
        im = im [:, :, ::-1]
        im = im-mean_value
        im = imgaug.AugmentorList(rgb_augmentor()).augment(im)
        im = im / 255.
        ret_frame.append(im)
    for density in density_list:
        im = cv2.imread(density, cv2.IMREAD_GRAYSCALE)
        im = imgaug.AugmentorList(rgb_augmentor()).augment(im)
        im = im / 255.
        ret_density.append(im)
    return ret_frame, ret_density

if __name__ == "__main__":
    train_frame_basedir = '../svsd/test/left_view_svsd'
    train_density_basedir = '../svsd/test/view_svsd_density/'
    videodataset = VideoDataset(train_frame_basedir,train_density_basedir, video_length=16, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
    videodataset.setup_video_dataset_p3d(overlap=15, training_example_props=0.8)
    videodataset.get_frame_p3d_tf()
    # print videodataset.final_train_list[0][0], videodataset.final_train_list
    df = ImageFromFile(videodataset.final_train_list)
    df = MultiThreadMapData(
        df, nr_thread=16,
        map_func=mapf,
        buffer_size=1000,
        strict=True)
    df = BatchData(df, 2, remainder=True, use_list=True)
    df = PrefetchDataZMQ(df, nr_proc=1)
    df.reset_state()
    # TestDataSpeed(df1).start()
    
    # df = VideoDataflow(train_frame_basedir, train_density_basedir)
    # ds = BatchData(df, 2)
    # for each in ds:
    #     print np.array(each).shape
    # TestDataSpeed(ds).start()
        