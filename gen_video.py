import cv2
import numpy as np
import glob
import os
#np.set_printoptions(threshold='nan')
#P3D
video_path = '/data/lishikai/svsd/p3d_results/p3d_392000_train/'
# save_path_left = '/data/lishikai/svsd/p3d_results/p3d_392000_train_video/'
save_path_left = '/data/lishikai/svsd/video/train/p3d/'


#Flownet
# video_path = '/data/lishikai/svsd/flownet_results/Flownet-results/flownet-train-20000/'
# video_path = '/data/lishikai/svsd/results/flownet_results/Flownet-results/flownet-test-20000/'
#gt_path = '/data/qiudan/videosaliency/videodataset/SVSDataset/fixation_left_density/sigma32/'
# save_path_left = '/data/lishikai/svsd/video/test/flownet/'

#GT
# video_path = '/data/lishikai/svsd/train/left_density_svsd/'
# save_path_left = '/data/lishikai/svsd/left_density_video/'

if os.path.isdir(save_path_left):
    pass
else:
    os.mkdir(save_path_left)

video_file = glob.glob(video_path+"*")
video_file.sort()
out_size = (112,112)
for j in range(len(video_file)):
    #print j
    video = video_file[j]
    print video
    VideoName_short = os.path.basename(video)
    print VideoName_short

    fps = 25 #float(VideoCap.get(cv2.CAP_PROP_FPS))
    img_file = glob.glob(os.path.join(video,"*.*"))
    save_name_left = save_path_left + VideoName_short + '.avi'
    print save_name_left
    videoWriter_left = cv2.VideoWriter(save_name_left, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, out_size, isColor=False)
    for i in range(6,len(img_file)-1):
        frame_name = video + '/' + 'frame_' + str(i) + '.jpg'
        frame = cv2.imread(frame_name)
        frame = cv2.resize(frame,out_size)
        frame = frame[:,:,0]
        videoWriter_left.write(frame)
    cv2.destroyAllWindows()
'''
 status, frame = VideoCap.read()
    while(status):
        #frame = cv2.resize(frame, (1920,1080))
        frame_left = frame[:,0:1920,:]
        videoWriter_left.write(frame_left)
        status, frame = VideoCap.read()
    VideoCap.release()
    VideoCap_temp.release() 	
    cv2.destroyAllWindows()
'''  
     