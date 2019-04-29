import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, shutil
import cPickle as pkl
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', type=str, default='hdreye_hdr', help='training dataset')
    parser.add_argument('--metric_dir', type=str, default='../metric-matlab', help='training dataset')
    parser.add_argument('--debug', type=int, default=0, help='parameter of sauc')

    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

# sal_base = ds.saliency_basedir
# dens_dir = ds.density_basedir
# fixa_dir = ds.fixation_basedir
if args.dsname =='videoset':
    dens_dir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/VideoSet/Results/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'
elif args.dsname=='msu':
    dens_dir='/data/SaliencyDataset/Video/MSU/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/MSU/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/MSU/fixation/image'
elif args.dsname=='ledov':
    dens_dir='/data/SaliencyDataset/Video/LEDOV/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/LEDOV/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/LEDOV/fixation'
elif args.dsname=='hollywood':
    dens_dir='/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/density'
    sal_base='/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/fixation'
elif args.dsname =='dhf1k':
    dens_dir='/data/SaliencyDataset/Video/DHF1K/density'
    sal_base='/data/SaliencyDataset/Video/DHF1K/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/DHF1K/fixation'
elif args.dsname =='diem':
    dens_dir='/data/SaliencyDataset/Video/DIEM/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/DIEM/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/DIEM/fixation_map/image'
elif args.dsname == 'gazecom':
    dens_dir='/data/SaliencyDataset/Video/GAZECOM/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/GAZECOM/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/GAZECOM/fixations'
elif args.dsname == 'coutort2':
    dens_dir='/data/SaliencyDataset/Video/Coutort2/density/sigma32'
    sal_base='/data/SaliencyDataset/Video/Coutort2/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/Coutort2/fixations'
elif args.dsname == 'dhf1k':
    dens_dir='/data/SaliencyDataset/Video/DHF1K/density'
    sal_base='/data/SaliencyDataset/Video/DHF1K/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/DHF1K/fixation'
elif args.dsname == 'svsd':
    dens_dir='/data/lishikai/svsd/DHF1K/density'
    sal_base='/data/SaliencyDataset/Video/DHF1K/saliency_map_1128'
    fixa_dir='/data/SaliencyDataset/Video/DHF1K/fixation'

else:
    raise NotImplementedError

save_base = os.path.join(args.metric_dir, args.dsname)
if not os.path.isdir(save_base):
    os.makedirs(save_base)

print sal_base,dens_dir,fixa_dir

model_list =  [ name for name in os.listdir(sal_base) if os.path.isdir(os.path.join(sal_base, name)) ]

# print sal_base, model_list;exit()

for model in model_list:
    sal_dir = os.path.join(sal_base, model)
    video_name_list = os.listdir(sal_dir)

    if not os.path.isdir(os.path.join(save_base, model)):
        os.makedirs(os.path.join(save_base, model))
    for video_name in video_name_list:

        save_path = os.path.join(save_base, model, os.path.splitext(video_name)[0]+'-'+model+'.mat')
        if os.path.isfile(save_path):
            print save_path, 'already exists, pass.'
            continue

        var_str = 'model_name=\'%s\';save_base=\'%s\';dsname=\'%s\';sal_dir=\'%s\';dens_dir=\'%s\';fixa_dir=\'%s\';'% (model, os.path.join(save_base, model), args.dsname, os.path.join(sal_dir,video_name), os.path.join(dens_dir, video_name), os.path.join(fixa_dir, video_name))
        # print var_str;exit()
        cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric\');%s metric_video_base;exit()"' % var_str
        # cmd = cmd + 'exit()"'
        # if args.debug ==0:
        # else:
        #     cmd = cmd + '"'

        print 'running:', cmd

        os.system(cmd)
        if args.debug == 1:
            exit()
    
for model in model_list:
    # stastics
    cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'metric\');save_base=\'%s\';model_name=\'%s\';metric_statistics;exit()"' % (os.path.join(save_base, model), model)
    os.system(cmd)