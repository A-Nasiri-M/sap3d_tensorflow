% delete(gcp)
pool = parpool(5)
% matlabpool 7
clc;
% clear;
metricsFolder = '/data/sunnycia/saliency_on_videoset/Train/scripts/metric/saliency/code_forMetrics'
addpath(genpath(metricsFolder))

% dsname='ledov';

cc_msk  = 1;
sim_msk = 1;
jud_msk = 1;
bor_msk = 0;
kl_msk  = 0;
nss_msk = 0;

if ~exist('model_name', 'var')
    fprintf('model_name variable not exists\n');
end
if ~exist('dsname', 'var')
    fprintf('dsname variable not exists\n');
end
if ~exist('sal_dir', 'var')
    fprintf('sal_dir variable not exists\n');
end
if ~exist('dens_dir', 'var')
    fprintf('dens_dir variable not exists\n');
end
if ~exist('fixa_dir', 'var')
    fprintf('fixa_dir variable not exists\n');
end
if ~exist('save_base', 'var')
    fprintf('save_base variable not exists\n');
end

dens_subdir_list = dir(dens_dir); % density/video1, video2....
fixa_subdir_list = dir(fixa_dir); % Fixation/video1, video2....
% baseline_map = imread('baseline_map.bmp');

% sal_dir = fullfile(base_sal_dir, model_name);   % _Saliencymap/SALICON
sal_subdir_list = dir(sal_dir);                 % _Saliencymap/SALICON/video1, video2....

ext = {'*.jpg','*.bmp', '*.png', '*.jpeg', '*.tif'};
saliencymap_path_list = [];
densitymap_path_list = [];
fixationmap_path_list = [];
for e = 1:length(ext)
    saliencymap_path_list = [saliencymap_path_list dir(fullfile(sal_dir, ext{e}))];
    densitymap_path_list = [densitymap_path_list dir(fullfile(dens_dir, ext{e}))];
    fixationmap_path_list = [fixationmap_path_list dir(fullfile(fixa_dir, ext{e}))];
end
saliencymap_path_list = natsortfiles({saliencymap_path_list.name});
densitymap_path_list = natsortfiles({densitymap_path_list.name});
fixationmap_path_list = natsortfiles({fixationmap_path_list.name});
saliencymap_path_list
[pathstr,name,saliency_ext] = fileparts(char(saliencymap_path_list(1)));
[pathstr,name,density_ext] = fileparts(char(densitymap_path_list(1)));
[pathstr,name,fixation_ext] = fileparts(char(fixationmap_path_list(1)));

LengthFiles = length(saliencymap_path_list);
saliency_score_CC = zeros(1,LengthFiles);
saliency_score_SIM = zeros(1,LengthFiles);
saliency_score_JUD = zeros(1,LengthFiles);
saliency_score_BOR = zeros(1,LengthFiles);
saliency_score_KL = zeros(1,LengthFiles);
saliency_score_NSS = zeros(1,LengthFiles);

%% CALCULATE METRICS %%
disp('calculate the metrics...');
t1=clock;
% for j = 1 : 2
[filepath,video_name, ext] = fileparts(sal_dir)
save_path = fullfile(save_base, strcat(video_name,'-', model_name, '.mat'));
if exist(save_path, 'file') == 1
    fprintf('%s alread exists, skip.', save_path);
    exit()
end
parfor j=1:LengthFiles
% for j=1:LengthFiles
    frame_name = char(saliencymap_path_list(j))

    [pathstr, frame_prefix, saliency_ext] = fileparts(frame_name);

    smap_path = fullfile(sal_dir,frame_name);
    density_path = fullfile(dens_dir,[frame_prefix density_ext]);
    fixation_path = fullfile(fixa_dir, [frame_prefix fixation_ext]);

    if exist(density_path, 'file') == 0
        continue
    end
    if exist(fixation_path, 'file') == 0
        continue
    end

    fprintf('Handling %s\n', smap_path);
    image_saliency = imread(smap_path);
    [rows, columns, numberOfColorChannels] = size(image_saliency);
    if numberOfColorChannels >1 
        image_saliency = rgb2gray(image_saliency);
    end

    image_density = imread(density_path);
    image_fixation = imread(fixation_path);

    [row,col] = size(image_saliency);
    other_map=zeros(row, col);

    if cc_msk
        %% CC %%
        % tic
        saliency_score_CC(j) = CC(image_saliency, image_density);
        % toc
    end
    
    if sim_msk
        %% SIM %% 
        % tic
        saliency_score_SIM(j) = similarity(image_saliency, image_density);
        % toc
    end
    
    if jud_msk
        %% AUCJUDD %%
        % tic
        % saliency_score_JUD(j) = AUC_Judd(image_saliency, image_density);
        saliency_score_JUD(j) = AUC_Judd(image_saliency, image_fixation);
        % toc
    end
    
    if bor_msk
        %% AUCBorji %%
        % tic
        % saliency_score_BOR(j) = AUC_Borji(image_saliency, image_density);
        saliency_score_BOR(j) = AUC_Borji(image_saliency, image_fixation);
        % toc
    end

    if kl_msk
        %% KL %%
        % tic
        % saliency_score_KL(j) = KLdiv(image_saliency, image_density);
        saliency_score_KL(j) = KLdiv(image_saliency, image_density);
        % saliency_score_KL(j) = KLdiv(image_saliency, image_fixation);
        % toc
    end
        
    if nss_msk
        %% NSS %%
        % tic
        % saliency_score_NSS(j)=NSS(image_saliency, image_density);
        saliency_score_NSS(j)=NSS(image_saliency, image_fixation);
        % toc
    end
end

% fprintf('Done for %s',sal_subdir_list(i).name)
saliency_score=[saliency_score_CC;saliency_score_SIM;
                saliency_score_JUD;saliency_score_BOR;
                saliency_score_KL;saliency_score_NSS;]
t2=clock;
time_cost=etime(t2,t1);

save(save_path, 'saliency_score','time_cost');
fprintf('%s saved\n',save_path);

fprintf('Done for %s',model_name);