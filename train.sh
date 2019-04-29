python train.py --dataset=svsdndhf1k --gpu=0 --batch=2 --epoch=1 --structure=unet++ --overlap=15 --SA=False --normalization=BN --info=sigmoid

python train.py --dataset=svsdndhf1k --gpu=1 --batch=2 --epoch=1 --structure=unet++ --overlap=15 --SA=True --normalization=BN --info=sigmoid_subsampleSA

python train.py --dataset=svsdndhf1k --gpu=2 --batch=2 --epoch=1 --structure=unet++ --overlap=8 --SA=True --normalization=BN --info=sigmoid_subsampleSA_ol8
