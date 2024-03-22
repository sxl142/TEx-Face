CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --outdir=~/remote_t4/eg3d/training-runs --cfg=ffhq --data=./ffhq/FFHQ_512.zip \
  --gpus=4 --batch=16 --gamma=1 --gen_pose_cond=True --mbstd-group 1
