

##### train stage1 ##### 
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_st1.py \
--cfg experiments/cephalometric/train_st1.yaml \
--gpus 0,1 

##### test stage1 ##### 
# CUDA_VISIBLE_DEVICES=0 python3 scripts/test_st1.py \
# --model ./output/cepha/hrnet_attnfuse/train_st1_2023-04-30-10-13_train/664epoch_best.pth.tar \
# --cfg experiments/cephalometric/train_st1.yaml \
# --gpus 0 --local-iter 4 --infer-train --bbox-mode tight #loose 

##### train stage2 ##### 
# CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_st2.py \
# --cfg experiments/cephalometric/train_st2.yaml \
# --gpus 0,1
