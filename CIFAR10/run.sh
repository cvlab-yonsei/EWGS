####################################################################################
# You may run the code using the following commands.
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.
####################################################################################

####### ResNet-20 full-precision model
# python train_fp.py --gpu_id '0' \
#                    --log_dir '../results/ResNet20_CIFAR10/fp/'

####### ResNet-20 W1A1 ours
# python train_quant.py --gpu_id '0' \
#                       --weight_levels 2 \
#                       --act_levels 2 \
#                       --baseline False \
#                       --use_hessian True \
#                       --load_pretrain True \
#                       --pretrain_path '../results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth' \
#                       --log_dir '../results/ResNet20_CIFAR10/ours(hess)/W1A1/'

####### ResNet-20 W1A1 ours (fixed scaling factor)
# python train_quant.py --gpu_id '1' \
#                       --weight_levels 2 \
#                       --act_levels 2 \
#                       --baseline False \
#                       --use_hessian False \
#                       --bkwd_scaling_factorW 0.001 \
#                       --bkwd_scaling_factorA 0.001 \
#                       --load_pretrain True \
#                       --pretrain_path '../results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth' \
#                       --log_dir '../results/ResNet20_CIFAR10/ours(fix)/W1A1_0.001/'

####### ResNet-20 W1A1 baseline (STE)
# python train_quant.py --gpu_id '1' \
#                       --weight_levels 2 \
#                       --act_levels 2 \
#                       --baseline True \
#                       --load_pretrain True \
#                       --pretrain_path '../results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth' \
#                       --log_dir '../results/ResNet20_CIFAR10/base/W1A1/'