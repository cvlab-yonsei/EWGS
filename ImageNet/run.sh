####################################################################################
# You may run the code using the following commands.
# Please change the directory for the dataset (i.e., /path/to/ILSVRC2012).
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.
####################################################################################

####### ResNet-18 W1A1 ours
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --visible_gpus '0,2,4,6' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --act_levels 2 \
#                                --baseline False \
#                                --use_hessian True \
#                                --update_scales_every 1 \
#                                --log_dir "../results/ResNet18/ours(hess)/W1A1/"

####### ResNet-18 W1A1 ours (fixed scaling factor)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --dist-url 'tcp://127.0.0.1:23457' \
#                                --visible_gpus '1,3,5,7' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --act_levels 2 \
#                                --baseline False \
#                                --use_hessian False \
#                                --bkwd_scaling_factorW 0.001 \
#                                --bkwd_scaling_factorA 0.001 \
#                                --log_dir "../results/ResNet18/ours(fix)/W1A1_0.001/"

####### ResNet-18 W1A1 baseline (STE)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --visible_gpus '0,1,2,3' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --act_levels 2 \
#                                --baseline True \
#                                --log_dir "../results/ResNet18/base/W1A1/"

####### ResNet-18 W1A32 ours
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --visible_gpus '0,2,4,6' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --QActFlag False \
#                                --baseline False \
#                                --use_hessian True \
#                                --update_scales_every 1 \
#                                --log_dir "../results/ResNet18/ours(hess)/W1A32/"

####### ResNet-18 W1A32 baseline (STE)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --dist-url 'tcp://127.0.0.1:23457' \
#                                --visible_gpus '1,3,5,7' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --QActFlag False \
#                                --baseline True \
#                                --log_dir "../results/ResNet18/base/W1A32/"