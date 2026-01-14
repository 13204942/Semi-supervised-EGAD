####################################################################
# Semi-supervised Learning
# SOTA Methods
# MT + unext
CUDA_LAUNCH_BLOCKING=1 
python -m sota.train_mean_teacher_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/MT_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_mean_teacher_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/MT_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_mean_teacher_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/MT_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_mean_teacher_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/MT_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# CPS + unext
python -m sota.train_cross_pseudo_supervision_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/CPS_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_cross_pseudo_supervision_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/CPS_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_cross_pseudo_supervision_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/CPS_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_cross_pseudo_supervision_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/CPS_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# ICT + unext
python -m sota.train_interpolation_consistency_training_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/ICT_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_interpolation_consistency_training_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/ICT_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_interpolation_consistency_training_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/ICT_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_interpolation_consistency_training_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/ICT_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# DAN + unext
python -m sota.train_adversarial_network_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/DAN_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_adversarial_network_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/DAN_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_adversarial_network_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/DAN_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_adversarial_network_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/DAN_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# UAMT + unext
python -m sota.train_uncertainty_aware_mean_teacher_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/UAMT_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_uncertainty_aware_mean_teacher_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/UAMT_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_uncertainty_aware_mean_teacher_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/UAMT_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_uncertainty_aware_mean_teacher_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/UAMT_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# DCT + unext 
python -m sota.train_deep_co_training_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/DCT_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_deep_co_training_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/DCT_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_deep_co_training_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/DCT_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_deep_co_training_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/DCT_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# CT-CT + unext 
python -m sota.train_cross_teaching_between_cnn_transformer_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/CTCT_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_cross_teaching_between_cnn_transformer_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/CTCT_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2
python -m sota.train_cross_teaching_between_cnn_transformer_2D --root_path /mnt/storage/fangyijie/HC18 --exp HC18/CTCT_Unext --model Unext --max_iterations 22500 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_cross_teaching_between_cnn_transformer_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/CTCT_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_cross_teaching_between_cnn_transformer_2D --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/CTCT_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2

# PCPCS + unext 
python -m sota.train_Semi_Mamba_UNet --root_path /mnt/storage/fangyijie/HC18 --exp HC18/PCPCS_Unext --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 25 --batch_size 5 --num_classes 2
python -m sota.train_Semi_Mamba_UNet --root_path /mnt/storage/fangyijie/HC18 --exp HC18/PCPCS_Unext --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2
python -m sota.train_Semi_Mamba_UNet --root_path /mnt/storage/fangyijie/HC18 --exp HC18/PCPCS_Unext --model Unext --max_iterations 22500 --labeled_bs 1 --labeled_num 50 --batch_size 5 --num_classes 2

python -m sota.train_Semi_Mamba_UNet --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/PCPCS_Unext --model Unext --max_iterations 6800 --labeled_bs 1 --labeled_num 17 --batch_size 5 --num_classes 2
python -m sota.train_Semi_Mamba_UNet --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/PCPCS_Unext --model Unext --max_iterations 13600 --labeled_bs 1 --labeled_num 34 --batch_size 5 --num_classes 2