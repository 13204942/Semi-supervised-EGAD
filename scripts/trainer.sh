########################################################################################################
# Active Learning
## RandomSampling
#------------------------------------------------- HC18 -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 4000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 0

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 12000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 16000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 2000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 0

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 4000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 4


#------------------------------------------------- ESTT -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 2800 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 0

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 5600 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 8400 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 11200 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 14000 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 1200 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 0

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 2400 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 3600 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 4800 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy RandomSampling --al_iter 4

## EntropySampling
#------------------------------------------------- HC18 -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 12000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 16000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 4000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 4

#------------------------------------------------- ESTT -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 5600 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 8400 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 11200 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 14000 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 2400 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 3600 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 4800 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy EntropySampling --al_iter 4


## HybridSampling
#------------------------------------------------- HC18 -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 12000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 16000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 20000 --labeled_bs 1 --labeled_num 10 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 4000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 8000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/HC18 --exp HC18/Semi_Unext_Swin_AL --model Unext --max_iterations 10000 --labeled_bs 1 --labeled_num 5 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 4


#------------------------------------------------- ESTT -------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 5600 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 8400 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 11200 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 14000 --labeled_bs 1 --labeled_num 7 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 4
#--------------------------------------------------------------------------------------------------------#
python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 2400 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 1

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 3600 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 2

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 4800 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 3

python train_semi_al.py --root_path /mnt/storage/fangyijie/ESTT --exp ESTT/Semi_Unext_Swin_AL --model Unext --max_iterations 6000 --labeled_bs 1 --labeled_num 3 --batch_size 5 --num_classes 2 --al_strategy HybridSampling --al_iter 4