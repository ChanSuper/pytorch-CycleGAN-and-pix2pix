python train.py --dataroot ./datasets/humanparsing --name humanparsing_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 20 --dataset_mode parsing --pool_size 0 --no_flip --output_nc 20 --parts 20 --pool_size 50 --no_dropout --softmax_out
