# Training Scripts for the MovingMNIST++ experiments

The models can be trained using the configurations files in the `configurations` folder. We given two examples:
```
# Train ConvGRU-K3D2 model
python mnist_rnn_main.py \
    --cfg configurations/convgru_1_64_1_96_1_96_K3D2.yml \
    --save_dir convgru_K3D2 \
    --ctx gpu0

# Train ConvGRU-K5 model
python mnist_rnn_main.py \
    --cfg configurations/convgru_1_64_1_96_1_96_K5.yml \
    --save_dir convgru_K5 \
    --ctx gpu0

# Train ConvGRU-K7 model
python mnist_rnn_main.py \
    --cfg configurations/convgru_1_64_1_96_1_96_K7.yml \
    --save_dir convgru_K7 \
    --ctx gpu0

# Train DFN model
python mnist_rnn_main.py \
    --cfg configurations/convgru_1_64_1_96_1_96_K5_DFN.yml \
    --save_dir convgru_K5_DFN \
    --ctx gpu0

# Train TrajGRU-L5 model
python mnist_rnn_main.py \
    --cfg configurations/trajgru_1_64_1_96_1_96_L5.yml \
    --save_dir trajgru_L5 \
    --ctx gpu0

# Train TrajGRU-L9 model
python mnist_rnn_main.py \
    --cfg configurations/trajgru_1_64_1_96_1_96_L9.yml \
    --save_dir trajgru_L9 \
    --ctx gpu0

# Train TrajGRU-L13 model
python mnist_rnn_main.py \
    --cfg configurations/trajgru_1_64_1_96_1_96_L13.yml \
    --save_dir trajgru_L13 \
    --ctx gpu0

# Train TrajGRU-L17 model
python mnist_rnn_main.py \
    --cfg configurations/trajgru_1_64_1_96_1_96_L17.yml \
    --save_dir trajgru_L17 \
    --ctx gpu0
 ```

Also, we have the training scripts for Conv2D and Conv3D models.
 ```
# Train Conv2D model
python deconvolution.py --cfg configurations/conv2d_3d/conv2d.yml --save_dir conv2d --ctx gpu0
# Train Conv3D model
python deconvolution.py --cfg configurations/conv2d_3d/conv3d.yml --save_dir conv3d --ctx gpu0
 ```

# Test with Pretrained Models

The pretrained models can be downloaded from [Dropbox](https://www.dropbox.com/sh/n7gxfdd1pdasoio/AAC8uC4yto5Uam_7f3BEl-3La?dl=0) or using the `download_pretrained.sh` in the folder. After the models are downloaded, you can test them using the following commands:
```
# ConvGRU with K=3, D=2
python mnist_rnn_test.py --cfg ConvGRU-K3D2/cfg0.yml --load_dir ConvGRU-K3D2 --load_iter 199999 --save_dir ConvGRU-K3D2 --ctx gpu0
# ConvGRU with K=5, D=1
python mnist_rnn_test.py --cfg ConvGRU-K5/cfg0.yml --load_dir ConvGRU-K5 --load_iter 199999 --save_dir ConvGRU-K5 --ctx gpu0
# ConvGRU with K=7, D=1
python mnist_rnn_test.py --cfg ConvGRU-K7/cfg0.yml --load_dir ConvGRU-K7 --load_iter 199999 --save_dir ConvGRU-K7 --ctx gpu0
# DFN
python mnist_rnn_test.py --cfg DFN/cfg0.yml  --load_dir DFN --load_iter 199999 --save_dir DFN --ctx gpu0
# TrajGRU with L=5
python mnist_rnn_test.py --cfg TrajGRU-L5/cfg0.yml --load_dir TrajGRU-L5 --load_iter 199999 --save_dir TrajGRU-L5 --ctx gpu0
# TrajGRU with L=9
python mnist_rnn_test.py --cfg TrajGRU-L9/cfg0.yml --load_dir TrajGRU-L9 --load_iter 199999 --save_dir TrajGRU-L9 --ctx gpu0
# TrajGRU with L=13
python mnist_rnn_test.py --cfg TrajGRU-L13/cfg0.yml --load_dir TrajGRU-L13 --load_iter 199999 --save_dir TrajGRU-L13 --ctx gpu0
# TrajGRU with L=17
python mnist_rnn_test.py --cfg TrajGRU-L17/cfg0.yml --load_dir TrajGRU-L17 --load_iter 199999 --save_dir TrajGRU-L17 --ctx gpu0
```