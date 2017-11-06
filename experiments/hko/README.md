# Scripts for the HKO-7 Benchmark

- Use `last_frame_prediction.py` to evaluate the naive baseline which uses the last frame to predict all the future frames.
It's a good example of how to test your own models using the offline setting of the benchmark.

- Use `rover.py` to evaluate the rover algorithm in the following paper. Before running the experiments, make sure that the VarFlow package (https://github.com/sxjscience/HKO7_Benchmark/tree/master/VarFlow) is installed.
    ```
    @article{woo2017operational,
      title={Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting},
      author={Woo, Wang-chun and Wong, Wai-kin},
      journal={Atmosphere},
      volume={8},
      number={3},
      pages={48},
      year={2017},
      publisher={Multidisciplinary Digital Publishing Institute}
    }
    ```

- Use `hko_main.py` to train the RNN models for precipitation nowcasting and use `hko_rnn_test.py` to test these models. The training scripts support multiple GPUs.
If you find you cannot train the model using a single GPU, try to decrease the batch_size.

    1. Commands for training the RNN models:
    ```
    # Train ConvGRU model with B-MSE + B-MAE
    python3 hko_main.py --cfg configurations/convgru_55_55_33_1_64_1_192_1_192_b4.yml --save_dir convgru_55_55_33_1_64_1_192_1_192_b4 --ctx gpu0,gpu1
    # Train TrajGRU model with B-MSE + B-MAE
    python3 hko_main.py --cfg configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml --save_dir trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4 --ctx gpu0,gpu1
    # Train ConvGRU model without B-MSE + B-MAE
    python3 hko_main.py --cfg configurations/convgru_55_55_33_1_64_1_192_1_192_nobal_b4.yml --save_dir convgru_55_55_33_1_64_1_192_1_192_nobal_b4 --ctx gpu0,gpu1
    ```

- Use `deconvolution.py` to run the 2D/3D models.

    1. Commands for trainnig the CNN models:
    ```
    # Train Conv2D model
    python deconvolution.py --cfg configurations/conv2d_3d/conv2d.yml --save_dir Conv2D --ctx gpu0
    # Train Conv3D model
    python deconvolution.py --cfg configurations/conv2d_3d/conv2d.yml --save_dir Conv3D --ctx gpu0
    ```

# Test with Pretrained Models

You can download the pretrained ConvGRU/TrajGRU models by manually visit [Download By Dropbox](https://www.dropbox.com/sh/cp8zpi08umfiyha/AAAS6HJSsDQPjpKlxnBHtHvga?dl=0) or using the "download_pretrained.sh" in the folder.

```
# Test ConvGRU model in the offline setting
python hko_rnn_test.py \
 --cfg ConvGRU/cfg0.yml \
 --load_dir ConvGRU \
 --load_iter 49999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir ConvGRU \
 --mode fixed \
 --dataset test

# Test ConvGRU model in the online setting
python hko_rnn_test.py \
 --cfg ConvGRU/cfg0.yml \
 --load_dir ConvGRU \
 --load_iter 49999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir ConvGRU \
 --mode online \
 --dataset test

# Test TrajGRU model in the offline setting
python hko_rnn_test.py \
 --cfg TrajGRU/cfg0.yml \
 --load_dir TrajGRU \
 --load_iter 79999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir TrajGRU \
 --mode fixed \
 --dataset test

# Test TrajGRU model in the online setting
python hko_rnn_test.py \
 --cfg TrajGRU/cfg0.yml \
 --load_dir TrajGRU \
 --load_iter 79999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir TrajGRU \
 --mode online \
 --dataset test

# Test ConvGRU model without balanced loss
python hko_rnn_test.py \
 --cfg ConvGRU-nobal/cfg0.yml \
 --load_dir ConvGRU-nobal \
 --load_iter 59999 \
 --finetune 0 \
 --ctx gpu0 \
 --save_dir ConvGRU-nobal \
 --mode fixed \
 --dataset test

# Test ConvGRU model without balanced loss
python hko_rnn_test.py \
 --cfg ConvGRU-nobal/cfg0.yml \
 --load_dir ConvGRU-nobal \
 --load_iter 59999 \
 --finetune 1 \
 --lr 1E-4 \
 --ctx gpu0 \
 --save_dir ConvGRU-nobal \
 --mode online \
 --dataset test
```