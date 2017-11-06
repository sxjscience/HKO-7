import argparse
import os
import random
import logging
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mnist_rnn_factory import MovingMNISTFactory
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.helpers.gifmaker import save_gif
from nowcasting.my_module import MyModule
from nowcasting.encoder_forecaster import *
from nowcasting.utils import parse_ctx, logging_config, load_params
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
from mnist_rnn_main import save_movingmnist_cfg, mnist_get_prediction

random.seed(12345)
mx.random.seed(930215)
np.random.seed(921206)


def parse_args():
    parser = argparse.ArgumentParser(description='Test the MovingMNIST++ dataset')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the testing process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Testing configuration file', default=None, type=str)
    parser.add_argument('--load_dir', help='The loading directory', default=None, type=str)
    parser.add_argument('--load_iter', help='The loading iterator', default=None, type=int)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg)
    if args.load_dir is not None:
        cfg.MODEL.LOAD_DIR = args.load_dir
    if args.load_iter is not None:
        cfg.MODEL.LOAD_ITER = args.load_iter
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args

def analysis(args):
    cfg.MODEL.TRAJRNN.SAVE_MID_RESULTS = True
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    base_dir = args.save_dir
    logging_config(folder=base_dir, name="testing")
    save_movingmnist_cfg(base_dir)
    mnist_iter = MovingMNISTAdvancedIterator(
        distractor_num=cfg.MOVINGMNIST.DISTRACTOR_NUM,
        initial_velocity_range=(cfg.MOVINGMNIST.VELOCITY_LOWER,
                                cfg.MOVINGMNIST.VELOCITY_UPPER),
        rotation_angle_range=(cfg.MOVINGMNIST.ROTATION_LOWER,
                              cfg.MOVINGMNIST.ROTATION_UPPER),
        scale_variation_range=(cfg.MOVINGMNIST.SCALE_VARIATION_LOWER,
                               cfg.MOVINGMNIST.SCALE_VARIATION_UPPER),
        illumination_factor_range=(cfg.MOVINGMNIST.ILLUMINATION_LOWER,
                                   cfg.MOVINGMNIST.ILLUMINATION_UPPER))
    mnist_rnn = MovingMNISTFactory(batch_size=1,
                                   in_seq_len=cfg.MODEL.IN_LEN,
                                   out_seq_len=cfg.MODEL.OUT_LEN)
    encoder_net, forecaster_net, loss_net = \
        encoder_forecaster_build_networks(
            factory=mnist_rnn,
            context=args.ctx)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    states = EncoderForecasterStates(factory=mnist_rnn, ctx=args.ctx[0])
    states.reset_all()
    # Begin to load the model if load_dir is not empty
    assert len(cfg.MODEL.LOAD_DIR) > 0
    load_encoder_forecaster_params(
        load_dir=cfg.MODEL.LOAD_DIR, load_iter=cfg.MODEL.LOAD_ITER,
        encoder_net=encoder_net, forecaster_net=forecaster_net)
    for iter_id in range(1):
        frame_dat, _ = mnist_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
        data_nd = mx.nd.array(frame_dat[0:cfg.MOVINGMNIST.IN_LEN, ...], ctx=args.ctx[0]) / 255.0
        target_nd = mx.nd.array(
            frame_dat[cfg.MOVINGMNIST.IN_LEN:(cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN), ...],
            ctx=args.ctx[0]) / 255.0
        pred_nd = mnist_get_prediction(data_nd=data_nd, states=states,
                                       encoder_net=encoder_net,
                                       forecaster_net=forecaster_net)
        save_gif(pred_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "pred.gif"))
        save_gif(data_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "in.gif"))
        save_gif(target_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "gt.gif"))

def test(args):
    cfg.MODEL.TRAJRNN.SAVE_MID_RESULTS = False
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    base_dir = args.save_dir
    logging_config(folder=base_dir, name="testing")
    save_movingmnist_cfg(base_dir)
    batch_size = 4
    mnist_iter = MovingMNISTAdvancedIterator(
        distractor_num=cfg.MOVINGMNIST.DISTRACTOR_NUM,
        initial_velocity_range=(cfg.MOVINGMNIST.VELOCITY_LOWER,
                                cfg.MOVINGMNIST.VELOCITY_UPPER),
        rotation_angle_range=(cfg.MOVINGMNIST.ROTATION_LOWER,
                              cfg.MOVINGMNIST.ROTATION_UPPER),
        scale_variation_range=(cfg.MOVINGMNIST.SCALE_VARIATION_LOWER,
                               cfg.MOVINGMNIST.SCALE_VARIATION_UPPER),
        illumination_factor_range=(cfg.MOVINGMNIST.ILLUMINATION_LOWER,
                                   cfg.MOVINGMNIST.ILLUMINATION_UPPER))
    mnist_iter.load(file=cfg.MOVINGMNIST.TEST_FILE)
    mnist_rnn = MovingMNISTFactory(batch_size=batch_size,
                                   in_seq_len=cfg.MODEL.IN_LEN,
                                   out_seq_len=cfg.MODEL.OUT_LEN)
    encoder_net, forecaster_net, loss_net = \
        encoder_forecaster_build_networks(
            factory=mnist_rnn,
            context=args.ctx)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    states = EncoderForecasterStates(factory=mnist_rnn, ctx=args.ctx[0])
    states.reset_all()
    # Begin to load the model if load_dir is not empty
    assert len(cfg.MODEL.LOAD_DIR) > 0
    load_encoder_forecaster_params(
        load_dir=cfg.MODEL.LOAD_DIR, load_iter=cfg.MODEL.LOAD_ITER,
        encoder_net=encoder_net, forecaster_net=forecaster_net)
    overall_mse = 0
    for iter_id in range(10000 // batch_size):
        frame_dat, _ = mnist_iter.sample(batch_size=batch_size,
                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN,
                                         random=False)
        data_nd = mx.nd.array(frame_dat[0:cfg.MOVINGMNIST.IN_LEN, ...], ctx=args.ctx[0]) / 255.0
        target_nd = mx.nd.array(
            frame_dat[cfg.MOVINGMNIST.IN_LEN:(cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN), ...],
            ctx=args.ctx[0]) / 255.0
        pred_nd = mnist_get_prediction(data_nd=data_nd, states=states,
                                       encoder_net=encoder_net, forecaster_net=forecaster_net)
        overall_mse += mx.nd.mean(mx.nd.square(pred_nd - target_nd)).asscalar()
        print(iter_id, overall_mse / (iter_id + 1))
    avg_mse = overall_mse / (10000 // batch_size)
    with open(os.path.join(base_dir, 'result.txt'), 'w') as f:
        f.write(str(avg_mse))
    print(base_dir, avg_mse)

if __name__ == "__main__":
    backup_test_file_path = cfg.MOVINGMNIST.TEST_FILE
    args = parse_args()
    cfg.MOVINGMNIST.TEST_FILE = backup_test_file_path
    test(args)
