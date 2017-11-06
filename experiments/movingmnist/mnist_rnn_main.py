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
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, train_step, EncoderForecasterStates
from nowcasting.utils import parse_ctx, logging_config, load_params
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict


# random.seed(12345)
# mx.random.seed(930215)
# np.random.seed(921206)

# random.seed(1234)
# mx.random.seed(93021)
# np.random.seed(92120)

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the MovingMNIST++ dataset')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Training configuration file', default=None, type=str)
    parser.add_argument('--resume', help='Continue to train the previous model', action='store_true',
                        default=False)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg)
    if args.batch_size is not None:
        cfg.MODEL.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.MODEL.TRAIN.LR = args.lr
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    if args.grad_clip is not None:
        cfg.MODEL.TRAIN.GRAD_CLIP = args.grad_clip
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args


def save_movingmnist_cfg(dir_path):
    tmp_cfg = edict()
    tmp_cfg.MOVINGMNIST = cfg.MOVINGMNIST
    tmp_cfg.MODEL = cfg.MODEL
    save_cfg(dir_path=dir_path, source=tmp_cfg)


def load_mnist_params(load_dir, load_iter, encoder_net, forecaster_net):
    logging.info("Loading parameters from {}, Iter = {}"
                 .format(os.path.realpath(load_dir), load_iter))
    encoder_arg_params, encoder_aux_params = load_params(prefix=os.path.join(load_dir,
                                                                             "encoder_net"),
                                                         epoch=load_iter)
    encoder_net.init_params(arg_params=encoder_arg_params, aux_params=encoder_aux_params,
                            allow_missing=False, force_init=True)
    forecaster_arg_params, forecaster_aux_params = load_params(prefix=os.path.join(load_dir,
                                                                             "forecaster_net"),
                                                               epoch=load_iter)
    forecaster_net.init_params(arg_params=forecaster_arg_params,
                               aux_params=forecaster_aux_params,
                               allow_missing=False, force_init=True)
    logging.info("Loading Complete!")


def mnist_get_prediction(data_nd, states, encoder_net, forecaster_net):
    encoder_net.forward(is_train=False,
                        data_batch=mx.io.DataBatch(data=[data_nd] + states.get_encoder_states()))
    states.update(encoder_net.get_outputs())
    # Forward Forecaster
    if cfg.MODEL.OUT_TYPE == "direct":
        forecaster_net.forward(is_train=False,
                                     data_batch=mx.io.DataBatch(data=states.get_forecaster_state()))
    else:
        last_frame_nd = data_nd[data_nd.shape[0] - 1]
        forecaster_net.forward(is_train=False,
                                     data_batch=mx.io.DataBatch(data=states.get_forecaster_state() + [last_frame_nd]))
    forecaster_outputs = forecaster_net.get_outputs()
    pred_nd = forecaster_outputs[0]
    return pred_nd


def train(args):
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    base_dir = args.save_dir
    logging_config(folder=base_dir, name="training")
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

    mnist_rnn = MovingMNISTFactory(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE // len(args.ctx),
                                   in_seq_len=cfg.MODEL.IN_LEN,
                                   out_seq_len=cfg.MODEL.OUT_LEN)

    encoder_net, forecaster_net, loss_net = \
        encoder_forecaster_build_networks(
            factory=mnist_rnn,
            context=args.ctx)
    t_encoder_net, t_forecaster_net, t_loss_net = \
        encoder_forecaster_build_networks(
            factory=mnist_rnn,
            context=args.ctx[0],
            shared_encoder_net=encoder_net,
            shared_forecaster_net=forecaster_net,
            shared_loss_net=loss_net,
            for_finetune=True)
    encoder_net.summary()
    forecaster_net.summary()
    loss_net.summary()
    # Begin to load the model if load_dir is not empty
    if len(cfg.MODEL.LOAD_DIR) > 0:
        load_mnist_params(load_dir=cfg.MODEL.LOAD_DIR, load_iter=cfg.MODEL.LOAD_ITER,
                          encoder_net=encoder_net, forecaster_net=forecaster_net)
    states = EncoderForecasterStates(factory=mnist_rnn, ctx=args.ctx[0])
    states.reset_all()
    for info in mnist_rnn.init_encoder_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" %info["__layout__"]
    iter_id = 0
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        frame_dat, _ = mnist_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
        data_nd = mx.nd.array(frame_dat[0:cfg.MOVINGMNIST.IN_LEN, ...], ctx=args.ctx[0]) / 255.0
        target_nd = mx.nd.array(
            frame_dat[cfg.MODEL.IN_LEN:(cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN), ...],
            ctx=args.ctx[0]) / 255.0
        train_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                   encoder_net=encoder_net, forecaster_net=forecaster_net,
                   loss_net=loss_net, init_states=states,
                   data_nd=data_nd, gt_nd=target_nd, mask_nd=None,
                   iter_id=iter_id)
        if (iter_id + 1) % 100 == 0:
            new_frame_dat, _ = mnist_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                                         seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
            data_nd = mx.nd.array(frame_dat[0:cfg.MOVINGMNIST.IN_LEN, ...], ctx=args.ctx[0]) / 255.0
            target_nd = mx.nd.array(
                frame_dat[cfg.MOVINGMNIST.IN_LEN:(cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN), ...],
                ctx=args.ctx[0]) / 255.0
            pred_nd = mnist_get_prediction(data_nd=data_nd, states=states,
                                           encoder_net=encoder_net, forecaster_net=forecaster_net)
            save_gif(pred_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "pred.gif"))
            save_gif(data_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "in.gif"))
            save_gif(target_nd.asnumpy()[:, 0, 0, :, :], os.path.join(base_dir, "gt.gif"))
        if (iter_id + 1) % cfg.MODEL.SAVE_ITER == 0:
            encoder_net.save_checkpoint(
                prefix=os.path.join(base_dir, "encoder_net"),
                epoch=iter_id)
            forecaster_net.save_checkpoint(
                prefix=os.path.join(base_dir, "forecaster_net"),
                epoch=iter_id)
        iter_id += 1


if __name__ == "__main__":
    args = parse_args()
    train(args)
