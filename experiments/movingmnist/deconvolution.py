import argparse
import logging
import os
import mxnet as mx
import numpy as np
from nowcasting.config import cfg, save_cfg
from nowcasting.utils import logging_config
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
from nowcasting.helpers.gifmaker import save_gif, save_gifs
from nowcasting.models.deconvolution import (construct_l2_loss,
                                             construct_modules, train_step,
                                             test_step, get_base_dir,
                                             model_args, training_args,
                                             mode_args, parse_mode_args,
                                             parse_training_args,
                                             parse_model_args)


### Arguments
def argument_parser():
    parser = argparse.ArgumentParser(
        description='Deconvolution baseline for MovingMNIST')

    cfg.DATASET = "MOVINGMNIST"

    mode_args(parser)
    training_args(parser)
    dataset_args(parser)
    model_args(parser)

    args = parser.parse_args()

    parse_mode_args(args)
    parse_training_args(args)
    parse_dataset_args(args)
    parse_model_args(args)

    base_dir = get_base_dir(args)
    logging_config(folder=base_dir, name="testing")
    save_cfg(base_dir, source=cfg.MODEL)

    logging.info(args)
    return args


def dataset_args(parser):
    group = parser.add_argument_group('MovingMNIST',
                                      'Configure MovingMNIST dataset.')
    group.add_argument(
        '--num_distractors',
        dest='num_distractors',
        help="Number of noise distractors for MovingMNIST++",
        type=int)


def parse_dataset_args(args):
    if args.num_distractors:
        cfg.MOVINGMNIST.DISTRACTOR_NUM = args.num_distractors


### Training
def train(args):
    base_dir = get_base_dir(args)

    ### Get modules
    generator_net, loss_net = construct_modules(args)

    ### Prepare data
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

    for i in range(cfg.MODEL.TRAIN.MAX_ITER):
        seq, flow = mnist_iter.sample(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
            seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.OUT_LEN)
        in_seq = seq[:cfg.MOVINGMNIST.IN_LEN, ...]
        gt_seq = seq[cfg.MOVINGMNIST.IN_LEN:(cfg.MOVINGMNIST.IN_LEN +
                                             cfg.MOVINGMNIST.OUT_LEN), ...]

        # Transform data to NCDHW shape needed for 3D Convolution encoder and normalize
        context_nd = mx.nd.array(in_seq) / 255.0
        gt_nd = mx.nd.array(gt_seq) / 255.0
        context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))
        gt_nd = mx.nd.transpose(gt_nd, axes=(1, 2, 0, 3, 4))

        # Train a step
        pred_nd, avg_l2, avg_real_mse, generator_grad_norm =\
            train_step(generator_net, loss_net, context_nd, gt_nd)

        # Logging
        logging.info((
            "Iter:{}, L2 Loss:{}, MSE Error:{}, Generator Grad Norm:{}").format(
                i, avg_l2, avg_real_mse, generator_grad_norm))

        logging.info("Iter:%d" % i)
        if (i + 1) % 100 == 0:
            save_gif(context_nd.asnumpy()[0, 0, :, :, :],
                     os.path.join(base_dir, "input.gif"))
            save_gif(gt_nd.asnumpy()[0, 0, :, :, :],
                     os.path.join(base_dir, "gt.gif"))
            save_gif(pred_nd.asnumpy()[0, 0, :, :, :],
                     os.path.join(base_dir, "pred.gif"))
        if cfg.MODEL.SAVE_ITER > 0 and (i + 1) % cfg.MODEL.SAVE_ITER == 0:
            generator_net.save_checkpoint(
                prefix=os.path.join(base_dir, "generator"), epoch=i)


# Testing
def recursive_generation(args):
    assert (cfg.MOVINGMNIST.OUT_LEN == 1)

    base_dir = get_base_dir(args)

    # Get modules
    generator_net, = construct_modules(args)

    # Prepare data
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

    frames = np.empty(
        shape=(cfg.MODEL.TEST.MAX_ITER, cfg.MODEL.TRAIN.BATCH_SIZE,
               cfg.MOVINGMNIST.TESTING_LEN, 1, cfg.MOVINGMNIST.IMG_SIZE,
               cfg.MOVINGMNIST.IMG_SIZE))

    for i in range(cfg.MODEL.TEST.MAX_ITER):
        seq, flow = mnist_iter.sample(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
            seqlen=cfg.MOVINGMNIST.IN_LEN + cfg.MOVINGMNIST.TESTING_LEN)
        in_seq = seq[:cfg.MOVINGMNIST.IN_LEN, ...]
        gt_seq = seq[cfg.MOVINGMNIST.IN_LEN:(cfg.MOVINGMNIST.IN_LEN +
                                             cfg.MOVINGMNIST.TESTING_LEN), ...]

        # Transform data to NCDHW shape needed for 3D Convolution encoder and normalize
        context_nd = mx.nd.array(in_seq) / 255.0
        gt_nd = mx.nd.array(gt_seq) / 255.0
        context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))
        gt_nd = mx.nd.transpose(gt_nd, axes=(1, 2, 0, 3, 4))

        # Train a step
        frames[i] = test_step(generator_net, context_nd)

    frames = frames.reshape(
        -1,
        cfg.MOVINGMNIST.TESTING_LEN,
        cfg.MOVINGMNIST.IMG_SIZE,
        cfg.MOVINGMNIST.IMG_SIZE, )
    save_gifs(frames, os.path.join(base_dir, "pred"))


def test(args):
    base_dir = get_base_dir(args)
    logging_config(folder=base_dir, name='testing')

    # Get modules
    generator_net, loss_net = construct_modules(args)

    # Prepare data
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
    num_samples, seqlen = mnist_iter.load(file=cfg.MOVINGMNIST.TEST_FILE)

    overall_mse = 0
    for iter_id in range(num_samples // cfg.MODEL.TRAIN.BATCH_SIZE):
        frame_dat, _ = mnist_iter.sample(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE, seqlen=seqlen, random=False)

        context_nd = mx.nd.array(
            frame_dat[:cfg.MOVINGMNIST.IN_LEN], ctx=args.ctx[0]) / 255.0
        gt_nd = mx.nd.array(
            frame_dat[cfg.MOVINGMNIST.IN_LEN:], ctx=args.ctx[0]) / 255.0

        # Transform data to NCDHW shape needed for 3D Convolution encoder
        context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))
        gt_nd = mx.nd.transpose(gt_nd, axes=(1, 2, 0, 3, 4))

        pred_nd = test_step(generator_net, context_nd)
        overall_mse += mx.nd.mean(mx.nd.square(pred_nd - gt_nd)).asscalar()
        print(iter_id, overall_mse / (iter_id + 1))

    avg_mse = overall_mse / (num_samples // cfg.MODEL.TRAIN.BATCH_SIZE)
    with open(os.path.join(base_dir, 'result.txt'), 'w') as f:
        f.write(str(avg_mse))
    print(base_dir, avg_mse)


if __name__ == "__main__":
    args = argument_parser()
    if not cfg.MODEL.TESTING:
        train(args)
    else:
        test(args)
