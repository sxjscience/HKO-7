from nowcasting.config import cfg, cfg_from_file, load_latest_cfg, save_cfg
from nowcasting.utils import *  # TODO use explicit import
from nowcasting.utils import load_params
from nowcasting.ops import fc_layer, activation
from nowcasting.my_module import MyModule
from nowcasting.models.deconvolution_symbol import discriminator_symbol, generator_symbol
from nowcasting.hko_factory import HKONowcastingFactory

import os
import sys
import logging
import random
from collections import namedtuple
import mxnet as mx
import numpy as np


### Losses
def construct_l2_loss(gt, pred, normalize_gt=False):
    """Construct symbol of L2 loss.

    Used variables:
        gt: ground truth
        pred: prediction (or real data during training)

    Args:
        gt: ground truth variable
        pred: prediction (or real data during training) variable
        normalize_gt: if True divide gt by 255.0
    """

    if normalize_gt:
        gt = gt / 255.0

    if cfg.DATASET == "MOVINGMNIST":
        return mx.sym.MakeLoss(
            mx.sym.mean(mx.sym.square(gt - pred)),
            grad_scale=cfg.MODEL.L2_LAMBDA,
            name="mse")
    elif cfg.DATASET == "HKO":
        factory = HKONowcastingFactory(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
            in_seq_len=cfg.HKO.BENCHMARK.IN_LEN,
            out_seq_len=cfg.HKO.BENCHMARK.OUT_LEN)

        return factory.loss_sym(pred=pred, target=gt)


### Modules
def construct_modules(args):
    """Construct modules for training or testing mode.

    If args.testing is False, returns [generator_net, loss_net].
    Otherwise only returns [generator_net]
    """
    ### Symbol construction
    context = mx.sym.Variable('context')
    gt = mx.sym.Variable('gt')
    pred = mx.sym.Variable('pred')

    if cfg.MODEL.TESTING:
        sym_g = generator_symbol(context, momentum=1)
        sym_d = discriminator_symbol(context, pred, momentum=1)
    else:
        sym_g = generator_symbol(context)
        sym_d = discriminator_symbol(context, pred)

    sym_l2_loss = construct_l2_loss(gt, pred)

    ### Module construction
    modules = []
    module_names = []

    generator_net = MyModule(
        sym_g, data_names=('context', ), label_names=None, context=args.ctx)

    modules.append(generator_net)
    module_names.append("generator")

    loss_data_names = ['gt', 'pred']
    if cfg.DATASET == "HKO":
        loss_data_names.append('mask')

    loss_net = MyModule(
        mx.sym.Group([
            sym_l2_loss, mx.sym.BlockGrad(
                mx.sym.mean(
                    mx.sym.square(mx.sym.clip(pred, a_min=0, a_max=1) - gt)),
                name="real_mse")
        ]),
        data_names=loss_data_names,
        label_names=None,
        context=args.ctx)
    modules.append(loss_net)
    module_names.append("loss")

    ### Module binding
    # Bind generator

    if cfg.DATASET == "MOVINGMNIST":
        IN_LEN = cfg.MOVINGMNIST.IN_LEN
        OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
        IMG_SIZE = cfg.MOVINGMNIST.IMG_SIZE
    elif cfg.DATASET == "HKO":
        IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
        OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
        IMG_SIZE = cfg.HKO.ITERATOR.WIDTH

    data_shapes = {
        'context':
        mx.io.DataDesc(
            name='context',
            shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1, IN_LEN, IMG_SIZE, IMG_SIZE),
            layout="NCDHW"),
        'gt':
        mx.io.DataDesc(
            name='gt',
            shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1, OUT_LEN, IMG_SIZE, IMG_SIZE),
            layout="NCDHW"),
        'pred':
        mx.io.DataDesc(
            name='pred',
            shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1, OUT_LEN, IMG_SIZE, IMG_SIZE),
            layout="NCDHW")
    }

    if cfg.DATASET == "HKO":
        data_shapes["mask"] = mx.io.DataDesc(
            name='mask',
            shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1, OUT_LEN, IMG_SIZE, IMG_SIZE),
            layout="NCDHW")

    label_shapes = {
        'label':
        mx.io.DataDesc(name='label', shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1))
    }

    init = mx.init.Xavier(rnd_type="gaussian", magnitude=1)

    for m, name in zip(modules, module_names):
        ds = [data_shapes[name] for name in m.data_names]
        ls = [label_shapes[name] for name in m.label_names]

        if len(ls) == 0:
            ls = None

        m.bind(data_shapes=ds, label_shapes=ls, inputs_need_grad=True)

        if not cfg.MODEL.RESUME or name not in ["generator", "gan"]:
            # Only "generator" and "gan" support being restored.
            # All other modules are freshly initialized, even if RESUME == True.
            m.init_params(initializer=init)
        else:
            logging.info("Loading parameters of {} from {}, Iter = {}".format(
                name, os.path.realpath(
                    cfg.MODEL.LOAD_DIR), cfg.MODEL.LOAD_ITER))
            arg_params, aux_params = load_params(
                prefix=os.path.join(cfg.MODEL.LOAD_DIR, name),
                epoch=cfg.MODEL.LOAD_ITER)
            m.init_params(
                arg_params=arg_params,
                aux_params=aux_params,
                allow_missing=False,
                force_init=True)
            logging.info("Loading complete!")

        lr_scheduler = mx.lr_scheduler.FactorScheduler(
            step=cfg.MODEL.TRAIN.LR_DECAY_ITER,
            factor=cfg.MODEL.TRAIN.LR_DECAY_FACTOR,
            stop_factor_lr=cfg.MODEL.TRAIN.MIN_LR)

        if cfg.MODEL.TESTING and cfg.MODEL.TEST.FINETUNE:
            optimizer_name = cfg.MODEL.TEST.ONLINE.OPTIMIZER
        else:
            optimizer_name = cfg.MODEL.TRAIN.OPTIMIZER

        if optimizer_name == "adam":
            m.init_optimizer(
                optimizer="adam",
                optimizer_params={
                    'learning_rate':
                    cfg.MODEL.TEST.ONLINE.LR if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.LR,
                    'rescale_grad':
                    1.0,
                    'epsilon':
                    cfg.MODEL.TEST.ONLINE.EPS if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.EPS,
                    'lr_scheduler':
                    None if cfg.MODEL.TESTING and cfg.MODEL.TEST.FINETUNE else
                    lr_scheduler,
                    'wd':
                    cfg.MODEL.TEST.ONLINE.WD if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.WD,
                    'beta1':
                    cfg.MODEL.TEST.ONLINE.BETA1 if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.BETA1
                })
        elif optimizer_name == "rmsprop":
            m.init_optimizer(
                optimizer="adagrad",
                optimizer_params={
                    'learning_rate':
                    cfg.MODEL.TEST.ONLINE.LR if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.LR,
                    'rescale_grad':
                    1.0,
                    'epsilon':
                    cfg.MODEL.TEST.ONLINE.EPS if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.EPS,
                    'lr_scheduler':
                    None if cfg.MODEL.TESTING and cfg.MODEL.TEST.FINETUNE else
                    lr_scheduler,
                    'wd':
                    cfg.MODEL.TEST.ONLINE.WD if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.WD,
                    'gamma1':
                    cfg.MODEL.TEST.ONLINE.GAMMA1 if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.GAMMA1
                })
        elif optimizer_name == "adagrad":
            m.init_optimizer(
                optimizer="adagrad",
                optimizer_params={
                    'learning_rate':
                    cfg.MODEL.TEST.ONLINE.LR if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.LR,
                    'rescale_grad':
                    1.0,
                    'lr_scheduler':
                    None if cfg.MODEL.TESTING and cfg.MODEL.TEST.FINETUNE else
                    lr_scheduler,
                    'wd':
                    cfg.MODEL.TEST.ONLINE.WD if cfg.MODEL.TESTING and
                    cfg.MODEL.TEST.FINETUNE else cfg.MODEL.TRAIN.WD
                })
        else:
            raise NotImplementedError

        m.summary()

    return modules


### Arguments
def mode_args(parser):
    group = parser.add_argument_group('Mode',
                                      'Run in training or testing mode.')
    group.add_argument(
        '--test',
        help='Run testing code. Implies --resume.',
        action='store_true')
    group.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Optional configuration file. '
        'Given command line options will override defaults set in this configuration file.',
        type=str)
    group.add_argument('--save_dir', help='The saving directory', type=str)
    group.add_argument(
        '--resume',
        help='Continue to train the previous model. This is implied by --test.',
        action='store_true',
        default=False)
    group.add_argument(
        '--load_dir',
        help='Load model parameters from load_dir to continue training the previous model. '
        'Only honoured if --resume is specified.',
        type=str)
    group.add_argument(
        '--load_iter',
        help='Load model parameters from specified iteration.',
        type=int)
    group.add_argument(
        '--saving_postfix',
        help='The postfix of the saving directory',
        type=str)
    group.add_argument(
        '--ctx',
        dest='ctx',
        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
        type=str,
        default='gpu')


def parse_mode_args(args):
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file:
        cfg_from_file(args.cfg_file, target=cfg)
    # Parameter loading
    if args.test or cfg.MODEL.TESTING:
        cfg.MODEL.TESTING = True
        args.resume = True
    if args.resume:
        cfg.MODEL.RESUME = True
    if args.load_dir:
        cfg.MODEL.LOAD_DIR = args.load_dir
    if args.load_iter:
        cfg.MODEL.LOAD_ITER = args.load_iter


def training_args(parser):
    group = parser.add_argument_group('Training',
                                      'Configure training/testing process.')
    group.add_argument(
        '--seed',
        help="Initialize mxnet and numpy random state with this seed.",
        type=int)
    group.add_argument(
        '--batch_size',
        dest='batch_size',
        help="batchsize of the training process",
        type=int)
    group.add_argument('--lr', dest='lr', help='learning rate', type=float)
    group.add_argument('--wd', dest='wd', help='weight decay', type=float)
    group.add_argument(
        '--grad_clip',
        dest='grad_clip',
        help='gradient clipping threshold',
        type=float)
    group.add_argument(
        '--optimizer', dest='optimizer', help='optimizer to use', type=str)
    group.add_argument(
        '--l2_lambda',
        dest='l2_lambda',
        help="GAN_loss * λ_gan + L2_loss * λ_l2",
        type=float)
    group.add_argument(
        '--gan_lambda',
        dest='gan_lambda',
        help="GAN_loss * λ_gan + L2_loss * λ_l2",
        type=float)
    group.add_argument(
        '--original_gan_loss',
        dest='use_original_gan_loss',
        help="Use 2D convolutions / deconvolutions with same number of parameters as 3D model",
        action="store_true")
    group.add_argument(
        '--label_smoothing_alpha',
        dest='label_smoothing_alpha',
        help="Change one sided label smoothing α",
        type=float)
    group.add_argument(
        '--label_smoothing_beta',
        dest='label_smoothing_beta',
        help="Change two sided label smoothing β",
        type=float)


def parse_training_args(args):
    if args.batch_size:
        cfg.MODEL.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr:
        cfg.MODEL.TRAIN.LR = args.lr
    if args.wd:
        cfg.MODEL.TRAIN.WD = args.wd
    if args.grad_clip:
        cfg.MODEL.TRAIN.GRAD_CLIP = args.grad_clip
    if args.optimizer:
        cfg.MODEL.TRAIN.OPTIMIZER = args.optimizer
    if args.l2_lambda:
        cfg.MODEL.L2_LAMBDA = args.l2_lambda
    if args.seed:
        cfg.SEED = args.seed

    if cfg.SEED:
        logging.info("Fixing random seed to {}".format(cfg.SEED))
        random.seed(cfg.SEED)
        mx.random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)


def model_args(parser):
    group = parser.add_argument_group('Model',
                                      'Configure model model architecture.')
    group.add_argument(
        '--use_2d',
        dest='use_2d',
        help="Use 2D convolutions / deconvolutions with same number of parameters as 3D model",
        action="store_true")
    group.add_argument(
        '--encoder',
        dest='encoder',
        help="'share', 'separate' or 'stack'. The way to encode context frames."
    )
    group.add_argument(
        '--no_bn',
        dest='bn',
        help="Disable batch norm everywhere.",
        action="store_false")
    group.add_argument(
        '--num_filter',
        dest='num_filter',
        help="Set the base number of filters.",
        type=int)


def parse_model_args(args):
    if args.use_2d:
        cfg.MODEL.DECONVBASELINE.USE_3D = not args.use_2d
    if args.encoder:
        assert args.encoder in ["concat", "shared", "separate"]
        cfg.MODEL.DECONVBASELINE.ENCODER = args.encoder
    if args.bn:
        cfg.MODEL.DECONVBASELINE.BN = args.bn
    if args.num_filter:
        cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER = args.num_filter


def get_base_dir(args):
    if args.save_dir:
        return args.save_dir

    return "conv2d" if not cfg.MODEL.DECONVBASELINE.USE_3D else "conv3d"


### Training
def train_step(generator_net,
               loss_net,
               context_nd,
               gt_nd,
               mask_nd=None):
    """Fine-tune the encoder and forecaster for one step

    Args:
        generator_net
        loss_net
        context_nd
        gt_nd

    """
    # Forward generator
    generator_net.forward(
        is_train=True, data_batch=mx.io.DataBatch(data=[context_nd]))
    generator_outputs = dict(
        zip(generator_net.output_names, generator_net.get_outputs()))
    pred_nd = generator_outputs["pred_output"]
    # Calculate the gradient of the normal loss functions
    loss_net.forward_backward(data_batch=mx.io.DataBatch(
        data=[gt_nd, pred_nd]
        if mask_nd is None else [gt_nd, pred_nd, mask_nd]))
    loss_input_grads = dict(
        zip(loss_net.data_names, loss_net.get_input_grads()))
    pred_grad = loss_input_grads["pred"]
    loss_out = dict(zip(loss_net.output_names, loss_net.get_outputs()))
    avg_l2 = float(loss_out["mse_output"].asnumpy())
    avg_real_mse = float(loss_out["real_mse_output"].asnumpy())
    # Backward generator
    generator_net.backward(out_grads=[pred_grad])
    # Update forecaster and encoder
    generator_grad_norm = generator_net.clip_by_global_norm(
        max_norm=cfg.MODEL.TRAIN.GRAD_CLIP)
    generator_net.update()
    # encoder_net.update()
    return generator_outputs["forecast_target_output"],\
        avg_l2, avg_real_mse, generator_grad_norm


### Testing
def test_step(generator_net, context_nd):
    """Returns generated frames.

    Returns:
        shape=(cfg.MODEL.TRAIN.BATCH_SIZE, cfg.MOVINGMNIST.TESTING_LEN, 1,
               cfg.MOVINGMNIST.IMG_SIZE, cfg.MOVINGMNIST.IMG_SIZE))
    """
    if cfg.DATASET != "MOVINGMNIST":
        raise NotImplementedError

    if cfg.MOVINGMNIST.OUT_LEN == 1:
        frames = np.empty(
            shape=(cfg.MOVINGMNIST.TESTING_LEN, cfg.MODEL.TRAIN.BATCH_SIZE, 1,
                   cfg.MOVINGMNIST.IMG_SIZE, cfg.MOVINGMNIST.IMG_SIZE))

        for frame_num in range(cfg.MOVINGMNIST.TESTING_LEN):
            # Generate 1 frame
            generator_net.forward(
                data_batch=mx.io.DataBatch(data=[context_nd]), is_train=False)
            generator_outputs = dict(
                zip(generator_net.output_names, generator_net.get_outputs()))
            pred_nd = generator_outputs["pred_output"]
            pred_np = pred_nd.asnumpy()

            # Insert new last frame
            context_np = context_nd.asnumpy()
            context_np = np.roll(a=context_np, shift=-1, axis=2)
            context_np[:, :, -1, ] = pred_np[:, :, -1, ]  # Construct context
            context_nd = mx.nd.array(context_np)

            # Store generated frame
            frames[frame_num, ] = pred_np[:, :, -1, ]

        return np.moveaxis(frames, 0, 1)
    else:
        generator_net.forward(
            data_batch=mx.io.DataBatch(data=[context_nd]), is_train=False)
        generator_outputs = dict(
            zip(generator_net.output_names, generator_net.get_outputs()))
        pred_nd = generator_outputs["pred_output"]

        return pred_nd
