from nowcasting.config import cfg, save_cfg, cfg_from_file
from nowcasting.utils import logging_config, parse_ctx
from hko_factory import HKONowcastingFactory
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, load_encoder_forecaster_params
from hko_main import run_benchmark
import os
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Test the HKO nowcasting model')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', type=str)
    parser.add_argument('--load_dir', help='The directory to load the model', default=None, type=str)
    parser.add_argument('--load_iter', help='The iterator to load', default=-1, type=int)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    parser.add_argument('--finetune', dest='finetune', help='Whether to do online finetuning',
                        default=None, type=int)
    parser.add_argument('--finetune_min_mse', dest='finetune_min_mse', help='Minimum error for finetuning',
                        default=None, type=float)
    parser.add_argument('--mode', dest='mode', help='Whether to used fixed setting or online setting',
                        required=True, type=str)
    parser.add_argument('--dataset', dest='dataset', help='Whether to used the test set or the validation set',
                        default="test", type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg.MODEL)
    if args.load_dir is not None:
        cfg.MODEL.LOAD_DIR = args.load_dir
    if args.load_iter != -1:
        cfg.MODEL.LOAD_ITER = args.load_iter
    if args.lr is not None:
        cfg.MODEL.TEST.ONLINE.LR = args.lr
    if args.wd is not None:
        cfg.MODEL.TEST.ONLINE.WD = args.wd
    if args.grad_clip is not None:
        cfg.MODEL.TEST.ONLINE.GRAD_CLIP = args.grad_clip
    if args.mode is not None:
        cfg.MODEL.TEST.MODE = args.mode
    if args.finetune is not None:
        cfg.MODEL.TEST.FINETUNE = (args.finetune != 0)
    if args.finetune_min_mse is not None:
        cfg.MODEL.TEST.ONLINE.FINETUNE_MIN_MSE = args.finetune_min_mse
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args


def test_hko(args):
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    assert len(cfg.MODEL.LOAD_DIR) > 0
    base_dir = args.save_dir
    logging_config(folder=base_dir, name="testing")
    save_cfg(dir_path=base_dir, source=cfg.MODEL)
    hko_nowcasting_online = HKONowcastingFactory(batch_size=1,
                                                 in_seq_len=cfg.MODEL.IN_LEN,
                                                 out_seq_len=cfg.MODEL.OUT_LEN)
    t_encoder_net, t_forecaster_net, t_loss_net =\
        encoder_forecaster_build_networks(
            factory=hko_nowcasting_online,
            context=args.ctx,
            for_finetune=True)
    t_encoder_net.summary()
    t_forecaster_net.summary()
    t_loss_net.summary()
    load_encoder_forecaster_params(load_dir=cfg.MODEL.LOAD_DIR,
                                   load_iter=cfg.MODEL.LOAD_ITER,
                                   encoder_net=t_encoder_net,
                                   forecaster_net=t_forecaster_net)
    if args.dataset == "test":
        pd_path = cfg.HKO_PD.RAINY_TEST
    elif args.dataset == "valid":
        pd_path = cfg.HKO_PD.RAINY_VALID
    else:
        raise NotImplementedError
    run_benchmark(hko_factory=hko_nowcasting_online,
                  context=args.ctx[0],
                  encoder_net=t_encoder_net,
                  forecaster_net=t_forecaster_net,
                  loss_net=t_loss_net,
                  save_dir=os.path.join(base_dir, "iter%d_%s_finetune%d"
                                        % (cfg.MODEL.LOAD_ITER + 1, args.dataset,
                                           cfg.MODEL.TEST.FINETUNE)),
                  finetune=cfg.MODEL.TEST.FINETUNE,
                  mode=cfg.MODEL.TEST.MODE,
                  pd_path=pd_path)


if __name__ == "__main__":
    args = parse_args()
    test_hko(args)
