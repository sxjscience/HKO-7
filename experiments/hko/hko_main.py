import argparse
import random
import mxnet as mx
import mxnet.ndarray as nd
from hko_factory import HKONowcastingFactory
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.my_module import MyModule
from nowcasting.hko_benchmark import *
from nowcasting.encoder_forecaster import encoder_forecaster_build_networks, train_step, EncoderForecasterStates, load_encoder_forecaster_params
from nowcasting.hko_evaluation import *
from nowcasting.utils import parse_ctx, logging_config
from nowcasting.hko_iterator import HKOIterator, precompute_mask

# Uncomment to try different seeds

# random.seed(12345)
# mx.random.seed(930215)
# np.random.seed(921206)

# random.seed(1234)
# mx.random.seed(93021)
# np.random.seed(92120)

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


def frame_skip_reshape(dat, frame_skip):
    """Reshape (seq_len, B, C, H, W) to (seq_len // frame_skip, B * frame_skip, C, H, W)
    
    Parameters
    ----------
    dat : np.ndarray
    frame_skip : int

    Returns
    -------
    ret : np.ndarray
    """
    seq_len, B, C, H, W = dat.shape
    assert seq_len % frame_skip == 0
    ret = dat.reshape((seq_len // frame_skip, frame_skip, B, -1)).transpose((0, 2, 1, 3))\
             .reshape((seq_len // frame_skip, B * frame_skip, C, H, W))
    return ret


def frame_skip_reshape_back(dat, frame_skip):
    """Reshape (seq_len, B, C, H, W) to (seq_len * frame_skip, B // frame_skip, C, H, W)

    It's the reverse operation of frame_skip_reshape
    Parameters
    ----------
    dat : np.ndarray
    frame_skip : int

    Returns
    -------
    ret : np.ndarray
    """
    seq_len, B, C, H, W = dat.shape
    assert B % frame_skip == 0
    ret = dat.reshape((seq_len, B // frame_skip, frame_skip, -1)).transpose((0, 2, 1, 3))\
             .reshape((seq_len * frame_skip, B // frame_skip, C, H, W))
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description='Train the HKO nowcasting model')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', default=None, type=str)
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
        cfg_from_file(args.cfg_file, target=cfg.MODEL)
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


def get_base_dir(args):
    if args.save_dir is not None:
        return args.save_dir
    else:
        return "encoder_forecaster_hko"


def run_benchmark(hko_factory, context, encoder_net, forecaster_net, loss_net=None,
                  sample_num=1, finetune=False, mode="fixed",
                  save_dir="hko7_rnn", pd_path=cfg.HKO_PD.RAINY_TEST):
    """Run the HKO7 Benchmark given the training sequences
    
    Parameters
    ----------
    hko_factory :
    context : mx.ctx
    encoder_net : MyModule
    forecaster_net : MyModule
    loss_net : MyModule
    sample_num : int
    finetune : bool
    mode : str
    save_dir : str
    pd_path : str

    Returns
    -------

    """
    logging.info("Begin Evaluation, mode=%s, finetune=%s, sample_num=%d,"
                 " results will be saved to %s" % (mode, str(finetune), sample_num, save_dir))
    if finetune:
        logging.info(str(cfg.MODEL.TEST.ONLINE))
    env = HKOBenchmarkEnv(pd_path=pd_path, save_dir=save_dir, mode=mode)
    states = EncoderForecasterStates(factory=hko_factory, ctx=context)

    stored_data = []
    if not cfg.MODEL.TEST.DISABLE_TBPTT:
        stored_states = []
    stored_prediction = []
    counter = 0
    finetune_iter = 0
    while not env.done:
        if finetune:
            if len(stored_data) >= 5:
                data_in = stored_data[0]
                data_gt = np.concatenate(stored_data[1:], axis=0)
                gt_mask = precompute_mask(data_gt)
                init_states = EncoderForecasterStates(factory=hko_factory, ctx=context)
                if not cfg.MODEL.TEST.DISABLE_TBPTT:
                    init_states.update(states_nd=[nd.array(ele, ctx=context) for ele in stored_states[0]])
                weights = get_balancing_weights_numba(data=data_gt, mask=gt_mask,
                                                      base_balancing_weights=cfg.HKO.EVALUATION.BALANCING_WEIGHTS,
                                                      thresholds=env._all_eval._thresholds)
                weighted_mse = (weights * np.square(stored_prediction[0] - data_gt)).sum(axis=(2, 3, 4))
                mean_weighted_mse = weighted_mse.mean()
                print("mean_weighted_mse = %g" % mean_weighted_mse)
                if mean_weighted_mse > cfg.MODEL.TEST.ONLINE.FINETUNE_MIN_MSE:
                    _, loss_dict =\
                        train_step(batch_size=1,
                                   encoder_net=encoder_net,
                                   forecaster_net=forecaster_net,
                                   loss_net=loss_net,
                                   init_states=init_states,
                                   data_nd=nd.array(data_in, ctx=context),
                                   gt_nd=nd.array(data_gt, ctx=context),
                                   mask_nd=nd.array(gt_mask, ctx=context),
                                   iter_id=finetune_iter)
                    finetune_iter += 1
                stored_data = stored_data[1:]
                stored_prediction = stored_prediction[1:]
                if not cfg.MODEL.TEST.DISABLE_TBPTT:
                    stored_states = stored_states[1:]
        if mode == "fixed" or cfg.MODEL.TEST.DISABLE_TBPTT:
            states.reset_all()
        in_frame_dat, in_datetime_clips, out_datetime_clips, begin_new_episode, need_upload_prediction =\
            env.get_observation(batch_size=1)
        if finetune:
            if begin_new_episode:
                stored_data = [in_frame_dat]
                stored_prediction = []
                if not cfg.MODEL.TEST.DISABLE_TBPTT:
                    stored_states = [[ele.asnumpy() for ele in states.get_encoder_states()]]
            else:
                stored_data.append(in_frame_dat)
                if not cfg.MODEL.TEST.DISABLE_TBPTT:
                    stored_states.append([ele.asnumpy() for ele in states.get_encoder_states()])
        in_frame_nd = nd.array(in_frame_dat, ctx=context)
        encoder_net.forward(is_train=False,
                            data_batch=mx.io.DataBatch(data=[in_frame_nd] +
                                                            states.get_encoder_states()))
        outputs = encoder_net.get_outputs()
        states.update(states_nd=outputs)
        if need_upload_prediction:
            counter += 1
            if cfg.MODEL.OUT_TYPE == "direct":
                forecaster_net.forward(is_train=False,
                                       data_batch=mx.io.DataBatch(
                                       data=states.get_forecaster_state()))
                pred_nd = forecaster_net.get_outputs()[0]
            else:
                forecaster_net.forward(is_train=False,
                                       data_batch=mx.io.DataBatch(
                                       data=states.get_forecaster_state() + [in_frame_nd[in_frame_nd.shape[0] - 1]]))
                pred_nd = forecaster_net.get_outputs()[0]
                flow_nd = forecaster_net.get_outputs()[1]
            pred_nd = nd.clip(pred_nd, a_min=0, a_max=1)
            env.upload_prediction(prediction=pred_nd.asnumpy())
            if finetune:
                stored_prediction.append(pred_nd.asnumpy())
    env.save_eval()


def train(args):
    assert cfg.MODEL.FRAME_STACK == 1 and cfg.MODEL.FRAME_SKIP == 1
    base_dir = get_base_dir(args)
    logging_config(folder=base_dir)
    save_cfg(dir_path=base_dir, source=cfg.MODEL)
    if cfg.MODEL.TRAIN.TBPTT:
        # Create a set of sequent iterators with different starting point
        train_hko_iters = []
        train_hko_iter_restart = []
        for _ in range(cfg.MODEL.TRAIN.BATCH_SIZE):
            ele_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN,
                                   sample_mode="sequent",
                                   seq_len=cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN,
                                   stride=cfg.MODEL.IN_LEN)
            ele_iter.random_reset()
            train_hko_iter_restart.append(True)
            train_hko_iters.append(ele_iter)
    else:
        train_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN,
                                     sample_mode="random",
                                     seq_len=cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN)

    hko_nowcasting = HKONowcastingFactory(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE // len(args.ctx),
                                          ctx_num=len(args.ctx),
                                          in_seq_len=cfg.MODEL.IN_LEN,
                                          out_seq_len=cfg.MODEL.OUT_LEN)
    hko_nowcasting_online = HKONowcastingFactory(batch_size=1,
                                                 in_seq_len=cfg.MODEL.IN_LEN,
                                                 out_seq_len=cfg.MODEL.OUT_LEN)
    encoder_net, forecaster_net, loss_net = \
        encoder_forecaster_build_networks(
            factory=hko_nowcasting,
            context=args.ctx)
    t_encoder_net, t_forecaster_net, t_loss_net = \
        encoder_forecaster_build_networks(
            factory=hko_nowcasting_online,
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
        load_encoder_forecaster_params(load_dir=cfg.MODEL.LOAD_DIR, load_iter=cfg.MODEL.LOAD_ITER,
                                       encoder_net=encoder_net, forecaster_net=forecaster_net)
    states = EncoderForecasterStates(factory=hko_nowcasting, ctx=args.ctx[0])
    for info in hko_nowcasting.init_encoder_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" %info["__layout__"]
    for info in hko_nowcasting.init_forecaster_state_info:
        assert info["__layout__"].find('N') == 0, "Layout=%s is not supported!" % info["__layout__"]
    test_mode = "online" if cfg.MODEL.TRAIN.TBPTT else "fixed"
    iter_id = 0
    while iter_id < cfg.MODEL.TRAIN.MAX_ITER:
        if not cfg.MODEL.TRAIN.TBPTT:
            # We are not using TBPTT, we could directly sample a random minibatch
            frame_dat, mask_dat, datetime_clips, _ = \
                train_hko_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE)
            states.reset_all()
        else:
            # We are using TBPTT, we should sample minibatches from the iterators.
            frame_dat_l = []
            mask_dat_l = []
            for i, ele_iter in enumerate(train_hko_iters):
                if ele_iter.use_up:
                    states.reset_batch(batch_id=i)
                    ele_iter.random_reset()
                    train_hko_iter_restart[i] = True
                if train_hko_iter_restart[i] == False and ele_iter.check_new_start():
                    states.reset_batch(batch_id=i)
                    ele_iter.random_reset()
                frame_dat, mask_dat, datetime_clips, _ = \
                    ele_iter.sample(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE)
                train_hko_iter_restart[i] = False
                frame_dat_l.append(frame_dat)
                mask_dat_l.append(mask_dat)
            frame_dat = np.concatenate(frame_dat_l, axis=1)
            mask_dat = np.concatenate(mask_dat_l, axis=1)
        data_nd = mx.nd.array(frame_dat[0:cfg.MODEL.IN_LEN, ...], ctx=args.ctx[0]) / 255.0
        target_nd = mx.nd.array(
            frame_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN), ...],
            ctx=args.ctx[0]) / 255.0
        mask_nd = mx.nd.array(
            mask_dat[cfg.MODEL.IN_LEN:(cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN), ...],
            ctx=args.ctx[0])
        states, _ = train_step(batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                               encoder_net=encoder_net, forecaster_net=forecaster_net,
                               loss_net=loss_net, init_states=states,
                               data_nd=data_nd, gt_nd=target_nd, mask_nd=mask_nd,
                               iter_id=iter_id)
        if (iter_id + 1) % cfg.MODEL.VALID_ITER == 0:
            run_benchmark(hko_factory=hko_nowcasting_online,
                          context=args.ctx[0],
                          encoder_net=t_encoder_net,
                          forecaster_net=t_forecaster_net,
                          loss_net=t_loss_net,
                          save_dir=os.path.join(base_dir, "iter%d_valid" % (iter_id + 1)),
                          mode=test_mode,
                          pd_path=cfg.HKO_PD.RAINY_VALID)
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
