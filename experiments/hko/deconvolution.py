import argparse
import os
import logging
import numpy as np
import mxnet as mx
from nowcasting.config import cfg, save_cfg
from nowcasting.utils import logging_config
from nowcasting.hko_iterator import HKOIterator, precompute_mask
from nowcasting.hko_benchmark import HKOBenchmarkEnv
from nowcasting.models.deconvolution import \
    construct_l2_loss, construct_modules, \
    train_step, test_step, get_base_dir, \
    model_args, training_args, mode_args, \
    parse_mode_args, parse_training_args, parse_model_args
from nowcasting.numba_accelerated import get_balancing_weights_numba


# Arguments
def argument_parser():
    parser = argparse.ArgumentParser(
        description='Deconvolution baseline for HKO')

    cfg.DATASET = "HKO"

    mode_args(parser)
    training_args(parser)
    dataset_args(parser)
    model_args(parser)

    args = parser.parse_args()

    parse_mode_args(args)
    parse_training_args(args)
    parse_model_args(args)

    base_dir = get_base_dir(args)
    logging_config(folder=base_dir, name="training")
    save_cfg(base_dir, source=cfg.MODEL)

    logging.info(args)
    return args


def dataset_args(parser):
    group = parser.add_argument_group('HKO', 'Configure HKO dataset.')
    group.add_argument(
        '--dataset',
        default="test",
        help="Whether to used the test set or the validation set.",
        type=str)


# Benchmark
def hko_benchmark(ctx,
                  generator_net,
                  loss_net,
                  sample_num,
                  finetune=False,
                  mode="fixed",
                  save_dir="hko7_rnn",
                  pd_path=cfg.HKO_PD.RAINY_TEST):
    """Run the HKO7 Benchmark given the training sequences

    Args:
        ctx
        generator_net
        sample_num
        save_dir
        pd_path
    """
    logging.info("Begin Evaluation, sample_num=%d,"
                 " results will be saved to %s" % (sample_num, save_dir))
    if finetune:
        logging.info(str(cfg.MODEL.TEST.ONLINE))
    env = HKOBenchmarkEnv(pd_path=pd_path, save_dir=save_dir, mode=mode)

    if finetune:
        assert (mode == "online")
        data_buffer = []
        stored_prediction = []
        finetune_iter = 0

    context_nd = None

    i = 0
    while not env.done:
        logging.info("Iter {} of evaluation.".format(i))
        i += 1
        if finetune:
            if len(data_buffer) >= 5:
                context_np = data_buffer[0]  # HKO.BENCHMARK.IN_LEN frames
                gt_np = np.concatenate(data_buffer[1:], axis=0)
                gt_np = gt_np[:cfg.HKO.BENCHMARK.OUT_LEN]

                mask_np = precompute_mask(gt_np)

                weights = get_balancing_weights_numba(
                    data=gt_np,
                    mask=mask_np,
                    base_balancing_weights=cfg.HKO.EVALUATION.
                    BALANCING_WEIGHTS,
                    thresholds=env._all_eval._thresholds)
                weighted_mse = (weights *
                                np.square(stored_prediction[0] - gt_np)).sum(
                                    axis=(2, 3, 4))
                mean_weighted_mse = weighted_mse.mean()
                print("mean_weighted_mse = %g" % mean_weighted_mse)

                if mean_weighted_mse > cfg.MODEL.TEST.ONLINE.FINETUNE_MIN_MSE:
                    context_nd = mx.nd.array(context_np, ctx=ctx)
                    context_nd = mx.nd.transpose(
                        context_nd, axes=(1, 2, 0, 3, 4))
                    gt_nd = mx.nd.array(gt_np, ctx=ctx)
                    gt_nd = mx.nd.transpose(gt_nd, axes=(1, 2, 0, 3, 4))
                    mask_nd = mx.nd.array(mask_np, ctx=ctx)
                    mask_nd = mx.nd.transpose(mask_nd, axes=(1, 2, 0, 3, 4))

                    train_step(
                        generator_net=generator_net,
                        loss_net=loss_net,
                        context_nd=context_nd,
                        gt_nd=gt_nd,
                        mask_nd=mask_nd)

                    finetune_iter += 1

                del data_buffer[0]
                del stored_prediction[0]

        if mode == "online":
            context_np, in_datetime_clips, out_datetime_clips,\
                begin_new_episode, need_upload_prediction = env.get_observation(
                    batch_size=1)
            context_np = np.repeat(
                context_np, cfg.MODEL.TRAIN.BATCH_SIZE, axis=1)
            orig_size = 1

        elif mode == "fixed":
            context_np, in_datetime_clips, out_datetime_clips,\
                begin_new_episode, need_upload_prediction = env.get_observation(
                    batch_size=cfg.MODEL.TRAIN.BATCH_SIZE)
            context_nd = mx.nd.array(context_np, ctx=ctx)
            context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))

            # Pad context_nd up to batch size if needed
            orig_size = context_nd.shape[0]
            while context_nd.shape[0] < cfg.MODEL.TRAIN.BATCH_SIZE:
                context_nd = mx.nd.concat(
                    context_nd, context_nd[0:1], num_args=2, dim=0)
        else:
            raise NotImplementedError

        if finetune:
            if begin_new_episode:
                data_buffer = [context_np]
                prediction_buffer = []
            else:
                data_buffer.append(context_np)

        if mode != "fixed":
            context_nd = mx.nd.array(context_np, ctx=ctx)
            context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))
        generator_net.forward(
            is_train=False, data_batch=mx.io.DataBatch(data=[context_nd]))

        if need_upload_prediction:
            generator_outputs = dict(
                zip(generator_net.output_names, generator_net.get_outputs()))
            pred_nd = generator_outputs["pred_output"]

            pred_nd = pred_nd[0:orig_size]

            pred_nd = mx.nd.clip(pred_nd, a_min=0, a_max=1)
            pred_nd = mx.nd.transpose(pred_nd, axes=(2, 0, 1, 3, 4))

            env.upload_prediction(prediction=pred_nd.asnumpy())

            if finetune:
                stored_prediction.append(pred_nd.asnumpy())

    env.save_eval()


# Training
def train(args):
    base_dir = get_base_dir(args)

    # Get modules
    generator_net, loss_net, = construct_modules(args)

    # Prepare data
    train_hko_iter = HKOIterator(
        pd_path=cfg.HKO_PD.RAINY_TRAIN,
        sample_mode="random",
        seq_len=cfg.HKO.BENCHMARK.IN_LEN + cfg.HKO.BENCHMARK.OUT_LEN)

    start_iter = 0 if not cfg.MODEL.RESUME else cfg.MODEL.LOAD_ITER
    for i in range(start_iter, cfg.MODEL.TRAIN.MAX_ITER):
        frame_dat, mask_dat, datetime_clips, _ = train_hko_iter.sample(
            batch_size=cfg.MODEL.TRAIN.BATCH_SIZE)

        context_nd = mx.nd.array(
            frame_dat[0:cfg.HKO.BENCHMARK.IN_LEN, ...],
            ctx=args.ctx[0]) / 255.0
        gt_nd = mx.nd.array(
            frame_dat[cfg.HKO.BENCHMARK.IN_LEN:(cfg.HKO.BENCHMARK.IN_LEN +
                                                cfg.HKO.BENCHMARK.OUT_LEN)],
            ctx=args.ctx[0]) / 255.0
        mask_nd = mx.nd.array(
            mask_dat[cfg.HKO.BENCHMARK.IN_LEN:(cfg.HKO.BENCHMARK.IN_LEN +
                                               cfg.HKO.BENCHMARK.OUT_LEN)],
            ctx=args.ctx[0])

        # Transform data to NCDHW shape needed for 3D Convolution encoder and normalize
        context_nd = mx.nd.transpose(context_nd, axes=(1, 2, 0, 3, 4))
        gt_nd = mx.nd.transpose(gt_nd, axes=(1, 2, 0, 3, 4))
        mask_nd = mx.nd.transpose(mask_nd, axes=(1, 2, 0, 3, 4))

        # Train a step
        pred_nd, avg_l2, avg_real_mse, generator_grad_norm  =\
            train_step(generator_net, loss_net, context_nd, gt_nd, mask_nd)

        if (i + 1) % cfg.MODEL.VALID_ITER == 0:
            hko_benchmark(
                ctx=args.ctx[0],
                generator_net=generator_net,
                loss_net=loss_net,
                sample_num=i,
                save_dir=os.path.join(base_dir, "iter{}_valid".format(i + 1)),
                pd_path=cfg.HKO_PD.RAINY_VALID)

        # Logging
        logging.info((
            "Iter:{}, L2 Loss:{}, MSE Error:{}, Generator Grad Norm:{}").format(
                i, avg_l2, avg_real_mse, generator_grad_norm))

        if cfg.MODEL.SAVE_ITER > 0 and (i + 1) % cfg.MODEL.SAVE_ITER == 0:
            generator_net.save_checkpoint(
                prefix=os.path.join(base_dir, "generator"), epoch=i)

            hko_benchmark(
                ctx=args.ctx[0],
                generator_net=generator_net,
                loss_net=loss_net,
                sample_num=i,
                save_dir=os.path.join(base_dir, "iter{}_test".format(i + 1)),
                pd_path=cfg.HKO_PD.RAINY_TEST)


def test(args):
    assert (args.resume is True)
    if cfg.MODEL.TEST.FINETUNE:
        assert (cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS is True)

    base_dir = args.save_dir
    logging_config(folder=base_dir, name="testing")
    save_cfg(dir_path=base_dir, source=cfg.MODEL)

    generator_net, loss_net, = construct_modules(args)

    if args.dataset == "test":
        pd_path = cfg.HKO_PD.RAINY_TEST
    elif args.dataset == "valid":
        pd_path = cfg.HKO_PD.RAINY_VALID
    else:
        raise NotImplementedError

    hko_benchmark(
        ctx=args.ctx[0],
        generator_net=generator_net,
        loss_net=loss_net,
        sample_num=1,
        save_dir=os.path.join(base_dir, "iter{}_{}_finetune{}".format(
            cfg.MODEL.LOAD_ITER + 1, args.dataset, cfg.MODEL.TEST.FINETUNE)),
        finetune=cfg.MODEL.TEST.FINETUNE,
        mode=cfg.MODEL.TEST.MODE,
        pd_path=pd_path)


if __name__ == "__main__":
    args = argument_parser()
    if not cfg.MODEL.TESTING:
        train(args)
    else:
        test(args)
