import numpy as np
from scipy.interpolate import NearestNDInterpolator
import mxnet as mx
import mxnet.ndarray as nd
import os
from varflow import VarFlowFactory
from nowcasting.config import cfg
from nowcasting.hko_evaluation import HKOEvaluation, pixel_to_dBZ, dBZ_to_pixel
from nowcasting.hko_benchmark import HKOBenchmarkEnv
from nowcasting.hko_iterator import precompute_mask
from nowcasting.helpers.visualization import save_hko_gif
from nowcasting.utils import logging_config


class NonLinearRoverTransform(object):
    def __init__(self, Zc=33, sharpness=4):
        self.Zc = float(Zc)
        self.sharpness = float(sharpness)

    def transform(self, img):
        dbz_img = pixel_to_dBZ(img)
        dbz_lower = pixel_to_dBZ(0.0)
        dbz_upper = pixel_to_dBZ(1.0)
        transformed_lower = np.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = np.arctan((dbz_upper - self.Zc) / self.sharpness)
        transformed_img = np.arctan((dbz_img - self.Zc) / self.sharpness)
        transformed_img = (transformed_img - transformed_lower) / \
                          (transformed_upper - transformed_lower)
        return transformed_img

    def rev_transform(self, transformed_img):
        dbz_lower = pixel_to_dBZ(0.0)
        dbz_upper = pixel_to_dBZ(1.0)
        transformed_lower = np.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = np.arctan((dbz_upper - self.Zc) / self.sharpness)
        img = transformed_img * (transformed_upper - transformed_lower) + transformed_lower
        img = np.tan(img) * self.sharpness + self.Zc
        img = dBZ_to_pixel(dBZ_img=img)
        return img


def nd_advection(im, flow):
    """

    Parameters
    ----------
    im : nd.NDArray
        Shape: (batch_size, C, H, W)
    flow : nd.NDArray
        Shape: (batch_size, 2, H, W)
    Returns
    -------
    new_im : nd.NDArray
    """
    grid = nd.GridGenerator(-flow, transform_type="warp")
    new_im = nd.BilinearSampler(im, grid)
    return new_im


def nearest_neighbor_advection(im, flow):
    """
    
    Parameters
    ----------
    im : np.ndarray
        Shape: (batch_size, C, H, W)
    flow : np.ndarray
        Shape: (batch_size, 2, H, W)
    Returns
    -------
    new_im : nd.NDArray
    """
    predict_frame = np.empty(im.shape, dtype=im.dtype)
    batch_size, channel_num, height, width = im.shape
    assert channel_num == 1
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    interp_grid = np.hstack([grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))])
    for i in range(batch_size):
        flow_interpolator = NearestNDInterpolator(interp_grid, im[i].ravel())
        predict_grid = interp_grid + np.hstack([flow[i][0].reshape((-1, 1)),
                                                flow[i][1].reshape((-1, 1))])
        predict_frame[i, 0, ...] = flow_interpolator(predict_grid).reshape((height, width))
    return predict_frame


def run(pd_path=cfg.HKO_PD.RAINY_TEST,
        mode="fixed",
        interp_type="bilinear",
        nonlinear_transform=True):
    transformer = NonLinearRoverTransform()
    flow_factory = VarFlowFactory(max_level=6, start_level=0,
                                  n1=2, n2=2,
                                  rho=1.5, alpha=2000,
                                  sigma=4.5)
    assert interp_type == "bilinear", "Nearest interpolation is implemented in CPU and is too slow." \
                                      " We only support bilinear interpolation for rover."
    if nonlinear_transform:
        base_dir = os.path.join('hko7_benchmark', 'rover-nonlinear')
    else:
        base_dir = os.path.join('hko7_benchmark', 'rover-linear')
    logging_config(base_dir)
    batch_size = 1
    env = HKOBenchmarkEnv(pd_path=pd_path,
                          save_dir=base_dir,
                          mode=mode)
    counter = 0
    while not env.done:
        in_frame_dat, in_datetime_clips, out_datetime_clips, \
        begin_new_episode, need_upload_prediction = \
            env.get_observation(batch_size=batch_size)
        if need_upload_prediction:
            counter += 1
            prediction = np.zeros(shape=(cfg.HKO.BENCHMARK.OUT_LEN,) + in_frame_dat.shape[1:],
                                  dtype=np.float32)
            I1 = in_frame_dat[-2, :, 0, :, :]
            I2 = in_frame_dat[-1, :, 0, :, :]
            mask_I1 = precompute_mask(I1)
            mask_I2 = precompute_mask(I2)
            I1 = I1 * mask_I1
            I2 = I2 * mask_I2
            if nonlinear_transform:
                I1 = transformer.transform(I1)
                I2 = transformer.transform(I2)
            flow = flow_factory.batch_calc_flow(I1=I1, I2=I2)
            if interp_type == "bilinear":
                init_im = nd.array(I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2])),
                                   ctx=mx.gpu())
                nd_flow = nd.array(np.concatenate((flow[:, :1, :, :], -flow[:, 1:, :, :]), axis=1),
                                                  ctx=mx.gpu())
                nd_pred_im = nd.zeros(shape=prediction.shape)
                for i in range(cfg.HKO.BENCHMARK.OUT_LEN):
                    new_im = nd_advection(init_im, flow=nd_flow)
                    nd_pred_im[i][:] = new_im
                    init_im[:] = new_im
                prediction = nd_pred_im.asnumpy()
            elif interp_type == "nearest":
                init_im = I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2]))
                for i in range(cfg.HKO.BENCHMARK.OUT_LEN):
                    new_im = nearest_neighbor_advection(init_im, flow)
                    prediction[i, ...] = new_im
                    init_im = new_im
            if nonlinear_transform:
                prediction = transformer.rev_transform(prediction)
            env.upload_prediction(prediction=prediction)
            if counter % 10 == 0:
                save_hko_gif(in_frame_dat[:, 0, 0, :, :],
                             save_path=os.path.join(base_dir, 'in.gif'))
                save_hko_gif(prediction[:, 0, 0, :, :],
                             save_path=os.path.join(base_dir, 'pred.gif'))
                env.print_stat_readable()
                # import matplotlib.pyplot as plt
                # Q = plt.quiver(flow[1, 0, ::10, ::10], flow[1, 1, ::10, ::10])
                # plt.gca().invert_yaxis()
                # plt.show()
                # ch = raw_input()
    env.save_eval()

# Running fixed
run(cfg.HKO_PD.RAINY_TEST, mode="fixed", nonlinear_transform=True)
run(cfg.HKO_PD.RAINY_TEST, mode="fixed", nonlinear_transform=False)
run(cfg.HKO_PD.RAINY_VALID, mode="fixed", nonlinear_transform=True)
run(cfg.HKO_PD.RAINY_VALID, mode="fixed", nonlinear_transform=False)


# Running online
# run(cfg.HKO_PD.RAINY_TEST, mode="online", nonlinear_transform=True)
# run(cfg.HKO_PD.RAINY_TEST, mode="online", nonlinear_transform=False)
# run(cfg.HKO_PD.RAINY_VALID, mode="online", nonlinear_transform=True)
# run(cfg.HKO_PD.RAINY_VALID, mode="online", nonlinear_transform=False)
