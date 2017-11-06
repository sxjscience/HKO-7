# TODO this is a copy of experiments/hko_factory.py and should be removed
# after nowcasting/models/deconvolution.py has been refactored to use a factory
# to get the symbols.
# Currently it needs to import the the factory directly to construct the symbol
# based on the cfg.DATASET variable

import mxnet as mx
from nowcasting.config import cfg
from nowcasting.hko_evaluation import rainfall_to_pixel
from nowcasting.encoder_forecaster import EncoderForecasterBaseFactory
from nowcasting.operators import *
from nowcasting.ops import *


def get_loss_weight_symbol(data, mask, seq_len):
    if cfg.MODEL.USE_BALANCED_LOSS:
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = mx.sym.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (data >= threshold)
        weights = weights * mask
    else:
        weights = mask
    if cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "same":
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "linear":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        temporal_mult = 1 + \
                        mx.sym.arange(start=0, stop=seq_len) * (upper - 1.0) / (seq_len - 1.0)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    elif cfg.MODEL.TEMPORAL_WEIGHT_TYPE == "exponential":
        upper = cfg.MODEL.TEMPORAL_WEIGHT_UPPER
        assert upper >= 1.0
        base_factor = np.log(upper) / (seq_len - 1.0)
        temporal_mult = mx.sym.exp(mx.sym.arange(start=0, stop=seq_len) * base_factor)
        temporal_mult = mx.sym.reshape(temporal_mult, shape=(seq_len, 1, 1, 1, 1))
        weights = mx.sym.broadcast_mul(weights, temporal_mult)
        return weights
    else:
        raise NotImplementedError

class HKONowcastingFactory(EncoderForecasterBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len,
                 name="hko_nowcasting"):
        super(HKONowcastingFactory, self).__init__(batch_size=batch_size,
                                                   in_seq_len=in_seq_len,
                                                   out_seq_len=out_seq_len,
                                                   height=cfg.HKO.ITERATOR.HEIGHT,
                                                   width=cfg.HKO.ITERATOR.WIDTH,
                                                   name=name)
        self._central_region = cfg.HKO.EVALUATION.CENTRAL_REGION

    def _slice_central(self, data):
        """Slice the central region in the given symbol
        
        Parameters
        ----------
        data : mx.sym.Symbol

        Returns
        -------
        ret : mx.sym.Symbol
        """
        x_begin, y_begin, x_end, y_end = self._central_region
        return mx.sym.slice(data,
                            begin=(0, 0, 0, y_begin, x_begin),
                            end=(None, None, None, y_end, x_end))

    def _concat_month_code(self):
        #TODO
        raise NotImplementedError

    def loss_sym(self,
                 pred=mx.sym.Variable('pred'),
                 mask=mx.sym.Variable('mask'),
                 target=mx.sym.Variable('target')):
        """Construct loss symbol.

        Optional args:
            pred: Shape (out_seq_len, batch_size, C, H, W)
            mask: Shape (out_seq_len, batch_size, C, H, W)
            target: Shape (out_seq_len, batch_size, C, H, W)
        """
        self.reset_all()
        weights = get_loss_weight_symbol(data=target, mask=mask, seq_len=self._out_seq_len)
        mse = weighted_mse(pred=pred, gt=target, weight=weights)
        mae = weighted_mae(pred=pred, gt=target, weight=weights)
        gdl = masked_gdl_loss(pred=pred, gt=target, mask=mask)
        avg_mse = mx.sym.mean(mse)
        avg_mae = mx.sym.mean(mae)
        avg_gdl = mx.sym.mean(gdl)
        global_grad_scale = cfg.MODEL.NORMAL_LOSS_GLOBAL_SCALE
        if cfg.MODEL.L2_LAMBDA > 0:
            avg_mse = mx.sym.MakeLoss(avg_mse,
                                      grad_scale=global_grad_scale * cfg.MODEL.L2_LAMBDA,
                                      name="mse")
        else:
            avg_mse = mx.sym.BlockGrad(avg_mse, name="mse")
        if cfg.MODEL.L1_LAMBDA > 0:
            avg_mae = mx.sym.MakeLoss(avg_mae,
                                      grad_scale=global_grad_scale * cfg.MODEL.L1_LAMBDA,
                                      name="mae")
        else:
            avg_mae = mx.sym.BlockGrad(avg_mae, name="mae")
        if cfg.MODEL.GDL_LAMBDA > 0:
            avg_gdl = mx.sym.MakeLoss(avg_gdl,
                                      grad_scale=global_grad_scale * cfg.MODEL.GDL_LAMBDA,
                                      name="gdl")
        else:
            avg_gdl = mx.sym.BlockGrad(avg_gdl, name="gdl")
        loss = mx.sym.Group([avg_mse, avg_mae, avg_gdl])
        return loss
