import logging
import os
import mxnet as mx
import numpy as np
import scipy.stats
import pickle
from ..utils import *

class IdentityOp(mx.operator.CustomOp):
    def __init__(self, logging_prefix="identity", input_debug=False, grad_debug=False):
        super(IdentityOp, self).__init__()
        self.logging_prefix=logging_prefix
        self.input_debug = input_debug
        self.grad_debug = grad_debug

    def forward(self, is_train, req, in_data, out_data, aux):
        if(self.input_debug):
            logging.info("%s: in_norm=%f, in_max=%f, in_mean=%f, in_min=%f, in_shape=%s"
                          %(self.logging_prefix, np.linalg.norm(in_data[0].asnumpy()), in_data[0].asnumpy().max(), np.abs(in_data[0].asnumpy()).mean(), in_data[0].asnumpy().min(), str(in_data[0].shape)))
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])
        if (self.grad_debug):
            logging.info("%s: grad_norm=%f, grad_shape=%s"
                          % (self.logging_prefix, np.linalg.norm(in_grad[0].asnumpy()), str(in_grad[0].shape)))


@mx.operator.register("identity")
class IdentityOpProp(mx.operator.CustomOpProp):
    def __init__(self, logging_prefix="identity", input_debug=False, grad_debug=False):
        super(IdentityOpProp, self).__init__(need_top_grad=True)
        self.input_debug = safe_eval(input_debug)
        self.grad_debug = safe_eval(grad_debug)
        self.logging_prefix = str(logging_prefix)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return IdentityOp(input_debug=self.input_debug,
                          grad_debug=self.grad_debug,
                          logging_prefix=self.logging_prefix)

class SaveNpyOp(mx.operator.CustomOp):
    def __init__(self, save_name="op", save_dir=None):
        super(SaveNpyOp, self).__init__()
        self._save_name = save_name
        self._save_dir = '.' if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._input_save_path = os.path.join(self._save_dir, '{}.npy'.format(save_name))
        self._grad_save_path = os.path.join(self._save_dir, '{}_grad.npy'.format(save_name))

    def forward(self, is_train, req, in_data, out_data, aux):
        logging.info("Saving Input to {}".format(os.path.realpath(self._input_save_path)))
        np.save(self._input_save_path, in_data[0].asnumpy())
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        logging.info("Saving Gradient to {}".format(os.path.realpath(self._input_save_path)))
        np.save(self._grad_save_path, out_grad[0].asnumpy())
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("save_npy")
class SaveNpyOpProp(mx.operator.CustomOpProp):
    def __init__(self, save_name="op", save_dir="."):
        super(SaveNpyOpProp, self).__init__(need_top_grad=True)
        self._save_name = save_name
        self._save_dir = save_dir

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return SaveNpyOp(save_name=self._save_name,
                         save_dir=self._save_dir)

class ConstantOp(mx.operator.CustomOp):
    """Implementation of mask on minibatch layer.
    """
    def __init__(self, data):
        super(ConstantOp, self).__init__()
        self.data = data

    def forward(self, is_train, req, in_data, out_data, aux):
        if self.data.context != out_data[0].context:
            self.data = self.data.copyto(out_data[0].context)
        self.assign(out_data[0], req[0], self.data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise RuntimeError("cannot bp to constant")


@mx.operator.register("constant")
class ConstantOpProp(mx.operator.CustomOpProp):
    def __init__(self, pkl_data):
        super(ConstantOpProp, self).__init__(need_top_grad=False)
        self.data = pickle.loads(pkl_data)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [self.data.shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ConstantOp(mx.nd.array(self.data, ctx=ctx))


class LogisticRegressionMaskOutput(mx.operator.CustomOp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutput, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 1.0 / (1.0 + nd.exp(- in_data[0])))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        output = out_data[0].asnumpy()
        label = in_data[1].asnumpy()
        data_grad = (output - label) * (label != self.ignore_label)
        self.assign(in_grad[0], req[0], data_grad)

@mx.operator.register("LogisticRegressionMaskOutput")
class LogisticRegressionMaskOutputProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutputProp, self).__init__(need_top_grad=False)
        self.ignore_label = safe_eval(ignore_label)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogisticRegressionMaskOutput(ignore_label=self.ignore_label)

class EntropyMultinomialDist(mx.operator.CustomOp):
    def __init__(self):
        super(EntropyMultinomialDist, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], scipy.stats.entropy(in_data[0].asnumpy().T))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        p = in_data[0]
        p_sum = nd.sum(p, axis=1, keepdims=True)
        logit = nd.log(p / p_sum)
        grad = - logit / p_sum + nd.sum(p * logit, axis=1, keepdims=True) / nd.square(p_sum)
        grad[:] = nd.expand_dims(out_grad[0], axis=1) * grad
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("entropy_multinomial")
class EntropyMultinomialDistProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(EntropyMultinomialDistProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0],)
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return EntropyMultinomialDist()


def logistic_regression_mask_output(data, label, ignore_label, name=None):
    return mx.sym.Custom(name=name,
                         op_type="LogisticRegressionMaskOutput",
                         ignore_label=ignore_label,
                         data=data,
                         label=label)


def constant(data, name="constant"):
    if isinstance(data, mx.nd.NDArray):
        data = data.asnumpy()
    pkl_data = pickle.dumps(data)
    return mx.symbol.Custom(name=name,
                            op_type="constant",
                            pkl_data=pkl_data)


def identity(data, name="identity", logging_prefix=None,
             input_debug=False, grad_debug=False):
    return mx.symbol.Custom(data=data,
                            name=name,
                            logging_prefix=name,
                            input_debug=input_debug,
                            grad_debug=grad_debug,
                            op_type="identity")


def save_npy(data, save_name="op", save_dir="."):
    return mx.symbol.Custom(data=data,
                            save_name=save_name,
                            save_dir=save_dir,
                            op_type="save_npy")


def entropy_multinomial(data, name="entropy"):
    return mx.symbol.Custom(name=name,
                            op_type="entropy_multinomial",
                            data=data)


def grid_generator(batch_size, height, width, normalize=True):
    """Generate the grid based on width and height

    Parameters
    ----------
    batch_size : int
    width : int
    height : int
    normalize : bool
        Whether to normalize the grid elements into [-1, 1]

    Returns
    -------
    ret : mx.sym.Symbol
        Shape : (batch_size, 2, height, width), the channel contains (x, y)
    """
    x = mx.sym.arange(start=0, stop=width)
    y = mx.sym.arange(start=0, stop=height)
    x = mx.sym.broadcast_to(mx.sym.Reshape(x, shape=(1, 1, 1, width)),
                            shape=(batch_size, 1, height, width))
    y = mx.sym.broadcast_to(mx.sym.Reshape(y, shape=(1, 1, height, 1)),
                            shape=(batch_size, 1, height, width))
    if normalize:
        x = x / float(width - 1) * 2.0 - 1.0
        y = y / float(height - 1) * 2.0 - 1.0
    ret = mx.sym.Concat(x, y, num_args=2, dim=1)
    return ret


def normalize_grid(un_norm_grid, width, height):
    """Normalize the grid to [-1, 1]

    Parameters
    ----------
    un_norm_grid : mx.sym.Symbol
        Shape : (batch_size, 2, height, width)
    width : int
    height : int

    Returns
    -------
    ret : mx.sym.Symbol
    """
    un_norm_grid = mx.sym.SliceChannel(un_norm_grid, axis=1, num_outputs=2, squeeze_axis=False)
    x = un_norm_grid[0] / float(width - 1) * 2.0 - 1.0
    y = un_norm_grid[1] / float(height - 1) * 2.0 - 1.0
    ret = mx.sym.Concat(x, y, num_args=2, dim=1)
    return ret


def multi_segment_slice_axis(data, axis, segment_lengths):
    """Split the data to multiple segments

    Parameters
    ----------
    data : mx.sym.Symbol
    axis : int
    segment_lengths : list or tuple
        Get the segment_lengths
    Returns
    -------
    ret : list
    """
    ret = []
    begin = 0
    for length in segment_lengths:
        seg_ele = mx.sym.slice_axis(data=data, axis=axis, begin=begin, end=begin + length)
        ret.append(seg_ele)
        begin += length
    return tuple(ret)


def group_add(lhs, rhs):
    """

    Parameters
    ----------
    lhs : list of mx.sym.Symbol
    rhs : list of mx.sym.Symbol

    Returns
    -------
    ret : list of mx.sym.Symbol
    """
    if isinstance(lhs, mx.sym.Symbol):
        return lhs + rhs
    assert len(lhs) == len(rhs)
    ret = []
    for i in range(len(lhs)):
        if isinstance(lhs[i], list):
            ret.append(group_add(lhs[i], rhs[i]))
        else:
            ret.append(lhs[i] + rhs[i])
    return ret


def one_step_diff(dat, axis):
    """

    Parameters
    ----------
    dat : mx.sym.Symbol
    axes : tuple

    Returns
    -------

    """
    return mx.sym.slice_axis(dat, axis=axis, begin=0, end=-1) - \
           mx.sym.slice_axis(dat, axis=axis, begin=1, end=None)


def masked_gdl_loss(pred, gt, mask):
    """

    Parameters
    ----------
    pred : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    gt : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    mask : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)

    Returns
    -------
    gdl : mx.sym.Symbol
        Shape: (seq_len, batch_size)
    """
    valid_mask_h = mx.sym.slice_axis(mask, axis=3, begin=0, end=-1) *\
                   mx.sym.slice_axis(mask, axis=3, begin=1, end=None)
    valid_mask_w = mx.sym.slice_axis(mask, axis=4, begin=0, end=-1) *\
                   mx.sym.slice_axis(mask, axis=4, begin=1, end=None)
    pred_diff_h = mx.sym.abs(one_step_diff(pred, axis=3))
    pred_diff_w = mx.sym.abs(one_step_diff(pred, axis=4))
    gt_diff_h = mx.sym.abs(one_step_diff(gt, axis=3))
    gt_diff_w = mx.sym.abs(one_step_diff(gt, axis=4))
    gd_h = mx.sym.abs(pred_diff_h - gt_diff_h)
    gd_w = mx.sym.abs(pred_diff_w - gt_diff_w)
    gdl = mx.sym.sum(valid_mask_h * gd_h, axis=(2, 3, 4)) +\
          mx.sym.sum(valid_mask_w * gd_w, axis=(2, 3, 4))
    return gdl


def weighted_l2(pred, gt, weight):
    """
    
    Parameters
    ----------
    pred : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    gt : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    weight : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)

    Returns
    -------
    l2 : mx.nd.NDArray
        Shape: (seq_len, batch_size)
    """
    l2 = weight * mx.sym.square(pred - gt)
    l2 = mx.sym.sum(l2, axis=(2, 3, 4))
    return l2


def weighted_mse(pred, gt, weight):
    return weighted_l2(pred, gt, weight)


def weighted_l1(pred, gt, weight):
    l1 = weight * mx.sym.abs(pred - gt)
    l1 = mx.sym.sum(l1, axis=(2, 3, 4))
    return l1


def weighted_mae(pred, gt, weight):
    return weighted_l1(pred, gt, weight)


def masked_hit_miss_counts(pred, gt, mask, thresholds):
    """
    
    Parameters
    ----------
    pred : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    gt : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    mask : mx.sym.Symbol
        Shape: (seq_len, batch_size, 1, H, W)
    thresholds : list

    Returns
    -------
    hits : mx.nd.NDArray
        Shape: (seq_len, batch_size, len(thresholds))
    misses : mx.nd.NDArray
        Shape: (seq_len, batch_size, len(thresholds))
    false_alarms : mx.nd.NDArray
        Shape: (seq_len, batch_size, len(thresholds))
    correct_negatives : mx.nd.NDArray
        Shape: (seq_len, batch_size, len(thresholds))
    """
    from nowcasting.hko_evaluation import rainfall_to_pixel
    thresholds = [rainfall_to_pixel(threshold) for threshold in thresholds]
    hits = []
    misses = []
    false_alarms = []
    correct_negatives = []
    for threshold in thresholds:
        pred_rain_mask = pred > threshold
        gt_rain_mask = gt > threshold
        hits_ele = pred_rain_mask * gt_rain_mask * mask
        misses_ele = (1 - pred_rain_mask) * gt_rain_mask * mask
        false_alarms_ele = pred_rain_mask * (1 - gt_rain_mask) * mask
        correct_negatives_ele = (1 - pred_rain_mask) * (1 - gt_rain_mask) * mask
        hits.append(mx.sym.sum(hits_ele, axis=(3, 4)))
        misses.append(mx.sym.sum(misses_ele, axis=(3, 4)))
        false_alarms.append(mx.sym.sum(false_alarms_ele, axis=(3, 4)))
        correct_negatives.append(mx.sym.sum(correct_negatives_ele, axis=(3, 4)))
    hits = mx.sym.concat(*hits, dim=2, num_args=len(thresholds))
    misses = mx.sym.concat(*misses, dim=2, num_args=len(thresholds))
    false_alarms = mx.sym.concat(*false_alarms, dim=2, num_args=len(thresholds))
    correct_negatives = mx.sym.concat(*correct_negatives, dim=2, num_args=len(thresholds))
    return hits, misses, false_alarms, correct_negatives
