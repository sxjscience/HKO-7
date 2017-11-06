import mxnet as mx
import numpy as np
from nowcasting.config import cfg

class ParamsReg(object):
    def __init__(self):
        self._params = {}
        self._old_params = []

    def get(self, name, **kwargs):
        if name not in self._params:
            self._params[name] = mx.sym.Variable(name, dtype=np.float32, **kwargs)
        return self._params[name]

    def get_inner(self):
        return self._params

    def reset(self):
        self._old_params.append(self._params)
        self._params = {}


_params = ParamsReg()


def reset_regs():
    global _params
    _params.reset()


def activation(data, act_type, name=None):
    if act_type == "leaky":
        if name is None:
            act = mx.sym.LeakyReLU(data=data, slope=0.2)
        else:
            act = mx.sym.LeakyReLU(data=data, slope=0.2, name='%s_%s' %(name, act_type))
        return act
    elif act_type == "identity":
        act = data
    else:
        if name is None:
            act = mx.sym.Activation(data=data, act_type=act_type)
        else:
            act = mx.sym.Activation(data=data, act_type=act_type, name='%s_%s' % (name, act_type))
    return act


def conv2d(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), no_bias=False,
           name=None, **kwargs):
    assert name is not None
    global _params
    weight = _params.get('%s_weight' % name, **kwargs)
    if no_bias:
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                  weight=weight, dilate=dilate, no_bias=True,
                                  pad=pad, name=name, workspace=256)
    else:
        bias = _params.get('%s_bias' % name, wd_mult=0.0, **kwargs)
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                  weight=weight, bias=bias, dilate=dilate, no_bias=no_bias,
                                  pad=pad, name=name, workspace=256)
    return conv


def conv2d_bn_act(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),  dilate=(1, 1),
                  no_bias=False, act_type="relu", momentum=0.9, eps=1e-5 + 1e-12, fix_gamma=True,
                  name=None, use_global_stats=False, **kwargs):
    conv = conv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                  pad=pad, dilate=dilate, no_bias=no_bias, name=name, **kwargs)
    assert name is not None
    global _params
    gamma = _params.get('%s_bn_gamma' % name, **kwargs)
    beta = _params.get('%s_bn_beta' % name, **kwargs)
    moving_mean = _params.get('%s_bn_moving_mean' % name, **kwargs)
    moving_var = _params.get('%s_bn_moving_var' % name, **kwargs)
    if fix_gamma:
        bn = mx.sym.BatchNorm(data=conv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=True,
                              momentum=momentum,
                              eps=eps,
                              name='%s_bn' %name,
                              use_global_stats=use_global_stats)
    else:
        bn = mx.sym.BatchNorm(data=conv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=False,
                              momentum=momentum,
                              eps=eps,
                              name='%s_bn' % name,
                              use_global_stats=use_global_stats)
    act = activation(bn, act_type=act_type, name=name)
    return act


def conv2d_act(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),  dilate=(1, 1),
               no_bias=False, act_type="relu", name=None, **kwargs):
    conv = conv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                  pad=pad, dilate=dilate, no_bias=no_bias, name=name, **kwargs)
    act = activation(conv, act_type=act_type, name=name)
    return act


def deconv2d(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), adj=(0, 0), no_bias=True,
             target_shape=None, name="deconv2d", **kwargs):
    global _params
    assert name is not None
    weight = _params.get('%s_weight' % name, **kwargs)
    if no_bias:
        if target_shape is None:
            deconv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, adj=adj,
                                          stride=stride,
                                          no_bias=True,
                                          weight=weight, pad=pad, name=name)
        else:
            deconv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, adj=adj,
                                          stride=stride,
                                          target_shape=target_shape, no_bias=True,
                                          weight=weight, pad=pad, name=name)
    else:
        bias = _params.get('%s_bias' % name, wd_mult=0.0, **kwargs)
        if target_shape is None:
            deconv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, adj=adj,
                                          stride=stride,
                                          no_bias=no_bias,
                                          weight=weight, bias=bias, pad=pad, name=name)
        else:
            deconv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, adj=adj,
                                          stride=stride,
                                          target_shape=target_shape, no_bias=no_bias,
                                          weight=weight, bias=bias, pad=pad, name=name)
    return deconv


def deconv2d_bn_act(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), adj=(0, 0),
                    no_bias=True, target_shape=None, act_type="relu",
                    momentum=0.9, eps=1e-5 + 1e-12, fix_gamma=True,
                    name="deconv2d", use_global_stats=False, **kwargs):
    global _params
    deconv = deconv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                      pad=pad, adj=adj, target_shape=target_shape, no_bias=no_bias, name=name, **kwargs)
    gamma = _params.get('%s_bn_gamma' % name, **kwargs)
    beta = _params.get('%s_bn_beta' % name, **kwargs)
    moving_mean = _params.get('%s_bn_moving_mean' % name, **kwargs)
    moving_var = _params.get('%s_bn_moving_var' % name, **kwargs)
    if fix_gamma:
        bn = mx.sym.BatchNorm(data=deconv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=True,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' %name)
    else:
        bn = mx.sym.BatchNorm(data=deconv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=False,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' % name)
    act = activation(bn, act_type=act_type, name=name)
    return act


def deconv2d_act(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), adj=(0, 0),
                 no_bias=True, target_shape=None, act_type="relu", name="deconv2d", **kwargs):

    deconv = deconv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                      pad=pad, adj=adj, target_shape=target_shape, no_bias=no_bias, name=name, **kwargs)
    act = activation(deconv, act_type=act_type, name=name)
    return act


def conv3d(data, num_filter, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0), dilate=(1, 1, 1), no_bias=False,
           name=None, **kwargs):
    return conv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                  pad=pad, dilate=dilate, no_bias=no_bias, name=name, **kwargs)


def conv3d_bn_act(data, num_filter, height, width, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0),
                  dilate=(1, 1, 1), no_bias=False, act_type="relu", momentum=0.9, eps=1e-5 + 1e-12,
                  fix_gamma=True, name=None, use_global_stats=False, **kwargs):
    conv = conv3d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                  pad=pad, dilate=dilate, no_bias=no_bias, name=name, **kwargs)
    assert name is not None
    global _params
    gamma = _params.get('%s_bn_gamma' % name, **kwargs)
    beta = _params.get('%s_bn_beta' % name, **kwargs)
    moving_mean = _params.get('%s_bn_moving_mean' % name, **kwargs)
    moving_var = _params.get('%s_bn_moving_var' % name, **kwargs)

    conv = mx.symbol.reshape(conv, shape=(0, 0, -1, width))

    if fix_gamma:
        bn = mx.sym.BatchNorm(data=conv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=True,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' %name)
    else:
        bn = mx.sym.BatchNorm(data=conv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=False,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' % name)

    bn = mx.symbol.reshape(bn, shape=(0, 0, -1, height, width))

    act = activation(bn, act_type=act_type, name=name)
    return act


def conv3d_act(data, num_filter, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0), dilate=(1, 1, 1),
               no_bias=False, act_type="relu", name=None, **kwargs):
    conv = conv3d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                  pad=pad, dilate=dilate, no_bias=no_bias, name=name, **kwargs)
    act = activation(conv, act_type=act_type, name=name)
    return act


def deconv3d(data, num_filter, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0), adj=(0, 0, 0), no_bias=True,
             target_shape=None, name=None, **kwargs):
    return deconv2d(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, adj=adj,
                   no_bias=no_bias, target_shape=target_shape, name=name, **kwargs)


def deconv3d_bn_act(data, num_filter, height, width, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0),
                    adj=(0, 0, 0), no_bias=True, target_shape=None, act_type="relu",
                    momentum=0.9, eps=1e-5 + 1e-12, fix_gamma=True, name=None, use_global_stats=False, **kwargs):
    global _params
    deconv = deconv3d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                      pad=pad, adj=adj, target_shape=target_shape, no_bias=no_bias, name=name, **kwargs)
    gamma = _params.get('%s_bn_gamma' % name, **kwargs)
    beta = _params.get('%s_bn_beta' % name, **kwargs)
    moving_mean = _params.get('%s_bn_moving_mean' % name, **kwargs)
    moving_var = _params.get('%s_bn_moving_var' % name, **kwargs)

    deconv = mx.symbol.reshape(deconv, shape=(0, 0, -1, width))

    if fix_gamma:
        bn = mx.sym.BatchNorm(data=deconv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=True,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' %name)
    else:
        bn = mx.sym.BatchNorm(data=deconv,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=False,
                              momentum=momentum,
                              eps=eps,
                              use_global_stats=use_global_stats,
                              name='%s_bn' % name)

    bn = mx.symbol.reshape(bn, shape=(0, 0, -1, height, width))

    act = activation(bn, act_type=act_type, name=name)
    return act


def deconv3d_act(data, num_filter, kernel=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0), adj=(0, 0, 0),
                 no_bias=True, target_shape=None, act_type="relu", name=None, **kwargs):

    deconv = deconv3d(data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                      pad=pad, adj=adj, target_shape=target_shape, no_bias=no_bias, name=name, **kwargs)
    act = activation(deconv, act_type=act_type, name=name)
    return act


def fc_layer(data, num_hidden, no_bias=False, name="fc", **kwargs):
    assert name is not None
    global _params
    weight = _params.get('%s_weight' % name, **kwargs)
    if not no_bias:
        bias = _params.get('%s_bias' % name, **kwargs)
        fc = mx.sym.FullyConnected(data=data, weight=weight, bias=bias,
                                   num_hidden=num_hidden, no_bias=False, name=name, **kwargs)
    else:
        fc = mx.sym.FullyConnected(data=data, weight=weight,
                                   num_hidden=num_hidden, no_bias=True, name=name, **kwargs)
    return fc


def fc_layer_act(data, num_hidden, no_bias=False, act_type="relu", name="fc", **kwargs):
    fc = fc_layer(data=data, num_hidden=num_hidden, no_bias=no_bias, name=name, **kwargs)
    act = activation(data=fc, act_type=act_type, name=name)
    return act


def fc_layer_bn_act(data, num_hidden, no_bias=False, act_type="relu",
                    momentum=0.9, eps=1e-5 + 1e-12, fix_gamma=True, name=None,
                    use_global_stats=False, **kwargs):
    fc = fc_layer(data=data, num_hidden=num_hidden, no_bias=no_bias, name=name, **kwargs)
    assert name is not None
    global _params
    gamma = _params.get('%s_bn_gamma' % name, **kwargs)
    beta = _params.get('%s_bn_beta' % name, **kwargs)
    moving_mean = _params.get('%s_bn_moving_mean' % name, **kwargs)
    moving_var = _params.get('%s_bn_moving_var' % name, **kwargs)
    if fix_gamma:
        bn = mx.sym.BatchNorm(data=fc,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=True,
                              momentum=momentum,
                              eps=eps,
                              name='%s_bn' %name,
                              use_global_stats=use_global_stats)
    else:
        bn = mx.sym.BatchNorm(data=fc,
                              beta=beta,
                              gamma=gamma,
                              moving_mean=moving_mean,
                              moving_var=moving_var,
                              fix_gamma=False,
                              momentum=momentum,
                              eps=eps,
                              name='%s_bn' % name,
                              use_global_stats=use_global_stats)
    act = activation(bn, act_type=act_type, name=name)
    return act


def downsample_module(data, num_filter, kernel, stride, pad, b_h_w, name, aggre_type=None):
    assert isinstance(data, list)
    data = mx.sym.concat(*data, dim=0)
    ret = conv2d_act(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                    act_type=cfg.MODEL.CNN_ACT_TYPE, name=name + "_conv")
    return ret


def upsample_module(data, num_filter, kernel, stride, pad, b_h_w, name, aggre_type=None):
    assert isinstance(data, list)
    data = mx.sym.concat(*data, dim=0)
    ret = deconv2d_act(data=data,
                       num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                       act_type=cfg.MODEL.CNN_ACT_TYPE,
                       name=name + "_deconv")
    return ret
