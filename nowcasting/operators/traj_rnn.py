import mxnet as mx
import logging
from nowcasting.ops import *
from nowcasting.operators.common import identity, grid_generator, group_add, constant, save_npy
from nowcasting.operators.conv_rnn import BaseConvRNN
import numpy as np


def flow_conv(data, num_filter, flows, weight, bias, name):
    assert isinstance(flows, list)
    warpped_data = []
    for i in range(len(flows)):
        flow = flows[i]
        grid = mx.sym.GridGenerator(data=-flow, transform_type="warp")
        ele_dat = mx.sym.BilinearSampler(data=data, grid=grid)
        warpped_data.append(ele_dat)
    data = mx.sym.concat(*warpped_data, dim=1)
    ret = mx.sym.Convolution(data=data,
                             num_filter=num_filter,
                             kernel=(1, 1),
                             weight=weight,
                             bias=bias,
                             name=name)
    return ret


class TrajGRU(BaseConvRNN):
    def __init__(self, b_h_w, num_filter, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type="leaky",
                 prefix="TrajGRU", lr_mult=1.0):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix=prefix)
        self._L = L
        self._zoneout = zoneout
        self.i2f_conv1_weight = self.params.get("i2f_conv1_weight", lr_mult=lr_mult)
        self.i2f_conv1_bias = self.params.get("i2f_conv1_bias", lr_mult=lr_mult)
        self.h2f_conv1_weight = self.params.get("h2f_conv1_weight", lr_mult=lr_mult)
        self.h2f_conv1_bias = self.params.get("h2f_conv1_bias", lr_mult=lr_mult)
        self.f_conv2_weight = self.params.get("f_conv2_weight", lr_mult=lr_mult)
        self.f_conv2_bias = self.params.get("f_conv2_bias", lr_mult=lr_mult)
        if cfg.MODEL.TRAJRNN.INIT_GRID:
            logging.info("TrajGRU: Initialize Grid Using Zeros!")
            self.f_out_weight = self.params.get("f_out_weight",
                                                lr_mult=lr_mult * cfg.MODEL.TRAJRNN.FLOW_LR_MULT,
                                                init=mx.init.Zero())
            self.f_out_bias = self.params.get("f_out_bias",
                                              lr_mult=lr_mult * cfg.MODEL.TRAJRNN.FLOW_LR_MULT,
                                              init=mx.init.Zero())
        else:
            self.f_out_weight = self.params.get("f_out_weight", lr_mult=lr_mult)
            self.f_out_bias = self.params.get("f_out_bias", lr_mult=lr_mult)
        self.i2h_weight = self.params.get("i2h_weight", lr_mult=lr_mult)
        self.i2h_bias = self.params.get("i2h_bias", lr_mult=lr_mult)
        self.h2h_weight = self.params.get("h2h_weight", lr_mult=lr_mult)
        self.h2h_bias = self.params.get("h2h_bias", lr_mult=lr_mult)

    @property
    def state_postfix(self):
        return ['h']

    @property
    def state_info(self):
        return [{'shape': (self._batch_size, self._num_filter,
                           self._state_height, self._state_width),
                 '__layout__': "NCHW"}]

    def _flow_generator(self, inputs, states, prefix):
        if inputs is not None:
            i2f_conv1 = mx.sym.Convolution(data=inputs,
                                           weight=self.i2f_conv1_weight,
                                           bias=self.i2f_conv1_bias,
                                           kernel=(5, 5),
                                           dilate=(1, 1),
                                           pad=(2, 2),
                                           num_filter=32,
                                           name="%s_i2f_conv1" % prefix)
        else:
            i2f_conv1 = None
        h2f_conv1 = mx.sym.Convolution(data=states,
                                       weight=self.h2f_conv1_weight,
                                       bias=self.h2f_conv1_bias,
                                       kernel=(5, 5),
                                       dilate=(1, 1),
                                       pad=(2, 2),
                                       num_filter=32,
                                       name="%s_h2f_conv1" % prefix)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = activation(f_conv1, act_type=self._act_type)
        # f_conv2 = mx.sym.Convolution(data=f_conv1,
        #                              weight=self.f_conv2_weight,
        #                              bias=self.f_conv2_bias,
        #                              kernel=(5, 5),
        #                              dilate=(1, 1),
        #                              pad=(2, 2),
        #                              num_filter=32,
        #                              name="%s_f_conv2" %prefix)
        # f_conv2 = activation(f_conv2, act_type=self._act_type)
        flows = mx.sym.Convolution(data=f_conv1,
                                   weight=self.f_out_weight,
                                   bias=self.f_out_bias,
                                   kernel=(5, 5),
                                   pad=(2, 2),
                                   num_filter=self._L * 2)
        if cfg.MODEL.TRAJRNN.SAVE_MID_RESULTS:
            import os
            flows = save_npy(flows, save_name="%s_flow" %prefix,
                             save_dir=os.path.join(cfg.MODEL.SAVE_DIR, "flows"))
        flows = mx.sym.split(flows, num_outputs=self._L, axis=1)
        flows = [flows[i] for i in range(self._L)]
        return flows

    def __call__(self, inputs, states=None, is_initial=False, ret_mid=False):
        self._counter += 1
        name = '%s_t%d' % (self._prefix, self._counter)
        if is_initial:
            states = self.begin_state()[0]
        else:
            states = states[0]
        assert states is not None
        if inputs is not None:
            i2h = mx.sym.Convolution(data=inputs,
                                     weight=self.i2h_weight,
                                     bias=self.i2h_bias,
                                     kernel=self._i2h_kernel,
                                     stride=self._i2h_stride,
                                     dilate=self._i2h_dilate,
                                     pad=self._i2h_pad,
                                     num_filter=self._num_filter * 3,
                                     name="%s_i2h" % name)
            i2h_slice = mx.sym.SliceChannel(i2h, num_outputs=3, axis=1)
        else:
            i2h_slice = None
        prev_h = states
        flows = self._flow_generator(inputs=inputs, states=states, prefix=name)
        # flows[0] = identity(flows[0], input_debug=True)
        h2h = flow_conv(data=prev_h, num_filter=self._num_filter * 3, flows=flows,
                        weight=self.h2h_weight, bias=self.h2h_bias, name="%s_h2h" % name)
        h2h_slice = mx.sym.SliceChannel(h2h, num_outputs=3, axis=1)
        if i2h_slice is not None:
            reset_gate = mx.sym.Activation(i2h_slice[0] + h2h_slice[0], act_type="sigmoid",
                                           name=name + "_r")
            update_gate = mx.sym.Activation(i2h_slice[1] + h2h_slice[1], act_type="sigmoid",
                                            name=name + "_u")
            new_mem = activation(i2h_slice[2] + reset_gate * h2h_slice[2],
                                 act_type=self._act_type,
                                 name=name + "_h")
        else:
            reset_gate = mx.sym.Activation(h2h_slice[0], act_type="sigmoid",
                                           name=name + "_r")
            update_gate = mx.sym.Activation(h2h_slice[1], act_type="sigmoid",
                                            name=name + "_u")
            new_mem = activation(reset_gate * h2h_slice[2],
                                 act_type=self._act_type,
                                 name=name + "_h")
        next_h = update_gate * prev_h + (1 - update_gate) * new_mem
        if self._zoneout > 0.0:
            mask = mx.sym.Dropout(mx.sym.ones_like(prev_h), p=self._zoneout)
            next_h = mx.sym.where(mask, next_h, prev_h)
        self._curr_states = [next_h]
        if not ret_mid:
            return next_h, [next_h]
        else:
            return next_h, [next_h], []
