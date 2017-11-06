import mxnet as mx
from nowcasting.ops import *
from nowcasting.operators.common import identity, grid_generator, group_add
from nowcasting.operators.base_rnn import MyBaseRNNCell
import numpy as np



class BaseConvRNN(MyBaseRNNCell):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type="tanh", prefix="ConvRNN", params=None):
        super(BaseConvRNN, self).__init__(prefix=prefix + "_", params=params)
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)\
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                             // self._i2h_stride[1] + 1
        print(self._prefix, self._state_height, self._state_width)
        self._curr_states = None
        self._counter = 0


class ConvRNN(BaseConvRNN):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type="leaky",
                 layer_norm=False,
                 prefix="ConvRNN",
                 params=None):
        super(ConvRNN, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_dilate=i2h_dilate,
                                      act_type=act_type,
                                      prefix=prefix,
                                      params=params)
        self._layer_norm = layer_norm
        self.i2h_weight = self.params.get('i2h_weight')
        self.i2h_bias = self.params.get('i2h_bias')
        self.h2h_weight = self.params.get('h2h_weight')
        self.h2h_bias = self.params.get('h2h_bias', init=mx.init.Normal())

    @property
    def state_info(self):
        return [{'shape': (self._batch_size, self._num_filter,
                           self._state_height, self._state_width),
                 '__layout__': "NCHW"}]

    def __call__(self, inputs, states=None, is_initial=False, ret_mid=False):
        name = '%s_t%d' % (self._prefix, self._counter)
        self._counter += 1
        states = self.begin_state()[0] if is_initial else states[0]
        assert states is not None
        if inputs is not None:
            i2h = mx.sym.Convolution(data=inputs,
                                     weight=self.i2h_weight,
                                     bias=self.i2h_bias,
                                     kernel=self._i2h_kernel,
                                     stride=self._i2h_stride,
                                     dilate=self._i2h_dilate,
                                     pad=self._i2h_pad,
                                     num_filter=self._num_filter,
                                     name="%s_i2h" % name)
        else:
            i2h = None
        h2h = mx.sym.Convolution(data=states,
                                 weight=self.h2h_weight,
                                 bias=self.h2h_bias,
                                 kernel=self._h2h_kernel,
                                 stride=(1, 1),
                                 dilate=self._h2h_dilate,
                                 pad=self._h2h_pad,
                                 num_filter=self._num_filter,
                                 name="%s_h2h" % name)
        if i2h is not None:
            if self._layer_norm:
                next_h = activation(layer_normalization(i2h + h2h,
                                                        num_filters=self._num_filter,
                                                        name=self._prefix + "ln"),
                                    act_type=self._act_type, name=name + "_state")
            else:
                next_h = activation(i2h + h2h,
                                    act_type=self._act_type, name=name + "_state")
        else:
            if self._layer_norm:
                next_h = activation(layer_normalization(h2h,
                                                        num_filters=self._num_filter,
                                                        name=self._prefix + "ln"),
                                    act_type=self._act_type, name=name + "_state")
            else:
                next_h = activation(h2h, act_type=self._act_type, name=name + "_state")
            # next_h = identity(next_h, name=name + "_state", input_debug=True, grad_debug=True)
        self._curr_states = [next_h]
        if not ret_mid:
            return next_h, [next_h]
        else:
            return next_h, [next_h], [i2h, h2h]


class ConvGRU(BaseConvRNN):
    def __init__(self, num_filter, b_h_w, zoneout=0.0,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 i2h_adj=(0, 0), no_i2h_bias=False, use_deconv=False,
                 act_type="leaky", prefix="ConvGRU", lr_mult=1.0):
        """Initializing a ConvGRU/DeconvGRU

        r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} + b_{W_r} + b_{R_r})
        u_t = \sigma(W_u \ast x_t + R_u \ast h_{t-1} + b_{W_u} + b_{R_u})
        h^\prime_t = tanh(W_h \ast x_t + r_t \circ (R_h \ast h_{t-1} + b_{R_h}) + b_{W_h})
        h_t = (1 - u_t) \circ h^\prime_t + u_t \circ h_{t-1}

        Parameters: (reset_gate, update_gate, new_mem)
            W_{i2h} = [W_r, W_u, W_h]
            b_{i2h} = [b_{W_r}, b_{W_u}, b_{W_h}]
            W_{h2h} = [R_r, R_u, R_h]
            b_{h2h} = [b_{R_r}, b_{R_u}, b_{R_h}]


        Parameters
        ----------
        num_hidden : int
        hidden_act_type : str
        name : str
        """
        super(ConvGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      i2h_dilate=i2h_dilate,
                                      act_type=act_type,
                                      prefix=prefix)
        self._no_i2h_bias = no_i2h_bias
        self._i2h_adj = i2h_adj
        self._use_deconv = use_deconv
        if self._no_i2h_bias:
            assert use_deconv
        self._zoneout = zoneout
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

    def __call__(self, inputs, states=None, is_initial=False, ret_mid=False):
        name = '%s_t%d' % (self._prefix, self._counter)
        self._counter += 1
        if is_initial:
            states = self.begin_state()[0]
        else:
            states = states[0]
        assert states is not None
        if inputs is not None:
            if self._use_deconv:
                if self._no_i2h_bias:
                    i2h = mx.sym.Deconvolution(data=inputs,
                                               weight=self.i2h_weight,
                                               kernel=self._i2h_kernel,
                                               stride=self._i2h_stride,
                                               pad=self._i2h_pad,
                                               adj=self._i2h_adj,
                                               no_bias=True,
                                               num_filter=self._num_filter * 3,
                                               name="%s_i2h" % name)
                else:
                    i2h = mx.sym.Deconvolution(data=inputs,
                                               weight=self.i2h_weight,
                                               bias=self.i2h_bias,
                                               kernel=self._i2h_kernel,
                                               stride=self._i2h_stride,
                                               pad=self._i2h_pad,
                                               adj=self._i2h_adj,
                                               num_filter=self._num_filter * 3,
                                               name="%s_i2h" % name)
            else:
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
        print("h2h_dilate=", self._h2h_dilate)
        h2h = mx.sym.Convolution(data=prev_h,
                                 weight=self.h2h_weight,
                                 bias=self.h2h_bias,
                                 no_bias=False,
                                 kernel=self._h2h_kernel,
                                 stride=(1, 1),
                                 dilate=self._h2h_dilate,
                                 pad=self._h2h_pad,
                                 num_filter=self._num_filter * 3,
                                 name="%s_h2h" % name)
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


if __name__ == '__main__':
    import numpy as np

    # Test ConvGRU
    data = mx.sym.Variable('data')
    data = mx.sym.SliceChannel(data, axis=0, num_outputs=11, squeeze_axis=True)
    conv_gru1 = ConvGRU(num_filter=100, b_h_w=(3, 40, 40),
                        prefix="conv_gru1")
    out, states = conv_gru1(inputs=data[0], is_initial=True)
    for i in range(1, 11):
        out, states = conv_gru1(inputs=data[i], states=states)
    conv_gru_forward_backward_time =\
        mx.test_utils.check_speed(out,
                                  location={'data': np.random.normal(size=(11, 3, 128, 40, 40))},
                                  N=2)
    net = mx.mod.Module(out, data_names=['data',], label_names=None, context=mx.gpu())
    net.bind(data_shapes=[('data', (11, 3, 128, 40, 40))],
             grad_req='add')
    net.init_params()
    net.forward(mx.io.DataBatch(data=[mx.random.normal(shape=(11, 3, 128, 40, 40))], label=None), is_train=False)
    print(net.get_outputs()[0].asnumpy())

    # Test ConvRNN
    data = mx.sym.Variable('data')
    data = mx.sym.SliceChannel(data, axis=0, num_outputs=11, squeeze_axis=True)
    conv_rnn1 = ConvRNN(num_filter=100, b_h_w=(3, 40, 40),
                        prefix="conv_rnn1")
    out, states = conv_rnn1(inputs=data[0], is_initial=True)
    for i in range(1, 11):
        out, states = conv_rnn1(inputs=data[i], states=states)
    conv_rnn_forward_backward_time = \
        mx.test_utils.check_speed(out,
                                  location={'data': np.random.normal(size=(11, 3, 128, 40, 40))},
                                  N=2)
    net = mx.mod.Module(out, data_names=['data', ], label_names=None, context=mx.gpu())
    net.bind(data_shapes=[('data', (11, 3, 128, 40, 40))],
             grad_req='add')
    net.init_params()
    net.forward(mx.io.DataBatch(data=[mx.random.normal(shape=(11, 3, 128, 40, 40))], label=None),
                is_train=False)
    print(net.get_outputs()[0].asnumpy())

    print("ConvGRU Time:", conv_gru_forward_backward_time)
    print("ConvRNN Time:", conv_rnn_forward_backward_time)
