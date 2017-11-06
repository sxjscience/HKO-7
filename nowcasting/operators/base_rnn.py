import mxnet as mx
from mxnet.rnn import BaseRNNCell
from nowcasting.ops import activation
from nowcasting.operators.common import group_add

class MyBaseRNNCell(BaseRNNCell):
    def __init__(self, prefix="MyBaseRNNCell", params=None):
        super(MyBaseRNNCell, self).__init__(prefix=prefix, params=params)

    def __call__(self, inputs, states, is_initial=False, ret_mid=False):
        raise NotImplementedError()

    def reset(self):
        super(MyBaseRNNCell, self).reset()
        self._curr_states = None

    def get_current_states(self):
        return self._curr_states

    def unroll(self, length, inputs=None, begin_state=None, ret_mid=False,
               input_prefix='', layout='TC', merge_outputs=False):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or None
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If inputs is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).

            If inputs is None, Placeholder variables are
            automatically created.
        begin_state : nested list of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if None.
        input_prefix : str
            prefix for automatically created input
            placehodlers.
        layout : str
            layout of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool
            if False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

        Returns
        -------
        outputs : list of Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
        mid_info : list of Symbol
        """
        self.reset()
        assert layout == 'TNC' or layout == 'TC'
        if inputs is not None:
            if isinstance(inputs, mx.sym.Symbol):
                assert len(inputs.list_outputs()) == 1, \
                    "unroll doesn't allow grouped symbol as input. Please " \
                    "convert to list first or let unroll handle slicing"
                if 'N' in layout:
                    inputs = mx.sym.SliceChannel(inputs, axis=0, num_outputs=length,
                                                 squeeze_axis=1)
                else:
                    inputs = mx.sym.SliceChannel(inputs, axis=0, num_outputs=length)
            else:
                assert len(inputs) == length
        else:
            inputs = [None] * length
        if begin_state is None:
            states = self.begin_state()
        else:
            states = begin_state
        outputs = []
        mid_infos = []
        for i in range(length):
            output, states, mid_info = self(inputs=inputs[i], states=states,
                                            is_initial=(i == 0 and (begin_state is None)),
                                            ret_mid=True)
            outputs.append(output)
            mid_infos.extend(mid_info)
        if merge_outputs:
            outputs = [mx.sym.expand_dims(i, axis=0) for i in outputs]
            outputs = mx.sym.Concat(*outputs, dim=0)
        if ret_mid:
            return outputs, states, mid_infos
        else:
            return outputs, states


class BaseStackRNN(object):
    def __init__(self, base_rnn_class, stack_num=1,
                 name="BaseStackRNN", residual_connection=True,
                 **kwargs):
        self._base_rnn_class = base_rnn_class
        self._residual_connection = residual_connection
        self._name = name
        self._stack_num = stack_num
        self._prefix = name + "_"
        self._rnns = [base_rnn_class(prefix=self._name + "_%d" %i, **kwargs) for i in range(stack_num)]
        self._init_counter = 0
        self._state_info = None

    def init_state_vars(self):
        """Initial state variable for this cell.

        Parameters
        ----------

        Returns
        -------
        state_vars : nested list of Symbol
            starting states for first RNN step
        """
        state_vars = []
        for i, info in enumerate(self.state_info):
            state = mx.sym.var(name='%s_begin_state_%s' % (self._name, self.state_postfix[i]), **info)
            state_vars.append(state)
        return state_vars

    def concat_to_split(self, concat_states):
        assert len(concat_states) == len(self.state_info)
        split_states = [[] for i in range(self._stack_num)]
        for i in range(len(self.state_info)):
            channel_axis = self.state_info[i]['__layout__'].lower().find('c')
            ele = mx.sym.split(concat_states[i], num_outputs=self._stack_num, axis=channel_axis)
            for j in range(self._stack_num):
                split_states[j].append(ele[j])
        return split_states

    def split_to_concat(self, split_states):
        # Concat the states together
        concat_states = []
        for i in range(len(self.state_info)):
            channel_axis = self.state_info[i]['__layout__'].lower().find('c')
            concat_states.append(mx.sym.concat(*[ele[i] for ele in split_states],
                                               dim=channel_axis))
        return concat_states

    def check_concat(self, states):
        ret = not isinstance(states[0], list)
        return ret

    def to_concat(self, states):
        if not self.check_concat(states):
            states = self.split_to_concat(states)
        return states

    def to_split(self, states):
        if self.check_concat(states):
            states = self.concat_to_split(states)
        return states

    @property
    def state_postfix(self):
        return self._rnns[0].state_postfix

    @property
    def state_info(self):
        if self._state_info is None:
            info = []
            for i in range(len(self._rnns[0].state_info)):
                ele = {}
                for rnn in self._rnns:
                    if 'shape' not in ele:
                        ele['shape'] = list(rnn.state_info[i]['shape'])
                    else:
                        channel_dim = rnn.state_info[i]['__layout__'].lower().find('c')
                        ele['shape'][channel_dim] += rnn.state_info[i]['shape'][channel_dim]
                    if '__layout__' not in ele:
                        ele['__layout__'] = rnn.state_info[i]['__layout__'].upper()
                    else:
                        assert rnn.state_info[i]['__layout__'] == ele['__layout__'].upper()
                ele['shape'] = tuple(ele['shape'])
                info.append(ele)
            self._state_info = info
            return info
        else:
            return self._state_info

    def flatten_add_layout(self, states, blocked=False):
        """
        
        Parameters
        ----------
        states : list of list or list

        Returns
        -------
        ret : list
        """
        states = self.to_concat(states)
        assert self.check_concat(states)
        ret = []
        for i, ele in enumerate(states):
            if blocked:
                ret.append(mx.sym.BlockGrad(ele, __layout__=self.state_info[i]['__layout__']))
            else:
                ele._set_attr(__layout__=self.state_info[i]['__layout__'])
                ret.append(ele)
        return ret

    def reset(self):
        for i in range(len(self._rnns)):
            self._rnns[i].reset()

    def unroll(self, length, inputs=None, begin_states=None, ret_mid=False):
        if begin_states is None:
            begin_states = self.init_state_vars()
        begin_states = self.to_split(begin_states)
        assert len(begin_states) == self._stack_num, len(begin_states)
        for ele in begin_states:
            assert len(ele) == len(self.state_info)
        outputs = []
        final_states = []
        mid_infos = []
        for i in range(len(self._rnns)):
            rnn_out_list, rnn_final_states, rnn_mid_infos =\
                self._rnns[i].unroll(length=length, inputs=inputs,
                                     begin_state=begin_states[i],
                                     layout="TC",
                                     ret_mid=True)
            if self._residual_connection and i > 0:
                # Use residual connections
                rnn_out_list = group_add(lhs=rnn_out_list, rhs=inputs)
            inputs = rnn_out_list
            outputs.append(rnn_out_list)
            final_states.append(rnn_final_states)
            mid_infos.append(rnn_mid_infos)
        if ret_mid:
            return outputs, final_states, mid_infos
        else:
            return outputs, final_states


class MyGRU(MyBaseRNNCell):
    """GRU cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is None)
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    """
    def __init__(self, num_hidden, zoneout=0.0, act_type="tanh", prefix='gru_', params=None):
        super(MyGRU, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._act_type = act_type
        self._zoneout = zoneout
        self._i2h_weight = self.params.get('i2h_weight')
        self._i2h_bias = self.params.get('i2h_bias')
        self._h2h_weight = self.params.get('h2h_weight')
        self._h2h_bias = self.params.get('h2h_bias')

    @property
    def state_info(self):
        """shape(s) of states"""
        return [{'shape': (0, self._num_hidden), '__layout__': "NC"}]

    def __call__(self, inputs, states=None, is_initial=False, ret_mid=False):
        name = '%s_t%d' % (self._prefix, self._counter)
        self._counter += 1
        if is_initial:
            prev_h = self.begin_state()[0]
        else:
            prev_h = states[0]
        assert states is not None
        if inputs is not None:
            inputs = mx.sym.reshape(inputs, shape=(0, -1))
            i2h = mx.sym.FullyConnected(data=inputs,
                                        num_hidden=self._num_hidden * 3,
                                        weight=self._i2h_weight,
                                        bias=self._i2h_bias,
                                        name="%s_i2h" %name)
            i2h_slice = mx.sym.SliceChannel(i2h, num_outputs=3, axis=1)
        else:
            i2h_slice = None
        h2h = mx.sym.FullyConnected(data=prev_h,
                                    num_hidden=self._num_hidden * 3,
                                    weight=self._h2h_weight,
                                    bias=self._h2h_bias,
                                    name="%s_h2h" %name)
        h2h_slice = mx.sym.SliceChannel(h2h, num_outputs=3, axis=1)
        if i2h_slice is not None:
            reset_gate = activation(i2h_slice[0] + h2h_slice[0], act_type="sigmoid",
                                    name=name + "_r")
            update_gate = activation(i2h_slice[1] + h2h_slice[1], act_type="sigmoid",
                                     name=name + "_u")
            new_mem = activation(i2h_slice[2] + reset_gate * h2h_slice[2],
                                 act_type=self._act_type,
                                 name=name + "_h")
        else:
            reset_gate = activation(h2h_slice[0], act_type="sigmoid",
                                    name=name + "_r")
            update_gate = activation(h2h_slice[1], act_type="sigmoid",
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
    from nowcasting.operators.conv_rnn import ConvGRU
    brnn1 = BaseStackRNN(base_rnn_class=ConvGRU, stack_num=5,
                         b_h_w=(4, 32, 32), num_filter=32)
    print(brnn1.state_info)
    inputs = mx.sym.var(name="inputs", shape=(8, 4, 16, 32, 32))
    outputs, final_states, mid_infos = brnn1.unroll(length=8, inputs=inputs, ret_mid=True)
    print(len(outputs), len(outputs[0]))
    print(len(final_states), len(final_states[0]))