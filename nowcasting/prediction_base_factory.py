import mxnet as mx
from nowcasting.config import cfg
from nowcasting.ops import reset_regs
from nowcasting.operators.common import grid_generator


class PredictionBaseFactory(object):
    def __init__(self, batch_size, in_seq_len, out_seq_len, height, width, name="forecaster"):
        self._out_typ = cfg.MODEL.OUT_TYPE
        self._batch_size = batch_size
        self._in_seq_len = in_seq_len
        self._out_seq_len = out_seq_len
        self._height = height
        self._width = width
        self._name = name
        self._spatial_grid = grid_generator(batch_size=batch_size, height=height, width=width)
        self.rnn_list = self._init_rnn()
        self._reset_rnn()

    def _pre_encode_frame(self, frame_data, seqlen):
        ret = mx.sym.Concat(frame_data,
                             mx.sym.broadcast_to(mx.sym.expand_dims(self._spatial_grid, axis=0),
                                                 shape=(seqlen, self._batch_size,
                                                        2, self._height, self._width)),
                             mx.sym.ones(shape=(seqlen, self._batch_size, 1,
                                                self._height, self._width)),
                             num_args=3, dim=2)
        return ret

    def _init_rnn(self):
        raise NotImplementedError

    def _reset_rnn(self):
        for rnn in self.rnn_list:
            rnn.reset()

    def reset_all(self):
        reset_regs()
        self._reset_rnn()


class RecursiveOneStepBaseFactory(PredictionBaseFactory):
    def __init__(self, batch_size, in_seq_len, out_seq_len, height, width, use_ss=False,
                 name="forecaster"):
        super(RecursiveOneStepBaseFactory, self).__init__(batch_size=batch_size,
                                                          in_seq_len=in_seq_len,
                                                          out_seq_len=out_seq_len,
                                                          height=height,
                                                          width=width,
                                                          name=name)
        self._use_ss = False

