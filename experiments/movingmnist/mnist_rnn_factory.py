from nowcasting.encoder_forecaster import *
from nowcasting.ops import *
from nowcasting.operators import *


class MovingMNISTFactory(EncoderForecasterBaseFactory):
    def __init__(self,
                 batch_size,
                 in_seq_len,
                 out_seq_len):
        super(MovingMNISTFactory, self).__init__(batch_size=batch_size,
                                                 in_seq_len=in_seq_len,
                                                 out_seq_len=out_seq_len,
                                                 height=cfg.MOVINGMNIST.IMG_SIZE,
                                                 width=cfg.MOVINGMNIST.IMG_SIZE)

    def loss_sym(self):
        self.reset_all()
        pred = mx.sym.Variable('pred')  # Shape: (out_seq_len, batch_size, 1, H, W)
        target = mx.sym.Variable('target')  # Shape: (out_seq_len, batch_size, 1, H, W)
        avg_mse = mx.sym.mean(mx.sym.square(target - pred))
        avg_mse = mx.sym.MakeLoss(avg_mse,
                                  name="mse")
        loss = mx.sym.Group([avg_mse])
        return loss
