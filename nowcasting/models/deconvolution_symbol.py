import mxnet as mx
from nowcasting.ops import \
    conv3d, conv3d_act, conv3d_bn_act, \
    conv2d, conv2d_act, conv2d_bn_act, \
    deconv3d, deconv3d_act, deconv3d_bn_act, \
    deconv2d, deconv2d_act, deconv2d_bn_act, \
    fc_layer, fc_layer_act
from nowcasting.config import cfg


### Network structure
def encode_net_symbol(data,
                      data_type,
                      no_bias=False,
                      momentum=0.9,
                      fix_gamma=False,
                      eps=1e-5 + 1e-12,
                      postfix=""):
    """Construct encode_net symbol.

    Args:
        data: input data (context or pred)
        data_type: If "context" use IN_LEN, if "pred" use OUT_LEN, if
            "contextpred" use IN_LEN + OUT_LEN.
        postfix: Postfix for symbol names. Parameters will be shared with
            between symbols created during calls to encode_net_symbol with same
            data_type and postfix argument,
    """

    if cfg.DATASET == "MOVINGMNIST":
        IN_LEN = cfg.MOVINGMNIST.IN_LEN
        OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
        IMG_SIZE = cfg.MOVINGMNIST.IMG_SIZE
    elif cfg.DATASET == "HKO":
        IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
        OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
        IMG_SIZE = cfg.HKO.ITERATOR.WIDTH

    # Input
    # (cfg.TRAIN.BATCH_SIZE, 1, IN_LEN, IMG_SIZE, IMG_SIZE)

    # Determine length.
    if data_type == "context":
        length = IN_LEN
    elif data_type == "pred":
        length = OUT_LEN
    elif data_type == "contextpred":
        length = IN_LEN + OUT_LEN
    else:
        raise NotImplementedError

    # Postfix for symbol names.
    postfix = "_" + data_type + "_" + postfix

    if not cfg.MODEL.DECONVBASELINE.USE_3D:
        data = mx.sym.reshape(
            data,
            shape=(cfg.MODEL.TRAIN.BATCH_SIZE, length, IMG_SIZE, IMG_SIZE))

    # Assertions
    if cfg.DATASET == "MOVINGMNIST":
        assert (length in [1, 10, 11, 20])
    elif cfg.DATASET == "HKO":
        assert (length in [1, 5, 20, 21, 25])

    k = [1, 1, 1]
    s = [1, 1, 1]
    p = [0, 0, 0]

    if cfg.DATASET == "HKO" or (cfg.DATASET == "MOVINGMNIST" and length == 20):
        # For MOVINGMNIST, if data_type == contextpred and OUT_LEN == 10 frames,
        # i.e. length == 20. If length == 11 we don't need this.
        # For HKO, if length in [20, 21, 25]
        if length > 11:
            k[0] = 4
            s[0] = 2
            p[0] = 1

        # For HKO, IMG_SIZE == 480, we scale it down to 96
        if cfg.DATASET == "HKO":
            k[1:] = [7, 7]
            s[1:] = [5, 5]
            p[1:] = [1, 1]

        data = conv2d_3d_act(
            use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
            data=data,
            name='encode_net_0' + postfix,
            act_type=cfg.MODEL.CNN_ACT_TYPE,
            kernel=k,
            stride=s,
            pad=p,
            num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER,
            no_bias=no_bias)

    # Set convolution parameters for height and width
    k[1:] = [4, 4]
    s[1:] = [2, 2]
    p[1:] = [1, 1]

    # Set convolution parameters for sequence length
    # I.e. start reducing sequence length, if input length >= 10
    if length >= 10:
        k[0] = 4
        s[0] = 2
        p[0] = 1

    # For HKO the HEIGHT and WIDTH is still 96,
    # we therefore increase stride to 3
    if cfg.DATASET == "HKO":
        s[1:] = [3, 3]

    e1 = conv2d_3d_act(
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        data=data,
        name='encode_net_1' + postfix,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER,
        no_bias=no_bias)

    # Set convolution parameters for sequence length
    # I.e. start reducing sequence length, if input length >= 5
    if length >= 5:
        k[0] = 4
        s[0] = 2
        p[0] = 1

    # Reset stride if previously changed
    if cfg.DATASET == "HKO":
        s[1:] = [2, 2]

    e2 = conv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        data=e1,
        name='encode_net_2' + postfix,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 2,
        no_bias=no_bias,
        height=16,
        width=16,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    e3 = conv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        data=e2,
        name='encode_net_3' + postfix,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 3,
        no_bias=no_bias,
        height=8,
        width=8,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    # Increase padding for sequence length
    p[0] = 2

    e4 = conv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        data=e3,
        name='encode_net_4' + postfix,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 4,
        no_bias=no_bias,
        height=4,
        width=4,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    # Output
    # (batch_size, 4 * num_filter, 1, 4, 4)
    # or in 2D case
    # (batch_size, 4 * num_filter, 4, 4)

    return e4


def video_net_symbol(encode_net,
                     no_bias=False,
                     momentum=0.9,
                     fix_gamma=False,
                     eps=1e-5 + 1e-12):
    if cfg.DATASET == "MOVINGMNIST":
        OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
    elif cfg.DATASET == "HKO":
        OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

    # Input
    # (batch_size, num_filter * 4, 1, 4, 4)
    # or in 2D case
    # (batch_size, 4 * num_filter, 4, 4)

    assert (OUT_LEN in [1, 10, 20])

    k = [1, 1, 1]
    s = [1, 1, 1]
    p = [0, 0, 0]

    if OUT_LEN > 1:
        k[0] = 2

    d1 = deconv2d_3d_act(
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        data=encode_net,
        name='video_net_d1',
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 8,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        no_bias=no_bias)

    k[1:] = [4, 4]
    s[1:] = [2, 2]
    p[1:] = [1, 1]

    if OUT_LEN >= 10:
        k[0] = 4
        s[0] = 2
        p[0] = 1

    d2 = deconv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        data=d1,
        name='video_net_d2',
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 4,
        no_bias=no_bias,
        height=8,
        width=8,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    if OUT_LEN == 10:
        p[0] = 2
    elif OUT_LEN == 20:
        p[0] = 0

    d3 = deconv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        data=d2,
        name='video_net_d3',
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER * 2,
        no_bias=no_bias,
        height=16,
        width=16,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    if OUT_LEN == 20:
        p[0] = 1

    d4 = deconv2d_3d_bn_act(
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
        use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
        use_bn=cfg.MODEL.DECONVBASELINE.BN,
        act_type=cfg.MODEL.CNN_ACT_TYPE,
        data=d3,
        name='video_net_d4',
        kernel=k,
        stride=s,
        pad=p,
        num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER,
        no_bias=no_bias,
        height=32,
        width=32,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum)

    out_filter = 1
    if OUT_LEN > 1:
        k[0] = 3
        s[0] = 1
        p[0] = 1

    # For HKO, scale up to 96 instead of 64
    if cfg.DATASET == "HKO":
        k[1:] = [5, 5]
        s[1:] = [3, 3]
        p[1:] = [1, 1]
        out_filter = 8

    if cfg.MODEL.DECONVBASELINE.USE_3D:
        gen_out = mx.sym.Deconvolution(
            data=d4,
            name='gen_out',
            kernel=k,
            stride=s,
            pad=p,
            # Generate grayscale video with only 1 channel
            num_filter=out_filter,
            no_bias=no_bias)
    else:
        gen_out = mx.sym.Deconvolution(
            data=d4,
            name='gen_out',
            kernel=k[1:],
            stride=s[1:],
            pad=p[1:],
            # Generate grayscale video with only 1 channel
            num_filter=OUT_LEN * out_filter,
            no_bias=no_bias)

    # For HKO we need to scale up further from 96 to 480
    if cfg.DATASET == "HKO":
        k[1:] = [7, 7]
        s[1:] = [5, 5]
        p[1:] = [1, 1]

        if cfg.MODEL.DECONVBASELINE.USE_3D:
            gen_out = mx.sym.Deconvolution(
                data=gen_out,
                name='gen_out_scale',
                kernel=k,
                stride=s,
                pad=p,
                # Generate grayscale video with only 1 channel
                num_filter=1 * out_filter,
                no_bias=no_bias)

        else:
            gen_out = mx.sym.Deconvolution(
                data=gen_out,
                name='gen_out_scale',
                kernel=k[1:],
                stride=s[1:],
                pad=p[1:],
                # Generate grayscale video with only 1 channel
                num_filter=OUT_LEN * out_filter,
                no_bias=no_bias)

    # For HKO we add a final refinement layer
    if cfg.DATASET == "HKO":
        k[1:] = [3, 3]
        s[1:] = [1, 1]
        p[1:] = [1, 1]

        if cfg.MODEL.DECONVBASELINE.USE_3D:
            gen_out = mx.sym.Deconvolution(
                data=gen_out,
                name='gen_out_scale2',
                kernel=k,
                stride=s,
                pad=p,
                # Generate grayscale video with only 1 channel
                num_filter=1,
                no_bias=no_bias)

        else:
            gen_out = mx.sym.Deconvolution(
                data=gen_out,
                name='gen_out_scale2',
                kernel=k[1:],
                stride=s[1:],
                pad=p[1:],
                # Generate grayscale video with only 1 channel
                num_filter=OUT_LEN,
                no_bias=no_bias)

    # Output
    # gen_out (batch_size, 1, 10, 64, 64)

    return gen_out


def generator_symbol(context,
                     no_bias=False,
                     momentum=0.9,
                     fix_gamma=False,
                     eps=1e-5 + 1e-12):

    encode_net = encode_net_symbol(
        data=context,
        data_type="context",
        no_bias=no_bias,
        momentum=momentum,
        fix_gamma=fix_gamma,
        eps=eps)

    if cfg.MODEL.DECONVBASELINE.FC_BETWEEN_ENCDEC:
        encode_net = mx.sym.FullyConnected(
            data=encode_net,
            num_hidden=cfg.MODEL.DECONVBASELINE.FC_BETWEEN_ENCDEC)

        if cfg.MODEL.DECONVBASELINE.USE_3D:
            encode_net = mx.sym.Reshape(
                data=encode_net, shape=(cfg.MODEL.TRAIN.BATCH_SIZE, -1, 1, 4, 4))
        else:
            encode_net = mx.sym.Reshape(
                data=encode_net, shape=(cfg.MODEL.TRAIN.BATCH_SIZE, -1, 4, 4))

    gen_net = video_net_symbol(
        encode_net,
        no_bias=no_bias,
        momentum=momentum,
        fix_gamma=fix_gamma,
        eps=eps)

    if cfg.DATASET == "MOVINGMNIST":
        OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
        IMG_SIZE = cfg.MOVINGMNIST.IMG_SIZE
    elif cfg.DATASET == "HKO":
        OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
        IMG_SIZE = cfg.HKO.ITERATOR.WIDTH

    # No operation if cfg.MODEL.DECONVBASELINE.USE_3D is True
    gen_net = mx.sym.reshape(
        gen_net,
        shape=(cfg.MODEL.TRAIN.BATCH_SIZE, 1, OUT_LEN, IMG_SIZE, IMG_SIZE),
        name="pred")

    return mx.sym.Group([
        gen_net,
        mx.sym.BlockGrad(
            mx.sym.clip(gen_net, a_min=0, a_max=1), name="forecast_target")
    ])


def discriminator_symbol(context,
                         pred,
                         no_bias=False,
                         momentum=0.9,
                         fix_gamma=False,
                         eps=1e-5 + 1e-12):
    # context: (batch_size, 1, input_length, 64, 64)
    # pred: (batch_size, 1, output_length, 64, 64)
    if cfg.DATASET == "MOVINGMNIST":
        OUT_LEN = cfg.MOVINGMNIST.OUT_LEN
    elif cfg.DATASET == "HKO":
        OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
        mask = mx.sym.Variable('mask')
        pred = pred * mask

    if cfg.MODEL.DECONVBASELINE.ENCODER in ["shared", "separate"]:
        postfix = "" if cfg.MODEL.DECONVBASELINE.ENCODER == "shared" else "_gan"

        context_encoding = encode_net_symbol(
            data=context,
            data_type="context",
            no_bias=no_bias,
            momentum=momentum,
            fix_gamma=fix_gamma,
            eps=eps,
            postfix=postfix)

        pred_encoding = encode_net_symbol(
            data=pred,
            data_type="pred",
            no_bias=no_bias,
            momentum=momentum,
            fix_gamma=fix_gamma,
            eps=eps)

        # context_encoding: (batch_size, 4 * num_filter, 1, 4, 4)
        # pred_encoding: (batch_size, 4 * num_filter, 1, 4, 4)

        if cfg.MODEL.DECONVBASELINE.USE_3D:
            context_pred = mx.sym.concat(
                context_encoding, pred_encoding, dim=2)
        else:
            context_pred = mx.sym.concat(
                context_encoding, pred_encoding, dim=1)

        # Compatibility code
        if cfg.MODEL.DECONVBASELINE.COMPAT.CONV_INSTEADOF_FC_IN_ENCODER:
            # Introduce extra layer to merge context and pred representations
            d5 = conv2d_3d_bn_act(
                use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS,
                use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
                use_bn=cfg.MODEL.DECONVBASELINE.BN,
                data=context_pred,
                name='discriminator_5',
                act_type=cfg.MODEL.CNN_ACT_TYPE,
                kernel=(1, 1, 1),
                stride=(1, 1, 1),
                pad=(0, 0, 0),
                num_filter=cfg.MODEL.DECONVBASELINE.BASE_NUM_FILTER,
                no_bias=no_bias,
                height=4,
                width=4,
                fix_gamma=fix_gamma,
                eps=eps,
                momentum=momentum)

            d6 = conv2d_3d(
                use_3d=cfg.MODEL.DECONVBASELINE.USE_3D,
                data=d5,
                name='discriminator_6',
                kernel=(1, 4, 4),
                stride=(1, 1, 1),
                pad=(0, 0, 0),
                num_filter=1,
                no_bias=no_bias)
            return mx.sym.Flatten(d6)
        else:
            # flattened_encoding: (batch_size, num_filter * 4^3)
            flattened_encoding = mx.sym.Flatten(data=context_pred)

    elif cfg.MODEL.DECONVBASELINE.ENCODER == "concat":
        context_pred = mx.sym.concat(context, pred, dim=2)

        encoding = encode_net_symbol(
            data=context_pred,
            data_type="contextpred",
            no_bias=no_bias,
            momentum=momentum,
            fix_gamma=fix_gamma,
            eps=eps)
        flattened_encoding = mx.sym.Flatten(data=encoding)

    else:
        raise NotImplementedError

    fc1 = fc_layer_act(
        data=flattened_encoding,
        num_hidden=256,
        name="discriminator_fc_1",
        act_type=cfg.MODEL.CNN_ACT_TYPE)
    return fc_layer(data=fc1, num_hidden=1, name="discriminator_fc_2")


### Helpers
def batchnorm_5d(data, height, width, name, fix_gamma, eps, momentum):
    data = mx.symbol.reshape(data, shape=(0, 0, -1, width))

    data = mx.sym.BatchNorm(
        data,
        name=name,
        fix_gamma=fix_gamma,
        eps=eps,
        momentum=momentum,
        use_global_stats=cfg.MODEL.DECONVBASELINE.BN_GLOBAL_STATS)

    return mx.symbol.reshape(data, shape=(0, 0, -1, height, width))


def conv2d_3d(data,
              num_filter,
              kernel=(1, 1, 1),
              stride=(1, 1, 1),
              pad=(0, 0, 0),
              dilate=(1, 1, 1),
              no_bias=False,
              name=None,
              use_3d=True,
              **kwargs):
    """If use_3d == False use a 2D convolution with the same number of parameters."""
    if use_3d:
        return conv3d(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            dilate=dilate,
            no_bias=no_bias,
            name=name,
            **kwargs)
    else:
        return conv2d(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            dilate=dilate[1:],
            no_bias=no_bias,
            name=name,
            **kwargs)


def conv2d_3d_bn_act(data,
                     num_filter,
                     height,
                     width,
                     kernel=(1, 1, 1),
                     stride=(1, 1, 1),
                     pad=(0, 0, 0),
                     dilate=(1, 1, 1),
                     no_bias=False,
                     act_type="relu",
                     momentum=0.9,
                     eps=1e-5 + 1e-12,
                     fix_gamma=True,
                     name=None,
                     use_3d=True,
                     use_bn=True,
                     use_global_stats=False,
                     **kwargs):
    """If use_3d == False use a 2D convolution with the same number of parameters."""
    if not use_bn:
        return conv2d_3d_act(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            dilate=dilate,
            no_bias=no_bias,
            act_type=act_type,
            name=name,
            use_3d=use_3d)

    if use_3d:
        return conv3d_bn_act(
            data=data,
            num_filter=num_filter,
            height=height,
            width=width,
            kernel=kernel,
            stride=stride,
            pad=pad,
            dilate=dilate,
            no_bias=no_bias,
            act_type=act_type,
            momentum=momentum,
            eps=eps,
            fix_gamma=fix_gamma,
            name=name,
            use_global_stats=use_global_stats,
            **kwargs)
    else:
        return conv2d_bn_act(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            dilate=dilate[1:],
            no_bias=no_bias,
            act_type=act_type,
            momentum=momentum,
            eps=eps,
            fix_gamma=fix_gamma,
            name=name,
            use_global_stats=use_global_stats,
            **kwargs)


def conv2d_3d_act(data,
                  num_filter,
                  kernel=(1, 1, 1),
                  stride=(1, 1, 1),
                  pad=(0, 0, 0),
                  dilate=(1, 1, 1),
                  no_bias=False,
                  act_type="relu",
                  name=None,
                  use_3d=True,
                  **kwargs):
    """If use_3d == False use a 2D convolution with the same number of parameters."""
    if use_3d:
        return conv3d_act(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            dilate=dilate,
            no_bias=no_bias,
            act_type=act_type,
            name=name,
            **kwargs)
    else:
        return conv2d_act(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            dilate=dilate[1:],
            no_bias=no_bias,
            act_type=act_type,
            name=name,
            **kwargs)


def deconv2d_3d(data,
                num_filter,
                kernel=(1, 1, 1),
                stride=(1, 1, 1),
                pad=(0, 0, 0),
                adj=(0, 0, 0),
                no_bias=True,
                target_shape=None,
                name=None,
                use_3d=True,
                **kwargs):
    """If use_3d == False use a 2D deconvolution with the same number of parameters."""
    if use_3d:
        return deconv3d_act(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            adj=adj,
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            name=name,
            **kwargs)
    else:
        return deconv2d_act(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            adj=adj[1:],
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            name=name,
            **kwargs)


def deconv2d_3d_bn_act(data,
                       num_filter,
                       height,
                       width,
                       kernel=(1, 1, 1),
                       stride=(1, 1, 1),
                       pad=(0, 0, 0),
                       adj=(0, 0, 0),
                       no_bias=True,
                       target_shape=None,
                       act_type="relu",
                       momentum=0.9,
                       eps=1e-5 + 1e-12,
                       fix_gamma=True,
                       name=None,
                       use_3d=True,
                       use_bn=True,
                       use_global_stats=False,
                       **kwargs):
    """If use_3d == False use a 2D deconvolution with the same number of parameters."""
    if not use_bn:
        return deconv2d_3d_act(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            adj=adj,
            no_bias=no_bias,
            act_type=act_type,
            name=name,
            use_3d=use_3d, )

    if use_3d:
        return deconv3d_bn_act(
            data=data,
            num_filter=num_filter,
            height=height,
            width=width,
            kernel=kernel,
            stride=stride,
            pad=pad,
            adj=adj,
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            momentum=momentum,
            eps=eps,
            fix_gamma=fix_gamma,
            name=name,
            use_global_stats=use_global_stats,
            **kwargs)
    else:
        return deconv2d_bn_act(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            adj=adj[1:],
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            momentum=momentum,
            eps=eps,
            fix_gamma=fix_gamma,
            name=name,
            use_global_stats=use_global_stats,
            **kwargs)


def deconv2d_3d_act(data,
                    num_filter,
                    kernel=(1, 1, 1),
                    stride=(1, 1, 1),
                    pad=(0, 0, 0),
                    adj=(0, 0, 0),
                    no_bias=True,
                    target_shape=None,
                    act_type="relu",
                    name=None,
                    use_3d=True,
                    **kwargs):
    """If use_3d == False use a 2D deconvolution with the same number of parameters."""
    if use_3d:
        return deconv3d_act(
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            adj=adj,
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            name=name,
            **kwargs)
    else:
        return deconv2d_act(
            data=data,
            num_filter=num_filter * kernel[0],
            kernel=kernel[1:],
            stride=stride[1:],
            pad=pad[1:],
            adj=adj[1:],
            no_bias=no_bias,
            target_shape=target_shape,
            act_type=act_type,
            name=name,
            **kwargs)
