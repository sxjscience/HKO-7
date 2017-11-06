import mxnet as mx
import numpy as np
from nowcasting.operators.common import constant


def CDNA(data, kernels, mask, batch_size, num_filter, kernel_size):
    """We assume that the kernels and masks are the output of an identity activation

    Parameters
    ----------
    data : mx.sym.symbol
        Shape: (batch_size, C, H, W)
    kernels : mx.sym.symbol
        Shape: (batch_size, M, K, K)
    mask : mx.sym.symbol
        Shape: (batch_size, M, H, W)
    batch_size : int
    num_filter : int
        M
    kernel_size : int
        K
    Returns
    -------
    ret : mx.sym.symbol
        Shape: (batch_size, C, H, W)
    """
    assert kernel_size % 2 == 1, "Only support odd kernel size"
    # Use softmax activation for the kernel and the mask
    kernels = mx.sym.SoftmaxActivation(mx.sym.Reshape(kernels,
                                                      shape=(-1, kernel_size * kernel_size)))
    kernels = mx.sym.Reshape(kernels, shape=(-1, num_filter, kernel_size, kernel_size))
    mask = mx.sym.SoftmaxActivation(mask, mode="channel")

    data_sliced = mx.sym.SliceChannel(mx.sym.expand_dims(data, axis=2), axis=0,
                                      num_outputs=batch_size, squeeze_axis=True) # Each Shape: (C, 1, H, W)
    kernels_sliced = mx.sym.SliceChannel(mx.sym.expand_dims(kernels, axis=2),
                                         axis=0, num_outputs=batch_size,
                                         squeeze_axis=True) # Each Shape: (M, 1, K, K)
    out = []
    for i in range(batch_size):
        ele = mx.sym.Convolution(data=data_sliced[i],
                                 num_filter=num_filter,
                                 kernel=(kernel_size, kernel_size),
                                 pad=(kernel_size/2, kernel_size/2),
                                 weight=kernels_sliced[i], no_bias=True) # Shape: (C, M, H, W)
        out.append(mx.sym.expand_dims(ele, axis=0))
    out = mx.sym.Concat(*out, num_args=batch_size, dim=0) # Shape: (batch_size, C, M, H, W)
    mask = mx.sym.Reshape(mask, reverse=True, shape=(batch_size, 1, num_filter, 0, 0))
    out = mx.sym.broadcast_mul(out, mask)
    out = mx.sym.sum(out, axis=2)
    return out

def STP(data, affine_transform_matrices, mask, num_filter, kernel_size):
    """Spatial Transformer Predictor

    Parameters
    ----------
    data : mx.sym.symbol
    affine_transform_matrices
    mask

    Returns
    -------

    """
    raise NotImplementedError()


def DFN(data, local_kernels, K, batch_size):
    """[NIPS2016] Dynamic Filter Network

    Parameters
    ----------
    data : mx.sym.symbol
        Shape: (batch_size, C, H, W)
    local_kernels : mx.sym.symbol
        Shape: (batch_size, K*K, H, W)
    K : int
        size of the local convolutional kernel
    batch_size : int
        size of the minibatch
    Returns
    -------

    """
    local_kernels = mx.sym.SoftmaxActivation(local_kernels, mode="channel")
    #filter_localexpand_npy = np.eye(K*K, K*K).reshape((K*K, 1, K, K)).astype(np.float32)
    #filter_localexpand = constant(filter_localexpand_npy, name="CDNA_kernels")
    filter_localexpand = mx.sym.one_hot(indices=mx.sym.arange(K * K), depth=K*K)
    filter_localexpand = mx.sym.reshape(mx.sym.transpose(filter_localexpand, axes=(1, 0)),
                                        shape=(K * K, 1, K, K))
    data_sliced = mx.sym.SliceChannel(data, num_outputs=batch_size, axis=0, squeeze_axis=True)
    vec = []
    for i in range(batch_size):
        ele = mx.sym.Convolution(data=mx.sym.expand_dims(data_sliced[i], axis=1),
                                 weight=filter_localexpand,
                                 num_filter=K*K,
                                 kernel=(K, K),
                                 pad=(K // 2, K // 2), no_bias=True)  # Shape (C, K*K, H, W)
        vec.append(mx.sym.expand_dims(ele, axis=0))
    input_localexpanded = mx.sym.Concat(*vec, num_args=len(vec), dim=0)   # Shape (batch_size, C, K*K, H, W)
    output = mx.sym.broadcast_mul(input_localexpanded, mx.sym.expand_dims(local_kernels, axis=1))
    output = mx.sym.sum(output, axis=2)
    return output



if __name__ == '__main__':
    data = mx.sym.Variable('data')
    local_kernels = mx.sym.Variable('local_kernels')
    K = 11
    C = 3
    H = 60
    W = 60
    batch_size = 32
    local_kernels_npy = np.random.normal(size=(batch_size, K*K, H, W))
    data_npy = np.random.normal(size=(batch_size, C, H, W))
    out = data
    for i in range(10):
        out = DFN(data=out, local_kernels=local_kernels, K=K, batch_size=batch_size)
    exe = out.simple_bind(ctx=mx.gpu(), data=(batch_size, C, H, W),
                          local_kernels=(batch_size, K*K, H, W))
    exe.forward(data=data_npy, local_kernels=local_kernels_npy)
    print(exe.outputs[0].asnumpy().shape)
