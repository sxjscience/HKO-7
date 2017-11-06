# Extension of MXNet Module
import logging
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from collections import OrderedDict
from mxnet.module import Module


def nd_global_norm(t_list):
    """Computes the global norm of multiple tensors.

    Given a tuple or list of tensors t_list, this operation returns the global norm of the elements
     in all tensors in t_list. The global norm is computed as:

    ``global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))``

    Any entries in t_list that are of type None are ignored.

    Parameters
    ----------
    t_list: list or tuple
        The NDArray list

    Returns
    -------
    ret: NDArray
        The global norm. The shape of the NDArray will be (1,)

    Examples
    --------
    >>> x = mx.nd.ones((2, 3))
    >>> y = mx.nd.ones((5, 6))
    >>> z = mx.nd.ones((4, 2, 3))
    >>> print(nd_global_norm([x, y, z]).asscalar())
    7.74597
    >>> xnone = None
    >>> ret = nd_global_norm([x, y, z, xnone])
    >>> print(ret.asscalar())
    7.74597
    """
    ret = None
    for arr in t_list:
        if arr is not None:
            if ret is None:
                ret = nd.square(nd.norm(arr))
            else:
                ret += nd.square(nd.norm(arr))
    ret = nd.sqrt(ret)
    return ret


class MyModule(Module):
    """Some enhancement to the mx.mod.Module

    """

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=mx.context.gpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None, name=None):
        self._name = name
        super(MyModule, self).__init__(symbol=symbol,
                                       data_names=data_names,
                                       label_names=label_names,
                                       logger=logger,
                                       context=context,
                                       work_load_list=work_load_list,
                                       fixed_param_names=fixed_param_names,
                                       state_names=state_names)
        self._tmp_grads = None

    def clip_by_global_norm(self, max_norm=1.0):
        """Clips gradient norm.
        The norm is computed over all gradients together, as if they were
         concatenated into a single vector. Gradients are modified in-place.
        The method is first used in
         `[ICML2013] On the difficulty of training recurrent neural networks`
        Parameters
        ----------
        max_norm : float or int
            The maximum clipping threshold of the gradient norm.
        Returns
        -------
        norm_val : float
            The computed norm of the gradients.
        Examples
        --------
        An example of using clip_grad_norm to clip the gradient before updating the parameters::
            >>> #Get the gradient via back-propagation
            >>> net.forward_backward(data_batch=data_batch)
            >>> norm_val = net.clip_by_global_norm(max_norm=1.0)
            >>> net.update()
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        norm_val = self.global_grad_norm()
        if norm_val > max_norm:
            ratio = max_norm / float(norm_val)
            for grads in self._exec_group.grad_arrays:
                for grad in grads:
                    grad *= ratio
        return norm_val

    def global_grad_norm(self):
        """Calculate global gradient norm.
        The L2 norm is computed over all gradients together, as if they were
         concatenated into a single vector.
        Could be used to debug the optimization process.
         See http://videolectures.net/deeplearning2015_goodfellow_network_optimization/
        Returns
        -------
        norm_val : float
            The computed norm of the gradients.
        Examples
        --------
        An example of using global_norm to calculate the gradient norm after back-propgation::
            >>> #Get the gradient via back-propagation
            >>> net.forward_backward(data_batch=data_batch)
            >>> norm_val = net.global_grad_norm()
            >>> print(norm_val)
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        # The code in the following will cause the estimated norm to be different for multiple gpus
        norm_val = 0.0
        for exe in self._exec_group.execs:
            norm_val += nd_global_norm(exe.grad_arrays).asscalar()
        norm_val /= float(len(self._exec_group.execs))
        norm_val *= self._optimizer.rescale_grad
        return norm_val

    def debug_norm_all(self, debug_gnorm=True):
        if debug_gnorm:
            for k, v, grad_v in zip(self._param_names, self._exec_group.param_arrays,
                                    self._exec_group.grad_arrays):
                logging.debug("%s: v-norm: %g, g-norm: %g"
                              %(k,
                                nd.norm(v[0]).asnumpy()[0],
                                nd.norm(grad_v[0]).asnumpy()[0]))
        else:
            for k, v in zip(self._param_names, self._exec_group.param_arrays):
                logging.debug("%s: v-norm: %g"
                              %(k,
                                nd.norm(v[0]).asnumpy()[0]))

    def summary(self, level=2):
        """Summarize the network parameters.

        Parameters
        ----------
        level : int, optional
            Level of the summarization logs to print.
            The log becomes more verbose with higher summary level.
            - Level = 0
                Print the total param number + aux param number
            - Level = 1
                Print the shape of all parameters + The total number of paremter numbers
            - Level = 2
                Print the shape of the data/state and other available information in Level 1
        """
        self.logger.info("Summary of %s" %self._name)
        assert self.binded and self.params_initialized
        assert 0 <= level <= 2, \
            "Level must be between 0 and 2, level=%d is not supported" % level

        def _log_var(key, value, typ="param"):
            if typ == "param":
                if k in self._fixed_param_names:
                    self.logger.info("   %s: %s, %d, req = %s, fixed"
                                     % (key,
                                        str(value.shape),
                                        np.prod(value.shape),
                                        self._exec_group.grad_req[k]))
                else:
                    self.logger.info("   %s: %s, %d, req = %s"
                                     % (key,
                                        str(value.shape),
                                        np.prod(value.shape),
                                        self._exec_group.grad_req[k]))
            elif typ == "data" or typ == "aux":
                self.logger.info("   %s: %s, %d"
                                 % (key,
                                    str(value.shape),
                                    np.prod(value.shape)))

        total_param_num = 0
        total_fixed_param_num = 0
        total_aux_param_num = 0
        if level >= 2:
            if len(self.data_names) == 0:
                self.logger.info("Data: None")
            else:
                self.logger.info("Data:")
                for k, v in zip(self.data_names, self.data_shapes):
                    _log_var(k, v, typ="data")
            if len(self._state_names) == 0:
                self.logger.info("State: None")
            else:
                self.logger.info("State:")
                for k in self._state_names:
                    v = self._exec_group.execs[0].arg_dict[k]
                    _log_var(k, v, typ="data")
        if level >= 1:
            if len(self._param_names) == 0:
                self.logger.info("Param: None")
            else:
                self.logger.info("Params:")
                for k in self._param_names:
                    v = self._arg_params[k]
                    _log_var(k, v)
                    if k in self._fixed_param_names:
                        total_fixed_param_num += np.prod(v.shape)
                    else:
                        total_param_num += np.prod(v.shape)
            if len(self._aux_names) == 0:
                self.logger.info("Aux States: None")
            else:
                self.logger.info("Aux States: ")
                for k in self._aux_names:
                    v = self._aux_params[k]
                    _log_var(k, v, typ="aux")
                    total_aux_param_num += np.prod(v.shape)
        else:
            for k in self._param_names:
                v = self._arg_params[k]
                total_param_num += np.prod(v.shape)
            for k in self._aux_names:
                v = self._aux_params[k]
                total_aux_param_num += np.prod(v.shape)
        self.logger.info("Total Param Num (exclude fixed ones): " + str(total_param_num))
        self.logger.info("Total Fixed Param Num: " + str(total_fixed_param_num))
        self.logger.info("Total Aux Param Num: " + str(total_aux_param_num))

    def get_output_dict(self):
        outputs = self.get_outputs()
        return OrderedDict([(k, v) for k, v in zip(self._output_names, outputs)])

    def clear_grad(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        # clear the gradient
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad[:] = 0

    def save_tmp_grad(self):
        if self._tmp_grads is None:
            self._tmp_grads = []
            for grads in self._exec_group.grad_arrays:
                vec = []
                for grad in grads:
                    vec.append(grad.copyto(grad.context))
                self._tmp_grads.append(vec)
        else:
            for i, grads in enumerate(self._exec_group.grad_arrays):
                for j, grad in enumerate(grads):
                    self._tmp_grads[i][j][:] = grad

    def acc_grad_with_tmp(self):
        assert self._tmp_grads is not None
        for i, grads in enumerate(self._exec_group.grad_arrays):
            for j, grad in enumerate(grads):
                grad += self._tmp_grads[i][j]


    def load_params_allow_missing(self, fname):
        """Loads model parameters from file.

        Parameters
        ----------
        fname : str
            Path to input param file.

        Examples
        --------
        >>> # An example of loading module parameters.
        >>> mod.load_params('myfile')
        """
        logging.info("Load Param From %s" %fname)
        save_dict = mx.nd.load(fname)
        arg_params = {}
        aux_params = {}
        for k, value in save_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                if name in self._param_names:
                    logging.info("set %s" %name)
                    arg_params[name] = value
            elif arg_type == 'aux':
                if name in self._aux_names:
                    logging.info("set %s" % name)
                    aux_params[name] = value
            else:
                raise ValueError("Invalid param file " + fname)
        self.set_params(arg_params, aux_params, allow_missing=True)
