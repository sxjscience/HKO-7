try:
    import cPickle as pickle
except:
    import pickle
import ast
import re
import inspect
import os
import logging
import numpy as np

def cross_entropy_npy(a, b):
    return a * np.log(b + 1E-9) + (1 - a) * np.log(1 - b + 1E-9)


def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr


def logging_config(folder=None, name=None,
                   level=logging.INFO,
                   console_level=logging.DEBUG):
    """

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level

    Returns
    -------

    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to %s" %logpath)
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    # Initialze the console logging
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)
    return folder


def load_params(prefix, epoch):
    """

    Parameters
    ----------
    prefix : str
    epoch : int

    Returns
    -------
    arg_params : dict
    aux_params : dict
    """
    import mxnet.ndarray as nd
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def parse_ctx(ctx_args):
    import mxnet as mx
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    ctx = [mx.Context(*ele) for ele in ctx]
    return ctx