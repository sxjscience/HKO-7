import numpy as np
from nowcasting.utils import logging_config
from nowcasting.hko_benchmark import HKOBenchmarkEnv
from nowcasting.config import cfg
import os

def run(pd_path=cfg.HKO_PD.RAINY_TEST, mode="fixed"):
    base_dir = os.path.join('hko7_benchmark', 'last_frame')
    logging_config(base_dir)
    batch_size = 1
    env = HKOBenchmarkEnv(pd_path=pd_path,
                          save_dir=base_dir,
                          mode=mode)
    while not env.done:
        in_frame_dat, in_datetime_clips, out_datetime_clips,\
        begin_new_episode, need_upload_prediction =\
            env.get_observation(batch_size=batch_size)
        if need_upload_prediction:
            prediction = np.zeros(shape=(cfg.HKO.BENCHMARK.OUT_LEN,) + in_frame_dat.shape[1:],
                                  dtype=in_frame_dat.dtype)
            prediction[:] = in_frame_dat[-1, ...]
            env.upload_prediction(prediction=prediction)
            env.print_stat_readable()
    env.save_eval()

run(cfg.HKO_PD.RAINY_TEST, "fixed")
run(cfg.HKO_PD.RAINY_VALID, "fixed")

