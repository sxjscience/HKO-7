from nowcasting.config import cfg
import os
import numpy as np
import matplotlib.pyplot as plt
base_dir = "flows"
prefixs = ["ebrnn1_0", "ebrnn2_0", "ebrnn3_0", "fbrnn1_0", "fbrnn2_0", "fbrnn3_0"]
forecaster_block_prefix = []
flow_num = 13
lengths = [10, 10, 10, 10, 10, 10]
heights = [64, 32, 16, 64, 32, 16]
widths = [64, 32, 16, 64, 32, 16]
flows = []
for prefix, height, width, length in zip(prefixs, heights, widths, lengths):
    flow_maps = np.empty((length, flow_num, 2, height, width), dtype=np.float32)
    X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    for frame_id in range(length):
        path = os.path.join(base_dir, "{}__t{}_flow.npy".format(prefix, frame_id))
        flow = np.load(path)
        flow_maps[frame_id, :, :, :, :] = flow.reshape((flow_num, 2, height, width))
        for i in range(flow_num):
            plt.title('Flow')
            Q = plt.quiver(X, Y[::-1, :],
                           flow_maps[frame_id, i, 0, :, :],
                           - flow_maps[frame_id, i, 1, :, :],
                           units='width')
            save_dir = os.path.join(base_dir, "%s_link%d" % (prefix, i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "frame%d.png" % (frame_id,))
            print("Generating the flow for %s, frame_id=%d, flow_id=%d, saving to %s"
                  %(prefix, frame_id, i, save_path))
            plt.savefig(save_path,
                        bbox_inches="tight")
            plt.close()
    flows.append(flow_maps)
print(flows[3].std(axis=2))
np.savez('trajgru_flows.npz', **dict([[prefix, flow]
                                      for prefix, flow in zip(prefixs, flows)]))