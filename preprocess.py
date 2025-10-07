# load one sample and plot the value

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def load_sample(sample_path):
    var_list = ['u', 'vx', 'vy']
    with h5py.File(sample_path, 'r') as f:
        sample = {var: f[var][:] for var in var_list}
        print("sample u .shape", sample['u'].shape, sample['u'].dtype)
        print("sample vx.shape", sample['vx'].shape, sample['vx'].dtype)
        print("sample vy.shape", sample['vy'].shape, sample['vy'].dtype)
        print("sample.keys()", sample.keys())
    return sample

# load the sample
load_path = '/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised'
sample = load_sample(f'{load_path}/NavierStokes2D_train_198010_0.10000_5000_sample_000000.h5')

sample_id = 0
def generate_gt_gif(sample_data, log_path=None):
    # keys are (u, vx, vy), shape (T, H, W)
    cmap = 'RdBu_r'
    
    keys = ['u', 'vx', 'vy']
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    titles = {}
    imgs = {}
    for i, key in enumerate(keys):
        data_c = sample_data[key]
        print("data_c.shape", data_c.shape)
        vmax = np.max(np.abs(data_c))
        vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
        imgs[key] = ax[i].imshow(data_c[0], vmin=vmin, vmax=vmax, cmap=cmap)
        ax[i].axis('off')
        titles[key] =ax[i].set_title(key + ' T=0')

    def update(frame_idx):
        print("frame_idx", frame_idx)
        for i, key in enumerate(keys):
            data_c = sample_data[key]
            vmax = np.max(np.abs(data_c))
            vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
            imgs[key].set_data(data_c[frame_idx])
            imgs[key].set_clim(vmin=vmin, vmax=vmax)
            titles[key].set_text(key + ' T=' + str(frame_idx))
        # Don't return anything when blit=False
        return []

    anim = FuncAnimation(fig, update, frames=sample_data['u'].shape[0], interval=200, blit=False)

    gif_path = f'{log_path}/sample_{sample_id}.gif'
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
    except Exception as e:
        print(f'Failed to save GIF due to: {e}')

    plt.close(fig)


log_path = '/home/wan410/pdearena'
generate_gt_gif(sample, log_path)

