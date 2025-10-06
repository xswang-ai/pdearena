# load one sample and plot the value

import h5py
import matplotlib.pyplot as plt


def load_sample(sample_path):
    var_list = ['u', 'vx', 'vy']
    with h5py.File(sample_path, 'r') as f:
        sample = {var: f[var][0] for var in var_list}
        print("sample u .shape", sample['u'].shape)
        print("sample vx.shape", sample['vx'].shape)
        print("sample vy.shape", sample['vy'].shape)
        print("sample.keys()", sample.keys())
    return sample

# load the sample
load_path = '/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised'
sample = load_sample(f'{load_path}/NavierStokes2D_train_198010_0.10000_5000_sample_000000.h5')


def generate_gt_gif(pred_data, sample_id=0, channel_id=0, model_name='FNO', log_path=None):
    print("saved_data shape", pred_data['pred'].shape, "pred_data.keys()", pred_data.keys())
    # keys are (input, output, pred), shape of  (B, H, W, T_in/out, C)
    cmap = 'RdBu_r'
    target = pred_data['output'][sample_id, ... , channel_id].detach().cpu().numpy() # (H, W, T_out)
    pred = pred_data['pred'][sample_id, ... , channel_id].detach().cpu().numpy() # (H, W, T_out)
    print("target shape", target.shape, "pred shape", pred.shape)
    vmax = np.max(np.abs(target))
    vmin = -vmax if np.min(target) <0 else np.min(target)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    img0 = ax[0].imshow(target[..., 0], vmin=vmin, vmax=vmax, cmap=cmap)
    img1 = ax[1].imshow(pred[..., 0], vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].axis('off')
    ax[1].axis('off')
    title1 =ax[0].set_title('Target T+1')
    title2 = ax[1].set_title('Pred T+1')

    def update(frame_idx):
        if frame_idx < target.shape[2]:
            img0.set_data(target[..., frame_idx])
            title1.set_text(f'Target T+{frame_idx+1}')
        img1.set_data(pred[..., frame_idx])
        title2.set_text(f'Pred T+{frame_idx+1}')
        return img0, img1, title1, title2

    anim = FuncAnimation(fig, update, frames=pred.shape[2], interval=200, blit=False)

    gif_path = f'{log_path}/{model_name}_target.gif'
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
    except Exception as e:
        print(f'Failed to save GIF due to: {e}')

    # Try MP4 as well if ffmpeg is available
    try:
        anim.save(f'{log_path}/{model_name}_target.mp4', writer='ffmpeg', fps=5)
    except Exception as e:
        print(f'FFmpeg not available or failed to save MP4: {e}')

    plt.close(fig)


