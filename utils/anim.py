import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from harmpi.harm_script import *
from utils.utils import read_dump_util
import torch

# structure dumps data for animation
def make_ground_truth_frames():
    ground_truths = []
    for dump in dumps:
        _, dump_dict = read_dump_util(dump=dump)
        ground_truth_array = dump_dict['rho'][:,:,0].transpose()
        ground_truth_array = np.log10(ground_truth_array)
        ground_truths.append(ground_truth_array)

    return ground_truths

# run FFNN from starting dump for length of dumps
# returns formatted predictions 
def make_prediciton_frames(
    net: torch.nn, 
    first_frame: torch.Tensor, 
    num_frames: int,
    make_latents: bool = False,
    flatten: bool = False
    ):
    # helper for adding formatted frame to predictions array

    # format prediction tensor to array for animation
    def _postprocess_prediction(frame: torch.Tensor):
        # NOTE ugliest thing ever please fix
        frame_array = frame.clone().detach()[0][0].unsqueeze(-1).detach().numpy()[:,:,0].transpose()
        # TODO second 0 above means rho! find out how to make this not stupid
        return frame_array
        
        # NOTE transform 
        # frame_array = np.log10(frame_array)
        
        predictions.append(frame_array)

    # format latent tensor to array for animation
    def _postprocess_latent(frame: torch.Tensor):
        frame_array = frame.clone().detach().detach().numpy()
        return frame_array

    net.eval()
    # get first datapoint
    frame = first_frame
    batch_len = 1
    predictions, latents = [], []
    # create prediction list
    for frame_num in range(num_frames):
        # copy and postprocess frame, add to predictions
        frame_array = _postprocess_prediction(frame=frame)
        predictions.append(frame_array)

        # reshape for network
        if flatten: frame = torch.reshape(frame, (batch_len, 8*128*128))

        # forward pass
        if not make_latents:
            frame = net.forward(frame)
        elif make_latents:
            latent = net.encode(frame)
            latent_array = _postprocess_latent(frame=latent)
            latents.append(latent_array)
            frame = net.decode(latent)
            
        # reshape for animation
        if flatten: frame = torch.reshape(frame, (batch_len, 8,128,128))

    return predictions, latents


# reshape latent for image rendering
def latent_to_im(latent: np.array):
    # get dimensions for image
    batch_size, flattened_dim = latent.shape
    dims = int(np.sqrt(flattened_dim // 3))
    # reshape to im dimensions
    reshaped_latent = np.reshape(latent, newshape=(dims, dims, 3))
    # turn 0,1 float im to 255 int im
    # reshaped_latent = np.round(255 * reshaped_latent).astype(int)
    return reshaped_latent

# animates list of frames of ground truths and list of predictions side by side
# NOTE data should be a () numpy array
def animate_comparison(
        ground_truths: list[np.array], 
        predictions: list[np.array],
        latents: list[np.array] = None, 
        save_path: str = './movies/movie.gif', 
        cb = True,
    ):
    if latents is None: anim_latents = False
    elif latents is not None: anim_latents = True 

    # set up figure axes    
    if not anim_latents:
        fig, axs = plt.subplots(1, 2)
    elif anim_latents:
        fig, axs = plt.subplots(2, 2, height_ratios=[3, 1])

    if not anim_latents:
        axs[0].tick_params(labelsize=10,bottom=False,left=False,labelbottom=False,labelleft=False)
        axs[0].set_title(f'Ground Truth')
        axs[1].tick_params(labelsize=10,bottom=False,left=False,labelbottom=False,labelleft=False)
        axs[1].set_title(f'Prediction')
    if anim_latents: 
        axs[0,0].set_aspect('equal')
        axs[0,1].set_aspect('equal')
        axs[0,0].tick_params(labelsize=10,bottom=False,left=False,labelbottom=False,labelleft=False)
        axs[0,0].set_title(f'Ground Truth')
        axs[0,1].tick_params(labelsize=10,bottom=False,left=False,labelbottom=False,labelleft=False)
        axs[0,1].set_title(f'Prediction')
        axs[1,0].tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
        axs[1,0].set_title(f'Latent Representation')
        axs[1,1].axis('off')

    # Initialize the contour, colorbar objects
    contour_truth, contour_pred = None, None  
    colorbar_truth, colorbar_pred = None, None 

    # update function for animation
    def update(frame):
        # get vars from outside update function
        nonlocal contour_truth, contour_pred, colorbar_truth, colorbar_pred
        # Clear the previous contour if it exists
        if contour_truth:
            for collection in contour_truth.collections: collection.remove()
            if colorbar_truth: colorbar_truth.remove()
        if contour_pred:
            for collection in contour_pred.collections: collection.remove()
            if colorbar_pred: colorbar_pred.remove()

        # get data from lists
        ground_truth = ground_truths[frame]
        prediction = predictions[frame]
        if anim_latents:
            latent = latents[frame]
            latent_im = latent_to_im(latent = latent)

        # compute contour 
        nc = 100
        if not anim_latents:
            contour_truth = axs[0].contourf(ground_truth, nc)
            contour_pred = axs[1].contourf(prediction, nc)
        if anim_latents:
            contour_truth = axs[0,0].contourf(ground_truth, nc)
            contour_pred = axs[0,1].contourf(prediction, nc)
            latent_im_obj = axs[1,0].imshow(latent_im)
        
        # if include colorbar, add colorbars
        if cb:
            if not anim_latents:
                colorbar_truth = plt.colorbar(contour_truth, cax=axs[0], orientation='horizontal')
                colorbar_pred = plt.colorbar(contour_pred, ax=axs[1], orientation='horizontal')
            if anim_latents:
                truth_divider = make_axes_locatable(axs[0,0])
                truth_cax = truth_divider.append_axes("right", size="5%", pad=0.05)

                pred_divider = make_axes_locatable(axs[0,1])
                pred_cax = pred_divider.append_axes("right", size="5%", pad=0.05)

                colorbar_truth = plt.colorbar(contour_truth, cax=truth_cax)
                colorbar_pred = plt.colorbar(contour_pred, cax=pred_cax)
            colorbar_truth.ax.tick_params(labelsize=5)
            colorbar_pred.ax.tick_params(labelsize=5)

        # Set the title based on the frame number
        fig.suptitle(f't = {frame}', fontsize=10)

         # Return the artists that have changed
        if not anim_latents:
            return contour_truth.collections + contour_pred.collections + [axs[0].title]
        if anim_latents:
            return contour_truth.collections + contour_pred.collections + [axs[0,0].title] + [latent_im_obj]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(ground_truths), 
        interval=200,
        blit=True
    )

    directory_chunks = save_path.split('/')
    directory = directory_chunks[0] + "/" + directory_chunks[1]

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Creating it...")
        os.makedirs(directory)
    else:
        print(f"Directory {directory} already exists.")
        
    anim.save(save_path)
    print(f'Saved animated dumps at {save_path}')    
