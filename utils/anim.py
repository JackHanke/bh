import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from harmpi.harm_script import *
from utils.utils import read_dump_util

# animates list of frames of ground truths and list of predictions side by side
# NOTE data should be a () numpy array
def animate_comparison(
        ground_truths: list[np.array], 
        predictions: list, 
        save_path: str, 
        cb = True,
    ):

    # set up figure axes    
    fig, axs = plt.subplots(1, 2)
    axs[0].tick_params(labelsize=10)
    axs[1].tick_params(labelsize=10)
    axs[0].set_title(f'Ground Truth')
    axs[1].set_title(f'Prediction')

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

        # compute contour 
        nc = 100
        contour_truth = axs[0].contourf(ground_truth, nc)
        contour_pred = axs[1].contourf(prediction, nc)
        
        # if include colorbar, add colorbars
        if cb:
            colorbar_truth = plt.colorbar(contour_truth, ax=axs[0], orientation='horizontal')
            colorbar_pred = plt.colorbar(contour_pred, ax=axs[1], orientation='horizontal')

            colorbar_truth.ax.tick_params(labelsize=5)
            colorbar_pred.ax.tick_params(labelsize=5)

        # Set the title based on the frame number
        fig.suptitle(f't = {frame}', fontsize=10)

         # Return the artists that have changed
        return contour_truth.collections + contour_pred.collections + [axs[0].title]

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


