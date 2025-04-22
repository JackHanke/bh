import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from harm_script import *
from utils import read_dump_util

def animate_dumps(dumps: list[str], preds: list, save_path: str, r, h):
    def generate_data(frame):
        # rd(dumps[frame])
        _, dump_dict = read_dump_util(dump=dumps[frame])
        return dump_dict

    fig, axs = plt.subplots(1, 2)
    axs[0].tick_params(labelsize=10)
    axs[1].tick_params(labelsize=10)
    axs[0].set_title(f'Ground Truth')
    axs[1].set_title(f'Prediction')

    contour_truth, contour_pred = None, None  # Initialize the contour object
    colorbar_truth, colorbar_pred = None, None # Initialize the colorbar object

    def update(frame):
        nonlocal contour_truth, contour_pred, colorbar_truth, colorbar_pred
        dump_dict = generate_data(frame)
        pred = preds[frame]

        # Clear the previous contour if it exists
        if contour_truth:
            for collection in contour_truth.collections:
                collection.remove()
            if colorbar_truth:
                colorbar_truth.remove()
        if contour_pred:
            for collection in contour_pred.collections:
                collection.remove()
            if colorbar_pred:
                colorbar_pred.remove()

        # 
        var = dump_dict['rho']
        var = np.log10(var)

        # 
        pred = pred[0][0].unsqueeze(-1).detach().numpy()
        pred = np.log10(pred)

        # NOTE from harmpi plc code
        def plc_transform(myvar):
            symmx = 0
            k = 0
            ny = 128
            xcoord = r * np.sin(h)
            ycoord = r * np.cos(h)
            return xcoord, ycoord
        
        xcoord_truth, ycoord_truth = plc_transform(myvar=var)
        xcoord_pred, ycoord_pred = plc_transform(myvar=pred)

        plotted_data_truth = var
        plotted_data_pred = pred

        nc = 100
        contour_truth = axs[0].contourf(plotted_data_truth[:,:,0].transpose(), nc)
        contour_pred = axs[1].contourf(plotted_data_pred[:,:,0].transpose(), nc)
        
        colorbar_truth = plt.colorbar(contour_truth, ax=axs[0], orientation='horizontal')
        colorbar_pred = plt.colorbar(contour_pred, ax=axs[1], orientation='horizontal')

        colorbar_truth.ax.tick_params(labelsize=5)
        colorbar_pred.ax.tick_params(labelsize=5)


        # Set the title based on the frame number
        fig.suptitle(f't = {frame}', fontsize=10)
        # axs[0].set_ylabel(f'Ground Truth')
        # axs[1].set_ylabel(f'Prediction')

         # Return the artists that have changed
        return contour_truth.collections + contour_pred.collections + [axs[0].title]
        # return contour_truth.collections

    # Create the animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(dumps), 
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
