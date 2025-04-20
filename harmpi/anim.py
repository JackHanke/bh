
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from harm_script import *
from utils import read_dump_util

def animate_dumps(dumps: list[str], save_path: str):
    def generate_data(frame):
        # rd(dumps[frame])
        _, dump_dict = read_dump_util(dump=dumps[frame])
        return dump_dict

    fig, axs = plt.subplots(2, 1)
    contour = None  # Initialize the contour object
    colorbar = None # Initialize the colorbar object

    def update(frame):
        nonlocal contour, colorbar
        dump_dict = generate_data(frame)

        # Clear the previous contour if it exists
        if contour:
            for collection in contour.collections:
                collection.remove()
            if colorbar:
                colorbar.remove()

        var = dump_dict['rho']
        var = np.log10(var)
        # Create the new contour plot
        contour, colorbar = plc(
            var,
            xy=1, 
            xmax=100, 
            ymax=50, 
            cb=True, 
            ax=axs[0],
            isfilled=True, 
            nc=100
        )
        # TODO change for predictions
        # contour, colorbar = plc(
        #     var,
        #     xy=1, 
        #     xmax=100, 
        #     ymax=50, 
        #     cb=True, 
        #     ax=axs[1],
        #     isfilled=True, 
        #     nc=100
        # )

        # Set the title based on the frame number
        axs[0].set_title(f'BH Sim (t = {frame})')
        axs[0].set_ylabel(f'Ground Truth')


        axs[1].set_ylabel(f'Prediction')

        return contour.collections + [axs[0].title] # Return the artists that have changed

    # Create the animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(dumps), 
        interval=200,
        blit=True
    )

    anim.save(save_path)
    print(f'Saved animated dumps at {save_path}')    
