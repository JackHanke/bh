# Black Hole Emulation

![](./assets/bh.jpg)

## Developer Notes

- [Perlmutter Docs](https://docs.nersc.gov/getting-started/)
    - To login: `ssh user@saul.nersc.gov`, then password + 6 digit Google Authenticator code
- scratch directories have huge storage compared to home on scs

- [BitBucket](https://bitbucket.org/atchekho/harm2d/src/master/)

## TODOs
- Design models (train on (first_frame, next_frame) pairs) 
    - [x] FFNN
    - [x] CNN
    - [.] UNet
    - [.] [Flow Based?](https://en.wikipedia.org/wiki/Flow-based_generative_model)
- Do we AE data, then model the movie?
    - [.] for this project, an AE (trained on (first_frame, first_frame) pairs)
- Visualization
    - [.] Make viz code variable 
    - [.] Visualize Latent Space
    - [.] Visualize error over course of prediction movie from ground truth
- 2D to 3D convolutions
- Design Cost
    - [x] MSE 
    - [.] Location specific MSE
    - [.] Conservation Laws (Conservation of Angular Momentum, Energy, Mass, etc.)
- Fully understand what we are shooting for
    - [.] Handling time 
- Once we have supercomputer access...
    - Test 3d conv code


## Practicum Catchup Meeting
- *I want to hear/see a crisp clean statement of exactly what you ae going to be working on with your clients. This is a place where less is more.*
    - We seek to train a physics-informed neural-network-based approximation of black hole gas accretion, using the `harmpi` program to generate simulation data for training. A network that preforms well-enough to be used for large $t$ will constitute a breakthrough in numerical simulations of this physical system.
- *I want an articulation of what you see as the metrics for success.*
    - We are currently considering three cost functions, a global MSE, a location specific MSE, and physics informed regularization terms to either. No threshold for accuracy, as long as we have reasonable predictions that are more computationally efficient to produce than the full `harmpi` simulations.
- *I want a walk through of the approach(es) you are using to get there.*
    - We are considering three model architectures, an FFNN, a CNN, and a UNet, as we will be dealing with spacial to spacial mappings for the prediction networks. Our FFNN will be the baseline, as the other networks focus on preseving spacial structure. We also plan to encode the high dimensional simulations using an autoencoder to make a "latent simulation" to further reduce prediction cost and compounding prediction errors. Results on the raw simulation and the latent simulation will be compared. 
- *I want to know that state of the data requirements and the whether you have what you need.*
    - Comparable toy datasets have already been created on our local machines using the single-threaded `harmpi` simulation code. This is being used to debug modeling and write preparation scripts. These are designed with the full, 3D simulations in mind, so that our work an easily trasfer to the larger, higher-resolution simulations stored on the supercomputers. The group was assigned to apply to 3 seperate supercomputers (OLCF, ALCF, and NERSC). The group has full access to NERSC, but does not know how to deploy jobs on the machine yet (focus of next meeting), and are in the process of applying for the other two. Additionally, we expect to receive some guidance on the supercomputer dataset, as we haven't seen this larger set yet. 
- *I want an outline of execution plan.*
    - Before we have full access to the SCs and know how to use them, we plan to develop as much as possible using the single-threaded simulation code and our local machines. We currently have demonstrated FFNN and CNN raw predictions, as well as visualized 
- *Expected challenges*
    - How does the SC dataset compare with the ones we work with locally?
    - `harmpi` codebase is dense, fetching relevant variables can be tricky (though recent work has improved this)
    - Ensuring the network follows physical principles and laws
    - Various formatting of code for ease of iteration
    - 