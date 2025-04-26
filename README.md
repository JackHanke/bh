# Black Hole Emulation

![](./assets/bh.jpg)

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
