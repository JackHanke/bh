ffmpeg -framerate 25 -start_number 3000 -i pred_rho_%3d.png -vf "fps=10,scale=500:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -gifflags -loop 0 pred.gif
