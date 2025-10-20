ffmpeg -framerate 25 -start_number 3000 -i rho_%3d.png -c:v libx264 -r 25 -pix_fmt yuv420p out.mp4 
