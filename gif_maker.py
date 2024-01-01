import os
import imageio
from natsort import natsorted

def create_gif(png_folder, img_prefix, gif_filename, duration=3.0):
    images = []

    # Assuming the PNG files are named in sequential order (e.g., frame1.png, frame2.png, ...)
    png_files = natsorted(os.listdir(png_folder))

    for png_file in png_files:
        if png_file.startswith(img_prefix) and png_file.endswith('.png'):
            file_path = os.path.join(png_folder, png_file)
            images.append(imageio.imread(file_path))

    if duration < 0:
        duration = len(images) / 24.0
    # Save the images as a GIF
    imageio.mimsave(gif_filename, images, duration=duration, format='GIF')

if __name__ == '__main__':
    create_gif('outputs', 'heuristic_iter', 'heuristic_gif.gif', duration=4.0)

