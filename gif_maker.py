import os
import imageio
from natsort import natsorted

def delete_all_temp_outputs(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def create_gif(png_folder, img_prefix, gif_filename, time=-3.0):
    images = []

    # Assuming the PNG files are named in sequential order (e.g., frame1.png, frame2.png, ...)
    png_files = natsorted(os.listdir(png_folder))

    for png_file in png_files:
        if png_file.startswith(img_prefix) and png_file.endswith('.png'):
            file_path = os.path.join(png_folder, png_file)
            images.append(imageio.imread(file_path))

    if time < 0:
        time = float(len(images)) / 12.0
    # Save the images as a GIF
    imageio.mimsave(gif_filename+'.gif', images, duration=time, format='GIF')

if __name__ == '__main__':
    create_gif('outputs', 'heuristic_iter', 'heuristic_gif', time=-1.0)

