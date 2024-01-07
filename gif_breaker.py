from PIL import Image

def gif_to_png(gif_path, output_folder):
    # Open the GIF file
    gif = Image.open(gif_path)

    # Iterate through each frame in the GIF
    for frame_num in range(gif.n_frames):
        # Select the current frame
        gif.seek(frame_num)

        # Convert the frame to RGBA (if not already in RGBA mode)
        rgba_frame = gif.convert("RGBA")

        # Save the frame as a PNG file
        output_path = f"{output_folder}/frame_{frame_num:03d}.png"
        rgba_frame.save(output_path, format="PNG")

if __name__ == "__main__":
    # Specify the path to the GIF file
    gif_path = "path/to/your/input.gif"

    # Specify the output folder where PNG files will be saved
    output_folder = "path/to/your/output_folder"

    # Create the output folder if it doesn't exist
    import os
    os.makedirs(output_folder, exist_ok=True)

    # Call the function to convert the GIF to PNG frames
    gif_to_png(gif_path, output_folder)
