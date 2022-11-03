import numpy as np
import img_scale
import matplotlib.pyplot as plt
import math


# Parameters
sig_fract = 3.0
per_fract = 1.0e-4
max_iter = 50
min_val = 0.0
non_linear_fact = 0.1


# import the image.
file = np.load("../data/cube_part1.npy",mmap_mode="r")
labels = np.load("../data/labels_part1.npy",mmap_mode="r")
num = input("Enter the sample to check: ")

img_data = [file[int(num), :, :, i] for i in range(5)]
# To plot individual images.
columns = 3
rows = 2
ax = []
fnames = ["u","g","i","r","z"]
fig = plt.figure(figsize=(8, 8))
for i in range(columns*rows):
    if i == 5:
        # create the rgb synthetic image.
        rgb_array = np.empty((64,64,3), dtype=float)

        # Blue filter
        img_data_b = np.array(img_data[0], dtype=float)
        sky, num_iter = img_scale.sky_median_sig_clip(img_data_b, sig_fract, per_fract, max_iter)
        img_data_b = img_data_b - sky
        b = img_scale.asinh(img_data_b, scale_min = min_val, non_linear=non_linear_fact)

        # Green filter
        img_data_g = np.array(img_data[1], dtype=float)
        sky, num_iter = img_scale.sky_median_sig_clip(img_data_g, sig_fract, per_fract, max_iter)
        img_data_g = img_data_g - sky
        g = img_scale.asinh(img_data_g, scale_min = min_val, non_linear=non_linear_fact)

        # Red filter
        img_data_r = np.array(img_data[3], dtype=float)
        sky, num_iter = img_scale.sky_median_sig_clip(img_data_r, sig_fract, per_fract, max_iter)
        img_data_r = img_data_r - sky
        r = img_scale.asinh(img_data_r, scale_min = min_val, non_linear=non_linear_fact)

        rgb_array[:,:,0] = r
        rgb_array[:,:,1] = g
        rgb_array[:,:,2] = b
        ax.append( fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("Reconstructed image")
        plt.imshow(rgb_array, interpolation='nearest', origin='lower')
        break

    img = img_data[i]
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("Filter: "+str(fnames[i])+"\nRedshift: "+str(labels[int(num)][5]))
    plt.imshow(img)
plt.show()




