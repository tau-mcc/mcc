import mcclib.utils as utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def plot_image(image, suffix, dpi = 200):
    image = utils.scaleToMax(255, image)
    plt.imshow(image, cmap=matplotlib.cm.get_cmap("Greys"))
    plt.axis('off') # clear x- and y-axes
    figure = plt.gcf() # get current figure
    shape = np.array(image.shape) / dpi 
    figure.set_size_inches(shape[0], shape[1])
    plt.savefig("temp/image_%s.png" % suffix, dpi = dpi)
    plt.close()

def plot_slice(image, suffix):
    image = utils.scaleToMax(1, image)
    npixels = image.shape[0]
    myslice = image[npixels/2, :]
    plt.plot(myslice)
    plt.ylim(0, 1)
    plt.savefig("temp/image_slice_%s.png" % suffix)
    plt.close()

stop_blurring_at = 0.01
white_limit = 0.6 # everything below white_limit * black_value is considered white
black_value = 1.0
white_value = 0.0
sigma = 13
sigma_final = 2
lightening_step = 0.3

filename = "resources/medium_1600.png"

original_maze = utils.loadImage(filename)
original_maze = utils.scaleToMax(black_value, original_maze)
white_pixels = (original_maze == 0.0)

average_grey = np.mean(original_maze)

current_image = original_maze
list_of_images = [current_image]

n = 0

while average_grey > stop_blurring_at and n < 10:
    print "Average value of pixels: %f" % (average_grey,)
    
    current_image = utils.scaleToMax(1.0, current_image)
    current_image = ndimage.gaussian_filter(current_image, sigma)
    
    current_image[current_image < white_limit] = 0
    
    #current_image = np.clip(current_image, white_value, black_value)
    
    #maximum_value = np.max(current_image)
    #current_image = np.clip(current_image, white_value, white_limit*maximum_value)
    
    list_of_images.append(current_image)
    
    average_grey = np.mean(current_image)
    n += 1

for i, image in enumerate(list_of_images):
    plot_image(image, str(i))
    plot_slice(image, str(i))

#sum up
summed_image = np.zeros_like(original_maze, dtype=np.float)
for i, image in enumerate(list_of_images):
    summed_image += image


summed_image = ndimage.gaussian_filter(summed_image, sigma_final)
summed_image[white_pixels] = 0

plot_image(summed_image, "sum")
plot_slice(summed_image, "sum")

#try to recreate original
recreation_of_original = np.zeros_like(original_maze)
recreation_of_original[summed_image > 0.1] = 1.0
plot_image(recreation_of_original, "orig")
plot_slice(recreation_of_original, "orig")

