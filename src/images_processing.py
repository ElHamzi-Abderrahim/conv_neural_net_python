
import numpy as np
import matplotlib.pyplot as plt



def open_data_file(data_test_file):
    """
    Extracting binary data from a data_test_file.
    
    Parameters:
        data_test_file : path to the test data set file
    
    Returns:
        binary_data    : data in binary format
    """
    with open(data_test_file, "rb") as file :
        binary_data = file.read()
    return binary_data




def binary_to_tuple(data_test_file, image_number, image_size):
    """
    Converting binary data to a list of tuples, each tuple is storing an image RGB with its label.
    
    Parameters:
        data_test_file : path to the test data set file
        image_number   : number of images to fetched from the binary data
        image_size     : size of each image (bytes)
    
    Returns:
        images         : stores a list of tuple of images
                            each tuple (label, red_32x32, green_32x32, blue_32x32)
    """
    start = 0
    images = []

    binary_data = open_data_file(data_test_file) 

    for _indx in range(image_number):
        bin_image = np.frombuffer(binary_data[start : start + image_size], dtype=np.uint8)
        label = bin_image[0]
        image_pixs = bin_image[1:]

        red_ch_32x32    = image_pixs[:1024].reshape(32, 32)
        green_ch_32x32  = image_pixs[1024:2048].reshape(32, 32)
        blue_ch_32x32   = image_pixs[2048:].reshape(32, 32)
        tuple_image = (label, red_ch_32x32, green_ch_32x32, blue_ch_32x32)
        start = start + 3073
        images.append(tuple_image)
    return images # image is : 0 => 9999 which means 10000



def center_channel(matrice_32x32) :
    """
    Centering a 32x32 channel (Red, Green, or Blue) to 24x24.
    
    Parameters:
        matrice_32x32 : channel with a dimensions 32x32
    
    Returns:
        matrice_24x24 : converted channel with dimensions 24x24 
    """

    debut_ligne = (32 - 24) // 2
    fin_ligne = debut_ligne + 24
    debut_colonne = (32 - 24) // 2
    fin_colonne = debut_colonne + 24
    matrice_24x24 = matrice_32x32[debut_ligne:fin_ligne, debut_colonne:fin_colonne]
    return matrice_24x24




def center_image(non_centered_image):
    """
    Centering a 32x32 image (Red, Green, and Blue) to an image of 24x24.
    
    Parameters:
        non_centered_image : non centered RGB image 32x32
    
    Returns:
        centered_image     : centered RGB image 32x32
    """
    label = non_centered_image[0]
    red_ch_24x24     =  center_channel(non_centered_image[1])
    green_ch_24x24   =  center_channel(non_centered_image[2])
    blue_ch_24x24    =  center_channel(non_centered_image[3])
    centered_image = (label, red_ch_24x24, green_ch_24x24, blue_ch_24x24)
    return centered_image




def normalise_image(centered_image):
    """
    Normalizing the image format without touching the image properties, check README to see the formula.

    Parameters:
        centered_image   : centered RGB image 24x24 image to be normalized
    
    Returns:
        normalized_image : RGB image 32x32 normalized image
    """
    red_24x24 = centered_image[1].flatten()
    green_24x24 = centered_image[2].flatten()
    blue_24x24 = centered_image[3].flatten()

    all_values = np.concatenate((red_24x24, green_24x24, blue_24x24))

    mean = np.mean(all_values)
    std_dev = np.std(all_values)

    max_value = max(std_dev, 1 / np.sqrt(1728))

    normalized_red = ((red_24x24 - mean) / max_value).reshape(24, 24)
    normalized_green = ((green_24x24 - mean) / max_value).reshape(24, 24)
    normalized_blue = ((blue_24x24 - mean) / max_value).reshape(24, 24)

    normalized_image = [normalized_red, normalized_green, normalized_blue]

    return normalized_image




def display_image(image):
    """
    Basic function to display the image
    
    Parameters:
        image: image to be displayed
    
    Returns:
        None
    """
    rgb_image = np.stack(image, axis=-1)
    plt.imshow(rgb_image)
    plt.show()

