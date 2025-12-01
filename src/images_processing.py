
import numpy as np
import matplotlib.pyplot as plt

image_number = 10000
image_size = 3073

###################################################
################ OPENING BATCH ####################
###################################################
def open_file_batch() :
    with open("project_data/test_batch.bin", "rb") as file :
        binary_data = file.read()
    return binary_data


################################################
################ EXTRACT IMAGES ################
################################################
def Binary_to_Tuple(): # (label, red_32x32, green_32x32, blue_32x32)
    start = 0
    images = []

    binary_data = open_file_batch() 

    for indx in range(image_number):
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



####################################################
################ CENTRALIZING IMAGE ################
####################################################
def center_channel(matrice_32x32) :
    debut_ligne = (32 - 24) // 2
    fin_ligne = debut_ligne + 24
    debut_colonne = (32 - 24) // 2
    fin_colonne = debut_colonne + 24
    matrice_24x24 = matrice_32x32[debut_ligne:fin_ligne, debut_colonne:fin_colonne]
    return matrice_24x24

def center_image(nCentered_Image):
    label = nCentered_Image[0]
    red_ch_24x24     =  center_channel(nCentered_Image[1])
    green_ch_24x24   =  center_channel(nCentered_Image[2])
    blue_ch_24x24    =  center_channel(nCentered_Image[3])
    Centered_Image = (label, red_ch_24x24, green_ch_24x24, blue_ch_24x24)
    return Centered_Image



###################################################
################ NORMALIZING IMAGE ################
###################################################
def normalise_image(centered_image):
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



##################################################
################ DISPLAYING IMAGE ################
##################################################
def Display_image(image):
    rgb_image = np.stack(image, axis=-1)
    plt.imshow(rgb_image)
    plt.show()

