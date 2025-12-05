


# Improting other files
from images_processing import *
from convolutions import *
from extracting_coeffs import *
from maxpooling import *
from reshape_perceptron_sfmax import *


# Global parameters that are passed to some functions inside other included files
out_dim_step1 = [64, 24, 24]
out_dim_step2 = [32, 12, 12]
out_dim_step3 = [20, 6, 6]
maxP_filter = [3, 3]
maxP_stride = [2, 2]



def testing_network(num_test_samples, data_set_test, coeffs_file):
    """
    Main testing function for the CNN implementation
        Parameters:
            num_test_samples: The number of images to be sampled for testing the CNN
        Returns:
            No retrun value, it just prints the success rate of the performed test on test data set
    """

    checked_statistic = 0
    succeed_statistic = 0

    # Extracting images from test data set
    images = binary_to_tuple(data_set_test, num_test_samples, image_size)
    
    # Extracting coeffecients for convolution operations
    matrice_conv1 = extract_conv1_w(coeffs_file, "tensor_name:  conv1/weights", 0)
    matrice_conv2 = extract_conv2_w(coeffs_file, "tensor_name:  conv2/weights", 452)
    matrice_conv3 = extract_conv3_w(coeffs_file, "tensor_name:  conv3/weights", 5080)
    # Extracting biases for convolution operations
    matrice_biases_1     = extract_biases(coeffs_file, "tensor_name:  conv1/biases", 0)
    matrice_biases_2     = extract_biases(coeffs_file, "tensor_name:  conv2/biases", 453)
    matrice_biases_3     = extract_biases(coeffs_file, "tensor_name:  conv3/biases", 5080)
    
    # Extracting coeffecients for perceptron operation
    matrice_local        = extract_local_w(coeffs_file, "tensor_name:  local3/weights", 6536)
    matrice_biases_local = extract_biases(coeffs_file, "tensor_name:  local3/biases", 6530)

    # Iterate a number of images and check if the identified class of the image is the same 
    # as the label of the image.
    for i in range(num_test_samples):
        centered_image = center_image(images[i])
        normalized_image = normalise_image(centered_image)

        result_convolution1 = calc_conv_relu_1(normalized_image, matrice_conv1, matrice_biases_1)
        max_pooled_1 = max_pool(result_convolution1, maxP_filter, maxP_stride)

        result_convolution2 = calc_conv_relu_2(max_pooled_1, matrice_conv2, matrice_biases_2)
        max_pooled_2 = max_pool(result_convolution2, maxP_filter, maxP_stride)

        result_convolution3 = calc_conv_relu_3(max_pooled_2, matrice_conv3, matrice_biases_3)
        max_pooled_3 = max_pool(result_convolution3, maxP_filter, maxP_stride)

        reshaped = reshape_matrice_to_180(max_pooled_3)

        fully_connected = perceptron(reshaped, matrice_local, matrice_biases_local)

        result_class = soft_max(fully_connected)

        activated_class = np.argmax(result_class)
        print(result_class)
        print(f"Activated Class: {activated_class}")

        checked_statistic += 1
        if activated_class == centered_image[0]:  
            succeed_statistic += 1
        success_rate = (succeed_statistic / checked_statistic) * 100
        print(f"Success Rate: {success_rate}%")




# Files used for the test
data_set_test = "project_data/test_batch.bin"
coeffs_file   = "project_data/CNN_coeffs_x3x.txt"

# Number of image to be inputed to the CNN
num_test_samples = 100
# The size of each image
image_size = 3073

# Run the test of the CNN network:
testing_network(num_test_samples, data_set_test, coeffs_file)
