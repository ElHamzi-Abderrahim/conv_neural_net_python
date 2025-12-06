


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



def testing_network(num_test_samples, data_set_test_f, coeffs_f, image_size):
    """
    Main testing function for the CNN implementation
        Parameters:
            num_test_samples: the number of images to be sampled for testing the CNN
            data_set_test_f : data set test file
            coeffs_f        : the file on which all coefficient, needed in the CNN, are stored
            image_size      : the size of each image stored in the data_set_test_f in bytes
        Returns:
            No retrun value, it just prints the success rate of the performed test on test data set
    """
    
    print("+-----------------------------------------------------------------------------")
    print("| The test data set file path              : ", data_set_test_f)
    print("| The coefficient to be used for operations: ", coeffs_f)
    print("| The number of images to be processed     : ", num_test_samples)
    print("| The size of each image in bytes          : ", image_size)
    print("+-----------------------------------------------------------------------------")
    print("")



    checked_statistic = 0
    succeed_statistic = 0

    # Extracting images from test data set
    images = binary_to_tuple(data_set_test_f, num_test_samples, image_size)
    
    # Extracting coeffecients for convolution operations
    matrice_conv1 = extract_conv1_w(coeffs_f, "tensor_name:  conv1/weights", 0)
    matrice_conv2 = extract_conv2_w(coeffs_f, "tensor_name:  conv2/weights", 452)
    matrice_conv3 = extract_conv3_w(coeffs_f, "tensor_name:  conv3/weights", 5080)
    # Extracting biases for convolution operations
    matrice_biases_1     = extract_biases(coeffs_f, "tensor_name:  conv1/biases", 0)
    matrice_biases_2     = extract_biases(coeffs_f, "tensor_name:  conv2/biases", 453)
    matrice_biases_3     = extract_biases(coeffs_f, "tensor_name:  conv3/biases", 5080)
    
    # Extracting coeffecients for perceptron operation
    matrice_local        = extract_local_w(coeffs_f, "tensor_name:  local3/weights", 6536)
    matrice_biases_local = extract_biases(coeffs_f, "tensor_name:  local3/biases", 6530)
    
    print("+=====================================================")
    print("                Start of the test                     ")
    print("+=====================================================")
    
    success_rate = 0
    # Iterate a number of images and check if the identified class of the image is the same 
    # as the label of the image.
    for i in range(num_test_samples):
        centered_image = center_image(images[i])
        normalized_image = normalise_image(centered_image)

        result_convolution1 = calc_conv_relu(normalized_image, matrice_conv1, matrice_biases_1, 3, [64, 24, 24])
        max_pooled_1 = max_pool(result_convolution1, maxP_filter, maxP_stride)

        result_convolution2 = calc_conv_relu(max_pooled_1, matrice_conv2, matrice_biases_2, 64 ,[32, 12, 12])
        max_pooled_2 = max_pool(result_convolution2, maxP_filter, maxP_stride)

        result_convolution3 = calc_conv_relu(max_pooled_2, matrice_conv3, matrice_biases_3, 32, [20, 6, 6])
        max_pooled_3 = max_pool(result_convolution3, maxP_filter, maxP_stride)

        reshaped = reshape_matrice_to_180(max_pooled_3)

        fully_connected = perceptron(reshaped, matrice_local, matrice_biases_local)

        result_class = soft_max(fully_connected)

        activated_class = np.argmax(result_class)
        # print(result_class)
        print("+-------------------------------------------------------------")
        print(f"| Activated Class: {activated_class}")

        checked_statistic += 1
        if activated_class == centered_image[0]:  
            succeed_statistic += 1
            print("|    > [Pass] Result matches image label.")
        else:    
            print("|    > [Fail] Result is different from image label.")
            
        success_rate = (succeed_statistic / checked_statistic) * 100
        print(f"| - Success Rate (in Real-Time): {success_rate}%")
        
    print("+=====================================================")
    print("                  End of the test                     ")
    print("+=====================================================")

    print("+-----------------------------------------------------")
    print(f"| The final report of Success Rate: {success_rate}%")
    print("+-----------------------------------------------------")



