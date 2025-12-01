


# Improting other files
from images_processing import *
from convolutions import *
from extracting_coeffs import *
from maxpooling import *
from reshape_perceptron_sfmax import *


# Some of parameters passed to the function
out_dim_step1 = [64, 24, 24]
out_dim_step2 = [32, 12, 12]
out_dim_step3 = [20, 6, 6]
maxP_filter = [3, 3]
maxP_stride = [2, 2]


num_test_samples = 1000

def testing_network(num_test_samples):
    checked_statistic = 0
    succeed_statistic = 0

    for i in range(num_test_samples):
        images = Binary_to_Tuple()
        centered_image = center_image(images[i])
        normalized_image = normalise_image(centered_image)

        result_convolution1 = Calc_Conv_RELU_1(normalized_image, matrice_conv1, matrice_biases_1)
        max_pooled_1 = Max_Pool(result_convolution1, maxP_filter, maxP_stride)

        result_convolution2 = Calc_Conv_RELU_2(max_pooled_1, matrice_conv2, matrice_biases_2)
        max_pooled_2 = Max_Pool(result_convolution2, maxP_filter, maxP_stride)

        result_convolution3 = Calc_Conv_RELU_3(max_pooled_2, matrice_conv3, matrice_biases_3)
        max_pooled_3 = Max_Pool(result_convolution3, maxP_filter, maxP_stride)

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

testing_network(num_test_samples)
