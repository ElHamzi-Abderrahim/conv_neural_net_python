
import matplotlib.pyplot as plt
import numpy as np
from images_processing import *


def calc_conv_relu(feature_map, kernel_w, biases_w, num_in_channel,out_dims):
    """
    Perform convolution opetation and RELU for each layer.
    
    Parameters:
        feature_map          : the input feature map for the layer
        kernel_w             : kernel weights
        biases_w             : biases coeffs
        num_in_channel (int) : number of input channels
    
    Returns:
        cnv_relu_calc        : output feature map with dimensions out_dims[num_kernels, x_out, y_out]
    """

    num_kernels         = out_dims[0]
    x_out, y_out        = out_dims[1], out_dims[2]

    cnv_calc            = np.zeros(( num_kernels, x_out, y_out))
    result_widnow_3x3   = np.zeros((3, 3))
    padded_feature_map     = np.zeros((y_out+2, x_out+2))

    for iter_kernel in range(num_kernels):
        cnv_calc[iter_kernel,:,:] += biases_w[iter_kernel]
        for iter_channel in range(num_in_channel):
            padded_feature_map[1:y_out+1,1:x_out+1] =  feature_map[iter_channel][:][:]
            for y in range(y_out):             # loop the rows
                for x in range(x_out):         # loop the columns
                    sum_win = 0
                    result_widnow_3x3 = padded_feature_map[y:y+3,x:x+3] * kernel_w[:,:,iter_channel,iter_kernel]
                    for ty in range(3):     # loop to calculte the multiplication of each element with the window
                        for tx in range(3):
                            sum_win += result_widnow_3x3[ty][tx]
                    cnv_calc[iter_kernel][y][x] += sum_win
    cnv_relu_calc = np.maximum(0, cnv_calc)       
    return cnv_relu_calc



