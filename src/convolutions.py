
import matplotlib.pyplot as plt
import numpy as np
from images_processing import *



 
def calc_conv_relu_1(image_in, kernel_w, biases_w): # TO-DO: change image_in param name to feature_map
    """
    Perform convolution opetation and RELU for the first layer.
    
    Parameters:
        image_in : the input feature map for the layer
        kernel_w : kernel weights
        biases_w : biases coeffs
    
    Returns:
        cnv_calc: output feature map with dimensions 64x24x24
    """

    num_kernels         = 64
    x_out, y_out        = 24, 24

    cnv_calc            = np.zeros(( x_out, y_out, num_kernels))
    result_widnow_3x3   = np.zeros((3, 3))
    padded_image_in     = np.zeros((y_out+2, x_out+2))

    for iter_kernel in range(64):
        cnv_calc[:,:, iter_kernel] += biases_w[iter_kernel]
        for iter_channel in range(3):
            padded_image_in[1:y_out+1,1:x_out+1] =  image_in[iter_channel][:][:]
            for y in range(24):             # loop the rows
                for x in range(24):         # loop the columns
                    sum_win = 0
                    result_widnow_3x3 = padded_image_in[y:y+3,x:x+3] * kernel_w[:,:,iter_channel,iter_kernel]
                    for ty in range(3):     # loop to calculte the multiplication of each element with the window
                        for tx in range(3):
                            sum_win += result_widnow_3x3[ty][tx]
                    cnv_calc[y][x][iter_kernel] += sum_win
    cnv_calc = np.transpose(cnv_calc, [2,0,1])
    cnv_calc = np.maximum(0, cnv_calc)    
    cnv_calc = np.array(cnv_calc)   
    return cnv_calc     # image shape : channel=64, columns=24, rows=24




def calc_conv_relu_2(image_in, kernel_w, biases_w):
    """
    Perform convolution opetation and RELU for the second layer.
    
    Parameters:
        image_in : the input feature map for the layer
        kernel_w : kernel weights
        biases_w : biases coeffs
    
    Returns:
        cnv_calc: output feature map with dimensions 24x12x12
    """
    num_kernels         = 32
    x_out, y_out        = 12, 12

    cnv_calc            = np.zeros(( num_kernels, x_out, y_out))
    result_widnow_3x3   = np.zeros((3, 3))
    padded_image_in     = np.zeros((y_out+2, x_out+2))

    for iter_kernel in range(32):
        cnv_calc[iter_kernel,:,:] += biases_w[iter_kernel]
        for iter_channel in range(64):
            padded_image_in[1:y_out+1,1:x_out+1] =  image_in[iter_channel][:][:]
            for y in range(12):             # loop the rows
                for x in range(12):         # loop the columns
                    sum_win = 0
                    result_widnow_3x3 = padded_image_in[y:y+3,x:x+3] * kernel_w[:,:,iter_channel,iter_kernel]
                    for ty in range(3):     # loop to calculte the multiplication of each element with the window
                        for tx in range(3):
                            sum_win += result_widnow_3x3[ty][tx]
                    cnv_calc[iter_kernel][y][x] += sum_win
    cnv_calc = np.maximum(0, cnv_calc)       
    return cnv_calc     # image shape : channel=24,  rows=12, columns=12




def calc_conv_relu_3(image_in, kernel_w, biases_w):
    """
    Perform convolution opetation and RELU for the third layer.
    
    Parameters:
        image_in : the input feature map for the layer
        kernel_w : kernel weights
        biases_w : biases coeffs

    Returns:
        cnv_calc: output feature map with dimensions 20x6x6
    """
    
    num_kernels         = 20
    x_out, y_out        = 6, 6

    cnv_calc            = np.zeros(( num_kernels, x_out, y_out))
    result_widnow_3x3   = np.zeros((3, 3))
    padded_image_in     = np.zeros((y_out+2, x_out+2))

    for iter_kernel in range(20):
        cnv_calc[iter_kernel,:,:] += biases_w[iter_kernel]
        for iter_channel in range(32):
            padded_image_in[1:y_out+1,1:x_out+1] =  image_in[iter_channel][:][:]
            for y in range(6):             # loop the rows
                for x in range(6):         # loop the columns
                    sum_win = 0
                    result_widnow_3x3 = padded_image_in[y:y+3,x:x+3] * kernel_w[:,:,iter_channel,iter_kernel]
                    for ty in range(3):     # loop to calculte the multiplication of each element with the window
                        for tx in range(3):
                            sum_win += result_widnow_3x3[ty][tx]
                    cnv_calc[iter_kernel][y][x] += sum_win
    cnv_calc = np.maximum(0, cnv_calc)       
    return cnv_calc     # image shape : channel=20,  rows=6, columns=6



