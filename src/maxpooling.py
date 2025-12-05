import numpy as np


def max_pool(input_fm ,filter_size, stride) :
    """
    Reshaping a matrice of 3x3x20 to 180 element.
        Parameters:
            input_fm    : The input feature map from the previous layer
            filter_size : The filter size from which the max value will be taken
            stride      : The stride matrice for stepping through the matrice
        Returns:
            output_fm : The output feature map of the layer
    """

    dim_fx = int(len(input_fm[0])/2.0)
    dim_fy = int(len(input_fm[0])/2.0)
    dim_ch = len(input_fm)

    output_fm               = np.empty( (dim_ch, dim_fx, dim_fy) )
    nbre_iteration_column   = len(input_fm[0][0]) - filter_size[0] + 3
    nbre_iteration_row      = len(input_fm[0]) - filter_size[1] + 3

    for index_channel in range(len(input_fm)):
        for row in range(0, nbre_iteration_row, stride[1]):
            for column in range(0 , nbre_iteration_column, stride[0]):
                filter_mat = input_fm[index_channel][ row: row+filter_size[1] , column: column+filter_size[0] ]
                column_out = int(column/stride[1])
                row_out = int(row/stride[0])
                output_fm[index_channel][row_out][column_out] = np.max(filter_mat) 
    return output_fm

