import numpy as np
import matplotlib.pyplot as plt



def extract_biases(coeffs_file, label_biases, start_line):
    """
    Extract biases coefficient for convolution operation.
    
    Parameters:
        coeffs_file : the text file from which the coefficient will be extracted
        label_biases: label of coefficients to be extracted
        start_line  : the starting line in the file (which is the line after 
                        the last line of previousely extracted coeffs), to save some time.
    
    Returns:
        biases_list: list of the coefficients
    """

    with open(coeffs_file, 'r') as file:
        lines = file.readlines()

    biases_list = []
 
    im_in = False
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][2:72]

        if not lines[line_indx].find(label_biases) and not im_in :
            im_in = True
            continue
        elif(im_in):
            endx = lines[line_indx].find(']')
            if endx == -1:
                text_line = lines[line_indx][1:]
            else:
                text_line = lines[line_indx][1:endx]

            for x in text_line.split():
                biases_list.append(float(x))
            if endx != -1 :
                break
    return biases_list





def extract_local_w(coeffs_file, label_weights, start_line):
    """
    Extract biases coefficient for perceptron operation.
    
    Parameters:
        coeffs_file : the text file from which the coefficient will be extracted
        label_biases: label of coefficients to be extracted
        start_line  : the starting line in the file (which is the line after 
                        the last line of previousely extracted coeffs), to save some time.
    
    Returns:
        local_matrice: list of the coefficients
    """
    
    with open(coeffs_file, 'r') as file:
        lines = file.readlines()


    local_matrice = np.empty((180, 10))

    im_in = False
    index_180x = 0
    index_10x = 0
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][4:69]
        if not lines[line_indx].find(label_weights) and not im_in :
            im_in = True
            continue
        elif(im_in):
            endx = lines[line_indx].find(']')
            if endx == -1:
                text_line = lines[line_indx][3:]
            else:
                text_line = lines[line_indx][3:endx]

            for x in text_line.split():
                #print(x)
                local_matrice[index_180x][index_10x] = float(x)
                index_10x += 1
            if endx != -1 :
                index_180x += 1
                index_10x = 0
                if index_180x == 180 :
                    break
    return local_matrice




def extract_conv1_w(coeffs_file, label_conv, start_line):
    """
    Extract weights for convolution operation.
    
    Parameters:
        coeffs_file : the text file from which the coefficient will be extracted
        label_biases: label of coefficients to be extracted
        start_line  : the starting line in the file (which is the line after 
                        the last line of previousely extracted coeffs), to save some time.
    
    Returns:
        weights_in_file: list of the extracted weights
    """
    with open(coeffs_file, 'r') as file:
        lines = file.readlines()

    dimensions = [64, 3, 3, 3] 

    kernel_dim  = dimensions[0]
    channel_dim = dimensions[1]
    row_dim     = dimensions[2]
    column_dim  = dimensions[3]

    m3x3_row        = 0
    m3x3_column     = 0
    m3x3_channel    = 0
    m3x3_kernel     = 0
    weights_in_file = np.zeros((column_dim, row_dim, channel_dim, kernel_dim))
    
    im_in = False
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][5:71]
        if not lines[line_indx].find(label_conv) and not im_in :
            im_in = True
            continue
        elif(im_in):
            for x in text_line.split():
                if(x == "_name:"):
                    break

                weights_in_file[m3x3_column][m3x3_row][m3x3_channel][m3x3_kernel] = float(x)

                m3x3_kernel += 1
                if m3x3_kernel > kernel_dim-1 :
                    m3x3_channel += 1
                    m3x3_kernel = 0
                    if m3x3_channel >  channel_dim-1 :
                        m3x3_channel = 0
                        m3x3_row += 1
                        if m3x3_row > row_dim-1 :
                            m3x3_row = 0
                            m3x3_column += 1
                            if m3x3_column == column_dim :
                                break

        if m3x3_column == 3 :
            break
    return weights_in_file




def extract_conv2_w(coeffs_file, label_conv, start_line):
    """
    Extract weights for convolution operation.
    
    Parameters:
        coeffs_file : the text file from which the coefficient will be extracted
        label_biases: label of coefficients to be extracted
        start_line  : the starting line in the file (which is the line after 
                        the last line of previousely extracted coeffs), to save some time.
    
    Returns:
        weights_in_file: list of the extracted weights
    """
    with open(coeffs_file, 'r') as file:
        lines = file.readlines()

    dimensions = [32, 64, 3, 3]

    kernel_dim  = dimensions[0]
    channel_dim = dimensions[1]
    row_dim     = dimensions[2]
    column_dim  = dimensions[3]

    m3x3_row        = 0
    m3x3_column     = 0
    m3x3_channel    = 0
    m3x3_kernel     = 0
    weights_in_file = np.zeros((column_dim, row_dim, channel_dim, kernel_dim))
    kernel_weights  = np.zeros((kernel_dim, channel_dim,row_dim, column_dim))

    im_in = False
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][5:71]
        if not lines[line_indx].find(label_conv) and not im_in :
            im_in = True
            continue
        elif(im_in):
            for x in text_line.split():
                if(x == "_name:"):
                    break

                weights_in_file[m3x3_column][m3x3_row][m3x3_channel][m3x3_kernel] = float(x)
                
                m3x3_kernel += 1
                if m3x3_kernel > kernel_dim-1 :
                    m3x3_channel += 1
                    m3x3_kernel = 0
                    if m3x3_channel >  channel_dim-1 :
                        m3x3_channel = 0
                        m3x3_row += 1
                        if m3x3_row > row_dim-1 :
                            m3x3_row = 0
                            m3x3_column += 1
                            if m3x3_column == column_dim :
                                break
        if m3x3_column == 3 :
            break
    return weights_in_file




def extract_conv3_w(coeffs_file, label_conv, start_line):
    """
    Extract weights for convolution operation.
    
    Parameters:
        coeffs_file : the text file from which the coefficient will be extracted
        label_biases: label of coefficients to be extracted
        start_line  : the starting line in the file (which is the line after 
                        the last line of previousely extracted coeffs), to save some time.
    
    Returns:
        weights_in_file: list of the extracted weights
    """
    with open(coeffs_file, 'r') as file:
        lines = file.readlines()

    dimensions = [20, 32, 3, 3]
   
    kernel_dim  = dimensions[0]
    channel_dim = dimensions[1]
    row_dim     = dimensions[2]
    column_dim  = dimensions[3]

    m3x3_row        = 0
    m3x3_column     = 0
    m3x3_channel    = 0
    m3x3_kernel     = 0
    weights_in_file = np.zeros((column_dim, row_dim, channel_dim, kernel_dim))
    kernel_weights  = np.zeros((kernel_dim, channel_dim,row_dim, column_dim))

    im_in = False
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][5:71]
        if not lines[line_indx].find(label_conv) and not im_in :
            im_in = True
            continue
        elif(im_in):
            for x in text_line.split():
                if(x == "_name:"):
                    break
                
                weights_in_file[m3x3_column][m3x3_row][m3x3_channel][m3x3_kernel] = float(x)

                m3x3_kernel += 1
                if m3x3_kernel > kernel_dim-1 :
                    m3x3_channel += 1
                    m3x3_kernel = 0
                    if m3x3_channel >  channel_dim-1 :
                        m3x3_channel = 0
                        m3x3_row += 1
                        if m3x3_row > row_dim-1 :
                            m3x3_row = 0
                            m3x3_column += 1
                            if m3x3_column == column_dim :
                                break
        if m3x3_column == 3 :
            break
    return weights_in_file











