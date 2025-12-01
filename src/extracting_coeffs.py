import numpy as np
import matplotlib.pyplot as plt

image_number = 10000
image_size = 3073


################################################
################ EXTRACT BIASES ################
################################################
def extract_biases(label_biases, start_line):
    with open("project_data/CNN_coeffs_3x3.txt", 'r') as file:
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

matrice_biases_1 = extract_biases("tensor_name:  conv1/biases", 0)
matrice_biases_2 = extract_biases("tensor_name:  conv2/biases", 453)
matrice_biases_3 = extract_biases("tensor_name:  conv3/biases", 5080)

matrice_biases_local = extract_biases("tensor_name:  local3/biases", 6530)


####################################################
################ EXTRACT KERNEL WIEGHTS ############
####################################################
def extract_local_weights():
    with open("project_data/CNN_coeffs_3x3.txt", 'r') as file:
        lines = file.readlines()

    label_local_weights = "tensor_name:  local3/weights"
    start_line = 6536

    local_matrice = np.empty((180, 10))

    im_in = False
    index_180x = 0
    index_10x = 0
    for line_indx in range(start_line, len(lines)):
        text_line = lines[line_indx][4:69]
        if not lines[line_indx].find(label_local_weights) and not im_in :
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

matrice_local = extract_local_weights()


########################################################
################ EXTRACT Conv 1 weights ################
########################################################
def extract_conv1_weights():
    with open("project_data/CNN_coeffs_3x3.txt", 'r') as file:
        lines = file.readlines()

    dimensions_step1 = [64, 3, 3, 3]
    dimensions = dimensions_step1
    start_line = 0
    label_conv = "tensor_name:  conv1/weights"

    kernel_dim  = 64
    channel_dim = 3
    row_dim     = 3
    column_dim  = 3

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

matrice_conv1 = extract_conv1_weights()

########################################################
################ EXTRACT Conv 2 weights ################
########################################################
def extract_conv2_weights():
    with open("project_data/CNN_coeffs_3x3.txt", 'r') as file:
        lines = file.readlines()
    start_line = 452
    label_conv = "tensor_name:  conv2/weights"

    kernel_dim  = 32
    channel_dim = 64
    row_dim     = 3
    column_dim  = 3

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

matrice_conv2 = extract_conv2_weights()

########################################################
################ EXTRACT Conv 3 weights ################
########################################################
def extract_conv3_weights():
    with open("project_data/CNN_coeffs_3x3.txt", 'r') as file:
        lines = file.readlines()

    dimensions_step3 = [20, 32, 3, 3]
    dimensions = dimensions_step3
    start_line = 5080
    label_conv = "tensor_name:  conv3/weights"

    kernel_dim  = 20
    channel_dim = 32
    row_dim     = 3
    column_dim  = 3

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

matrice_conv3 = extract_conv3_weights()










