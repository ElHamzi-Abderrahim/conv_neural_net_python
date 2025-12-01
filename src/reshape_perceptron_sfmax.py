
import numpy as np

#####################################################
################ RESHAPING OPERATION ################
#####################################################
def reshape_matrice_to_180(matrices_20x3x3): # FIX NAME
    reshaped_matrice = np.zeros(20*3*3)
    for rows in range (3):
        for columns in range(3):
            for ch in range(20):
                reshaped_matrice[ch + 20*columns + 2*rows*20 ] = matrices_20x3x3[ch][rows][columns]
    return reshaped_matrice




#####################################################
################## FULLY CONNECTING #################
#####################################################
def perceptron(I_1x180, M_180x10, Biases_1x10):
    result_1x10 = np.empty(10)
    for j in range(10) :
        result_sum_Mult = 0
        for i in range(180) :
            result_sum_Mult += I_1x180[i] * M_180x10[i][j]
        result_1x10[j] = result_sum_Mult + Biases_1x10[j]

    return result_1x10




#####################################################
##################### SOFT MAX #####################
#####################################################
def soft_max(matrice_10x1):
    Prob_appart_10x1 = np.empty(10)
    sum = 0 
    for inter in range(len(matrice_10x1)):
        sum += np.exp(matrice_10x1[inter])
    for iter in range(len(matrice_10x1)):
        Prob_appart_10x1[iter] = np.exp(matrice_10x1[iter])/sum

    return Prob_appart_10x1