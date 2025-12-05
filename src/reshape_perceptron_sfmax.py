
import numpy as np


def reshape_matrice_to_180(matrices_20x3x3): # FIX NAME
    """
    Reshaping a matrice of 3x3x20 to 180 element.
        Parameters:
            matrices_20x3x3 : the input matrice
        Returns:
            reshaped_matrice: one dimension array of 180 elements
    """
    reshaped_matrice = np.zeros(20*3*3)
    for rows in range (3):
        for columns in range(3):
            for ch in range(20):
                reshaped_matrice[ch + 20*columns + 2*rows*20 ] = matrices_20x3x3[ch][rows][columns]
    return reshaped_matrice




def perceptron(i_1x180, m_180x10, biases_1x10):
    """
    Performs the perceptron operation.
        Parameters:
            i_1x180     : the Input matrice
            m_180x10    : the Weights matrice
            biases_1x10 : the Biases matrice
        Returns:
            result_1x10: 
    """
    result_1x10 = np.empty(10)
    for j in range(10) :
        result_sum_mult = 0
        for i in range(180) :
            result_sum_mult += i_1x180[i] * m_180x10[i][j]
        result_1x10[j] = result_sum_mult + biases_1x10[j]

    return result_1x10




def soft_max(matrice_10x1):
    """
    Calculates the probability of belonging to a class.
        Parameters:
            matrice_10x1    : the Input matrice
        Returns:
            prob_belong_10x1: returns 10 elements, each represents the probability of belonging to the corresponding class 
    """
    prob_belong_10x1 = np.empty(10)
    _sum = 0 
    for inter in range(len(matrice_10x1)):
        _sum += np.exp(matrice_10x1[inter])
    for iter in range(len(matrice_10x1)):
        prob_belong_10x1[iter] = np.exp(matrice_10x1[iter])/_sum

    return prob_belong_10x1