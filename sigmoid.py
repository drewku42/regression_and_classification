import numpy as np

def sigmoid(z):
    """
    Computes the sigmoid of function z
    Args:
        z (ndarray): A scalar, numpy array of any size.
        
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    
    g = 1 / (1 + np.exp(-z)) #python representation of the sigmoid function
    
    return g
