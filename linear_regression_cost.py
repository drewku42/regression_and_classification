import pandas as pd
import numpy as np

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m = x.shape[0]   # number of training examples
    
    # initialize parameters
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):                 # for each feature x
        f_wb = w * x[i] + b            # find value for f_wb(x)
        dj_dw_i = (f_wb - y[i]) * x[i] # derive f_wb with respect to w
        dj_db_i = f_wb - y[i]          # derive f_wb with respect to b
        dj_db += dj_db_i               # update b value
        dj_dw += dj_dw_i               # update w value
    dj_dw = dj_dw / m                  # divide by number of training examples
    dj_db = dj_db / m                  # divide by number of training examples 
        
    return dj_dw, dj_db                # return weights
