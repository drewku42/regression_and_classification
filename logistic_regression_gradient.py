import numpy as np
import pandas as pd

def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)   # find partial derivative of w
    dj_db = 0.   # partial derivative of b is 0
  
    for i in range(m):                             # for each training example
        f_wb_i = sigmoid(np.dot(X[i],w) + b)       # plug logistic regression function into sigmoid function
        err_i = f_wb_i - y[i]                      # calculate error of ith term
        for j in range(n):                         # for each feature
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]   # calculate partial of jth feature
        dj_db = dj_db + err_i                      # update bias parameter
    dj_dw = dj_dw/m                                # update w parameter
    dj_db = dj_db/m                                # update b parameter

        
    return dj_db, dj_dw   # return weights
