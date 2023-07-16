import numpy as np

def compute_cost(X, y, w, b):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns:
      total_cost : (scalar) cost 
    """
  
    m = X.shape[0]   # get number of training examples
    total_cost = 0.0   # initialize cost as type float
    for i in range(m):   # for each example...
        z_i = np.dot(X[i],w) + b   # compute dot product of x examples and weights
        f_wb_i = sigmoid(z_i)   # plug dot product into sigmoid function
        total_cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)   # incrament total cost by cost of each training example
    total_cost = total_cost / m   # divide cost by number of training examples

    return total_cost
