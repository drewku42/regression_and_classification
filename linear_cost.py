import numpy as np
import pandas as pd

def compute_cost(x, y, w, b):
  """
  Computes the cost function for linear regression.

  Args:
    x (ndarray (m,)): Data, m examples
    y (ndarray (m,)): target values
    w,b (scalar)    : model parameters

  Returns:
    total_cost (float): The cost of using w,b as the parameters for linear regression to fit the data points in x and y
  """
  m = x.shape[0]   # get the number of training examples

  cost_sum = 0
  for i in range(m):                      # for each training example...
    f_wb = w * x[i] + b                   # calculate single variable linear regression model for ith term
    cost = (f_wb - y[i]) ** 2             # compute mean squared error for ith term
    cost_sum += cost                      # incrament sum of each term's mean squared error
  total_cost = (1 / (2 * m)) * cost_sum   #  divide sum by 1/2m

  return total_cost                       # return total cost
