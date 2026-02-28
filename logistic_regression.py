import numpy as np

class LogisticRegression():
  def __init__(self, epoch=1000, L=1e-4, threshold=0.5):
    self.epoch = epoch
    self.L = L
    self.weights = None
    self.threshold = threshold
  
  def __proba__(self, x_test):
    z = np.dot(x_test, self.weights)
    return 1/(1+np.exp(-z))
  
  def train(self, x_train, y_train):
    m, features = x_train.shape
    self.weights = np.zeros(features)
    
    for _ in range(self.epoch):
      z = np.dot(x_train, self.weights)
      y_h = 1/(1 + np.exp(-z))
      err = y_h - y_train
      gradient = np.dot(x_train.T, err) / m
      self.weights -= self.L * gradient
      
  def predict(self, x_test):
    probabilities = self.__proba__(x_test)
    return (probabilities >= self.threshold).astype(int)