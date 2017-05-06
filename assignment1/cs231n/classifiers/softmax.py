import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  scores = X.dot(W)
  for i in range(num_train):
    score = scores[i]
    logC = -np.max(score)
    score = score + logC
    loss += - np.log(np.exp(score[y[i]])/np.sum(np.exp(score)))
    for j in range(num_class):
      softmax_output = np.exp(score[j]) / np.sum(np.exp(score))
      if j == y[i]:
        dW[:, j] += (-1 + softmax_output) * X[i].T
      else:
        dW[:, j] += softmax_output * X[i].T

  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  logC = -np.max(scores,axis=1)
  scores = scores + logC.reshape((-1,1))
  sum_j = np.sum(np.exp(scores),axis=1).reshape((-1,1))
  score_yi = scores[np.arange(0,num_train,1),y].reshape((-1,1))
  loss = np.sum(-np.log(np.exp(score_yi)/sum_j))
  softmax_output = np.exp(scores)/sum_j
  indh = np.zeros_like(softmax_output)
  indh[np.arange(0,num_train,1),y] = 1
  dW = X.T.dot(softmax_output-indh)
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW/num_train + reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

