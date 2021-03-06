from torch import nn, Tensor, tanh
import torch
from math import sqrt


class BasicRNNCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BasicRNNCell, self).__init__()
        """
        Creates an RNN cell with a tanH activation function
         
        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell. 
        
        """
        self.vocab_size = vocab_size  # n
        self.hidden_size = hidden_size  # m

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables
        # init all parameters uniformly from [-k, k] where k = (1/m)**0.5
        k = torch.sqrt(torch.tensor(1 / hidden_size))

        # W, the input weights matrix has size (n x m) where n is
        # the number of input features and m is the hidden size
        self.W = torch.nn.Parameter(torch.Tensor(vocab_size, hidden_size).uniform_(-k, k))
        # V, the hidden state weights matrix has size (m, m)
        self.V = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size).uniform_(-k, k))
        # b, the vector of bias, has size (m)
        self.b = torch.nn.Parameter(torch.Tensor(hidden_size).uniform_(-k, k))


    def forward(self, x, h):
        """
        Defines the forward propagation of an RNN cell with a tanH as activation function

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input features. x is
            the input data of the current time-step. In a multi-layer RNN, x is the previous layer's hidden state
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous layer

        Return
        ------
        h: (Tensor) of size (B x m), the new hidden state

        """
        a = self.b + x.mm(self.W) + h.mm(self.V)
        h = tanh(a)
        return h

