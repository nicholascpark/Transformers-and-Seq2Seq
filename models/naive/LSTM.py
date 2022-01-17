import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size # i_t
        self.hidden_size = hidden_size # h_t-1

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        
        #i_t: input gate

        self.wii = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bii = nn.Parameter(torch.Tensor(self.hidden_size))
        self.whi = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhi = nn.Parameter(torch.Tensor(self.hidden_size))

        # f_t: the forget gate

        self.wif = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bif = nn.Parameter(torch.Tensor(self.hidden_size))
        self.whf = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhf = nn.Parameter(torch.Tensor(self.hidden_size))


        # g_t: the cell gate
        
        self.wig = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.big = nn.Parameter(torch.Tensor(self.hidden_size))
        self.whg = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhg = nn.Parameter(torch.Tensor(self.hidden_size))
        
        
        # o_t: the output gate

        self.wio = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bio = nn.Parameter(torch.Tensor(self.hidden_size))
        self.who = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bho = nn.Parameter(torch.Tensor(self.hidden_size))
        #
        # self.sig = torch.nn.Sigmoid()
        # self.tanh = torch.nn.Sigmoid()


        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        h_t, c_t = None, None
        N, seq_len, input_size = x.shape
        h_t = torch.zeros(N, self.hidden_size) # h_(t-1) not h_t
        c_t = torch.zeros(N, self.hidden_size)

        if init_states != None:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:,t,:] #(N, input_size)
            # print(x_t.shape)
            i_t = torch.sigmoid(x_t @ self.wii + self.bii + h_t @ self.whi + self.bhi)
            f_t = torch.sigmoid(x_t @ self.wif + self.bif + h_t @ self.whf + self.bhf)
            g_t = torch.tanh(x_t @ self.wig + self.big + h_t @ self.whg + self.bhg)
            o_t = torch.sigmoid(x_t @ self.wio + self.bio + h_t @ self.who + self.bho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

# if __name__ == "__main__":

