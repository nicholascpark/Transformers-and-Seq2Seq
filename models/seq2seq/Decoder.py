import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size #2
        self.encoder_hidden_size = encoder_hidden_size #2
        self.decoder_hidden_size = decoder_hidden_size #2
        self.output_size = output_size #10
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        # hidden from encoder : (1,5,2)
        # input : (5,1)
        self.embedding_layer = nn.Embedding(self.output_size, self.emb_size)

        if self.model_type == "RNN":
            self.recurrent_layer = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True)
            # output: (batch size, seq len, decoder_hidden_size)
            # last hidden: (batch size, seq len, decoder_hidden_size)
        else:
            self.recurrent_layer = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True)

        self.linear = nn.Linear(self.decoder_hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.dropout = nn.Dropout(p = dropout)

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        # print(input.shape)
        # print((input))
        embeddings = self.embedding_layer(input)
        # print("embeddings:", embeddings)
        embeddings = self.dropout(embeddings)
        # print(embeddings)
        # print(hidden)
        # print(hidden.shape)
        if self.model_type == "RNN":
            output, hidden = self.recurrent_layer(embeddings, hidden)
        else:
            # hidden = (hidden, )
            output, hidden = self.recurrent_layer(embeddings, hidden)
        # print("output:", output)
        # print("output shape:", output.shape)
        # print("hiddens:", output)
        # print("hiddens shape:", output.shape)
        # print(hidden)
        # print(hidden.shape)

        output = self.linear(output)
        # print("output:", output)
        # print("output shape:", output.shape)
        output = self.logsoftmax(output)[:,0,:]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
            
