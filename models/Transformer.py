# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43, dropout = 0):
        '''
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(p = self.dropout)

        seed_torch(0)

        print("The number of self attention heads:", self.num_heads)
        print("The dropout probability:", self.dropout)
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.word_embedding = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.word_embedding_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        if self.num_heads > 2:
            # Head #3
            self.k3 = nn.Linear(self.hidden_dim, self.dim_k)
            self.v3 = nn.Linear(self.hidden_dim, self.dim_v)
            self.q3 = nn.Linear(self.hidden_dim, self.dim_q)

            # Head #4
            self.k4 = nn.Linear(self.hidden_dim, self.dim_k)
            self.v4 = nn.Linear(self.hidden_dim, self.dim_v)
            self.q4 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.linear_ff1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.relu_ff = nn.ReLU()
        self.linear_ff2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.linear_final = nn.Linear(self.hidden_dim, self.output_size)
        self.softmax_final = nn.Softmax(dim=2)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        outputs = None
        inputs = inputs.to(self.device)
        outputs = self.embed(inputs)
        # outputs = self.dropout_layer(outputs)
        outputs = self.multi_head_attention(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.final_layer(outputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        # print(inputs.shape)
        inputs = inputs.to(self.device)
        word_embeddings = self.word_embedding(inputs)
        position_encoding = np.arange(self.max_length)
        position_encoding = np.tile(position_encoding,(inputs.shape[0],1))
        position_encoding = torch.from_numpy(position_encoding).type(torch.LongTensor).to(self.device)
        # print(type(position_encoding[0,0].item()))
        # print(type(inputs[0,0].item()))
        # print(word_embeddings)
        # print(word_embeddings)
        position_embeddings = self.position_embedding(position_encoding)
        embeddings = word_embeddings + position_embeddings
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        outputs = None
        inputs = inputs.to(self.device)

        k1 = self.k1(inputs) # (N, T, dim_k1)
        v1 = self.v1(inputs) # (N, T, dim_v1)
        q1 = self.q1(inputs) # (N, T, dim_q1)

        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)

        if self.num_heads > 2:

            k3 = self.k3(inputs)
            v3 = self.v3(inputs)
            q3 = self.q3(inputs)

            k4 = self.k4(inputs)
            v4 = self.v4(inputs)
            q4 = self.q4(inputs)


        k1 = torch.transpose(k1, 1, 2)
        q1k1 = torch.einsum('ijk, ikl-> ijl', q1, k1)# (N, T, T)
        q1k1 = torch.div(q1k1, np.sqrt(self.dim_k))
        q1k1 = self.softmax(q1k1) #(N,T,T)
        q1k1v1 = torch.einsum('ijl, ilm -> ijm',q1k1,v1) #(N,T ,T)
        # print(q1k1v1.shape)

        k2 = torch.transpose(k2, 1, 2)
        q2k2 = torch.einsum('ijk, ikl-> ijl', q2, k2)# (N, T, T)
        q2k2 = torch.div(q2k2, np.sqrt(self.dim_k))
        q2k2 = self.softmax(q2k2) #(N,T,T)
        q2k2v2 = torch.einsum('ijl, ilm -> ijm',q2k2,v2) #(N,T ,T)
        # print(q2k2v2.shape)
        concat = torch.cat((q1k1v1, q2k2v2), dim=2)

        if self.num_heads > 2:

            k3 = torch.transpose(k3, 1, 2)
            q3k3 = torch.einsum('ijk, ikl-> ijl', q3, k3)# (N, T, T)
            q3k3 = torch.div(q3k3, np.sqrt(self.dim_k))
            q3k3 = self.softmax(q3k3) #(N,T,T)
            q3k3v3 = torch.einsum('ijl, ilm -> ijm',q3k3,v3) #(N,T ,T)

            k4 = torch.transpose(k4, 1, 2)
            q4k4 = torch.einsum('ijk, ikl-> ijl', q4, k4)# (N, T, T)
            q4k4 = torch.div(q4k4, np.sqrt(self.dim_k))
            q4k4 = self.softmax(q4k4) #(N,T,T)
            q4k4v4 = torch.einsum('ijl, ilm -> ijm',q4k4,v4) #(N,T ,T)

            concat = torch.cat((q1k1v1, q2k2v2, q3k3v3, q4k4v4), dim=2)

        projection = self.attention_head_projection(concat)
        # print(projection.shape)
        # print(inputs.shape)
        outputs = self.norm_mh(inputs + projection)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = None
        inputs = inputs.to(self.device)
        outputs = self.linear_ff1(inputs)
        outputs = self.relu_ff(outputs)
        outputs = self.linear_ff2(outputs)
        # print(outputs.shape)
        # print(inputs.shape)
        outputs = self.norm_ff(inputs + outputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = None
        inputs = inputs.to(self.device)
        outputs = self.linear_final(inputs)
        # print(outputs)
        # print(outputs.shape)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True