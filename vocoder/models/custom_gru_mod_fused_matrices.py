import math
import torch
import torch.nn as nn
import torch.nn.parameter as param

class GRUCell(torch.jit.ScriptModule):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, 
        rnn1_weights_hh,
        rnn1_weights_ih,
        rnn1_bias_hh,
        rnn1_bias_ih,
        rnn2_weights_hh,
        rnn2_weights_ih,
        rnn2_bias_hh,
        rnn2_bias_ih,
        I_learned_params,
        fc1_learned_params,
        fc2_learned_params,
        fc3_learned_params
        ):
        super(GRUCell, self).__init__()
        self.input_size = input_size,
        self.hidden_size = hidden_size
        self.rnn1_weight_hh = rnn1_weights_hh
        self.rnn1_weight_ih = rnn1_weights_ih
        self.rnn1_bias_hh = rnn1_bias_hh
        self.rnn1_bias_ih = rnn1_bias_ih

        self.rnn2_weight_hh = rnn2_weights_hh
        self.rnn2_weight_ih = rnn2_weights_ih
        self.rnn2_bias_hh = rnn2_bias_hh
        self.rnn2_bias_ih = rnn2_bias_ih
        
        self.I_weight, self.I_bias = I_learned_params
        self.fc1_weight, self.fc1_bias = fc1_learned_params
        self.fc2_weight, self.fc2_bias = fc2_learned_params
        self.fc3_weight, self.fc3_bias = fc3_learned_params
        
        self.I_weight = param.Parameter( self.I_weight.t() )
        self.fc1_weight = param.Parameter(self.fc1_weight.t() )
        self.fc2_weight = param.Parameter(self.fc2_weight.t() )
        self.fc3_weight = param.Parameter(self.fc3_weight.t() )

        self.I_bias = param.Parameter( self.I_bias )
        self.fc1_bias = param.Parameter(self.fc1_bias)
        self.fc2_bias = param.Parameter(self.fc2_bias)
        self.fc3_bias = param.Parameter(self.fc3_bias)

        # rnn1 ----------------------------------------------------------------------------
        self.weight1_ih_1, self.weight1_ih_2, self.weight1_ih_3 = self.rnn1_weight_ih.chunk( 3 )
        self.bias1_ih_1, self.bias1_ih_2, self.bias1_ih_3 = self.rnn1_bias_ih.chunk( 3 )

        self.weight1_hh_1, self.weight1_hh_2, self.weight1_hh_3 = self.rnn1_weight_hh.chunk( 3 )
        self.bias1_hh_1, self.bias1_hh_2, self.bias1_hh_3 = self.rnn1_bias_hh.chunk( 3 )

        self.bias11 = param.Parameter(self.bias1_hh_1 + self.bias1_ih_1)
        self.bias12 = param.Parameter(self.bias1_hh_2 + self.bias1_ih_2)

        self.weight1_hh_1 = param.Parameter(self.weight1_hh_1.t())
        self.weight1_hh_2 = param.Parameter(self.weight1_hh_2.t())
        self.weight1_hh_3 = param.Parameter(self.weight1_hh_3.t())

        self.weight1_ih_1 = param.Parameter(self.weight1_ih_1.t())
        self.weight1_ih_2 = param.Parameter(self.weight1_ih_2.t())
        self.weight1_ih_3 = param.Parameter(self.weight1_ih_3.t())

        self.bias1_ih_3 = param.Parameter(self.bias1_ih_3)
        self.bias1_hh_3 = param.Parameter(self.bias1_hh_3)


        # rnn2 ----------------------------------------------------------------------------
        self.weight2_ih_1, self.weight2_ih_2, self.weight2_ih_3 = self.rnn2_weight_ih.chunk( 3 )
        self.bias2_ih_1, self.bias2_ih_2, self.bias2_ih_3 = self.rnn2_bias_ih.chunk( 3 )

        self.weight2_hh_1, self.weight2_hh_2, self.weight2_hh_3 = self.rnn2_weight_hh.chunk( 3 )
        self.bias2_hh_1, self.bias2_hh_2, self.bias2_hh_3 = self.rnn2_bias_hh.chunk( 3 )

        self.bias21 = param.Parameter(self.bias2_hh_1 + self.bias2_ih_1)
        self.bias22 = param.Parameter(self.bias2_hh_2 + self.bias2_ih_2)

        self.weight2_hh_1 = param.Parameter(self.weight2_hh_1.t())
        self.weight2_hh_2 = param.Parameter(self.weight2_hh_2.t())
        self.weight2_hh_3 = param.Parameter(self.weight2_hh_3.t())

        self.weight2_ih_1 = param.Parameter(self.weight2_ih_1.t())
        self.weight2_ih_2 = param.Parameter(self.weight2_ih_2.t())
        self.weight2_ih_3 = param.Parameter(self.weight2_ih_3.t())

        self.bias2_ih_3 = param.Parameter(self.bias2_ih_3)
        self.bias2_hh_3 = param.Parameter(self.bias2_hh_3)
        self.I_weight_mul_weight1_ih_1 = param.Parameter(torch.mm( self.I_weight, self.weight1_ih_1 ))
        self.I_weight_mul_weight1_ih_2 = param.Parameter(torch.mm( self.I_weight, self.weight1_ih_2 ))
        self.I_weight_mul_weight1_ih_3 = param.Parameter(torch.mm( self.I_weight, self.weight1_ih_3 ))

        v = torch.unsqueeze(self.I_bias, 0)
        self.I_bias_mul_weight1_ih_1 = torch.mm( v, self.weight1_ih_1 )
        self.I_bias_mul_weight1_ih_2 = torch.mm( v, self.weight1_ih_2 )
        self.I_bias_mul_weight1_ih_3 = torch.mm( v, self.weight1_ih_3 )

        self.bias11 = param.Parameter(self.bias11 + self.I_bias_mul_weight1_ih_1)
        self.bias12 = param.Parameter(self.bias12 + self.I_bias_mul_weight1_ih_2)
        self.bias1_ih_3 = param.Parameter(self.I_bias_mul_weight1_ih_3 + self.bias1_ih_3)
        # self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    

    @torch.jit.script_method
    def forward(self, x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t):
        
        x = torch.cat([x, m_t, a1_t], dim=1)
                
        # x = torch.mm( x, self.I_weight ) + self.I_bias

        #rnn1------------------------------
        # x = x + h1
        # i_r = torch.mm( x, self.weight1_ih_1 )
        i_r = torch.mm( x, self.I_weight_mul_weight1_ih_1 )
        # i_i = torch.mm( x, self.weight1_ih_2 )
        i_i = torch.mm( x, self.I_weight_mul_weight1_ih_2 )
        # i_n = torch.mm( x, self.weight1_ih_3 ) + self.bias1_ih_3
        i_n = torch.mm( x, self.I_weight_mul_weight1_ih_3 ) + self.bias1_ih_3

        h_r = torch.mm( h1, self.weight1_hh_1 )
        h_i = torch.mm( h1, self.weight1_hh_2 )
        h_n = torch.mm( h1, self.weight1_hh_3 ) + self.bias1_hh_3
    
        resetgate = torch.sigmoid(i_r + h_r + self.bias11)
        inputgate = torch.sigmoid(i_i + h_i + self.bias12)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        h1 = newgate + inputgate * (h1 - newgate)
        #rnn1------------------------------

        x = x + h1
        inp = torch.cat([x, a2_t], dim=1)
        
        
        #rnn2--------------------------------------------------------
        # h2 = rnn2(inp, h2)
        
        # inp = inp + h2
        i_r = torch.mm( inp, self.weight2_ih_1 )
        i_i = torch.mm( inp, self.weight2_ih_2 )
        i_n = torch.mm( inp, self.weight2_ih_3 ) + self.bias2_ih_3

        h_r = torch.mm( h2, self.weight2_hh_1 )
        h_i = torch.mm( h2, self.weight2_hh_2 )
        h_n = torch.mm( h2, self.weight2_hh_3 ) + self.bias2_hh_3
        
        resetgate = torch.sigmoid(i_r + h_r + self.bias21)
        inputgate = torch.sigmoid(i_i + h_i + self.bias22)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        h2 = newgate + inputgate * (h2 - newgate)
        #rn2-------------------------------------------------------

        x = x + h2
        x = torch.cat([x, a3_t], dim=1)
        
        x = torch.mm(x, self.fc1_weight) + self.fc1_bias
        x = torch.relu(x)
        #fuse------------------------------------

        x = torch.cat([x, a4_t], dim=1)
        x = torch.mm(x, self.fc2_weight) + self.fc2_bias
        x = torch.relu(x)

        logits = torch.mm(x, self.fc3_weight) + self.fc3_bias
        logits = torch.relu(logits)

        return logits, h1, h2
