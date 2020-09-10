import torch
import torch.nn as nn
import numpy as np

class SampleConditioningNetwork( nn.Module ):
    def __init__( self, num_layers, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( num_layers ):
            self.layers.append( nn.Conv1d( indims, outdims, padding=2, kernel_size=3, bias=False, dilation=2**i ) )
    def forward( self, x ):
        # print(x.shape)
        concat = x
        for f in self.layers: 
            x = f(x)
            # print(x.shape)
            concat = torch.cat( [concat,x], dim=-1 )
        return concat

if __name__ == "__main__":
    net = SampleConditioningNetwork( 3, 1, 1 )
    inputs = torch.from_numpy( np.random.randn( 8, 1, 4 ) )
    inputs = inputs.type(torch.float32)
    outputs = net( inputs )
    print(outputs.shape)
    # outputs = outputs.reshape(4, 2, -1).squeeze()
    # print(outputs.shape)
    # print(outputs.unsqueeze(1).shape)