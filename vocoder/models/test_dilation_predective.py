import torch
import torch.nn as nn
import numpy as np

class SampleConditioningNetwork16( nn.Module ):
    def __init__( self, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( 4 ):
            self.layers.append( nn.Conv1d( indims, outdims, kernel_size=2, bias=False, dilation=2**i ) )
    
    def forward( self, x ):
        for i in range(15):
            a = x
            for j in self.layers:
                a = j(a)
            x = torch.cat([x[:,:,1:], a], dim=-1)
        return x


class SampleConditioningNetwork8( nn.Module ):
    def __init__( self, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( 3 ):
            self.layers.append( nn.Conv1d( indims, outdims, kernel_size=2, bias=False, dilation=2**i ) )
    
    def forward( self, x ):
        for i in range(7):
            a = x
            for j in self.layers:
                a = j(a)
            x = torch.cat([x[:,:,1:], a], dim=-1)
            # print(x)
        return x

if __name__ == "__main__":
    # net = SampleConditioningNetwork16( 1, 1 )
    # inputs = torch.from_numpy( np.random.randn( 1, 1, 16 ) )

    net = SampleConditioningNetwork8( 1, 1 )
    inputs = torch.from_numpy( np.random.randn( 5, 1, 8 ) )
    
    inputs = inputs.type(torch.float32)
    outputs = net( inputs )
    print( inputs )
    # print( outputs )
    print(outputs.shape)