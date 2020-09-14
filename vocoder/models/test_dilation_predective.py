import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SampleConditioningNetwork64_16( nn.Module ):
    def __init__( self, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( 6 ):
            self.layers.append( nn.Conv1d( indims, outdims, kernel_size=2, bias=False, dilation=2**i ) )
    
    def forward( self, x ):
        # times = 16 if self.training else 15
        for j in self.layers:
            x = j(x)
            print(x.size())
        return x

if __name__ == "__main__":
    # net = SampleConditioningNetwork16( 1, 1 )
    inputs = torch.from_numpy( np.random.randn( 2, 1, 128 ) )
    print( inputs.shape )

    # net = SampleConditioningNetwork8( 1, 1 )
    net = SampleConditioningNetwork64_16( 1, 1 )
    # inputs = torch.from_numpy( inputs )
    inputs = F.pad(inputs, (63,0))
    inputs = inputs.type(torch.float32)
    outputs = net( inputs )
    # print( outputs )
    print(outputs.shape)