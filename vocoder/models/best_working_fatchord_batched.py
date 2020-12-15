import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from vocoder.distribution import sample_from_discretized_mix_logistic
from vocoder.display import *
from vocoder.audio import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x


class Stretch4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, 1, 1, 4)
        return x.view(b, c, h , w * 4)


class Stretch16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, 1, 1, 16)
        return x.view(b, c, h , w * 16)

class Stretch256(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, 1, 1, 256)
        return x.view(b, c, h , w * 256)


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super( UpsampleNetwork, self ).__init__()
        self.total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * self.total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch256()
        self.up_layers = nn.ModuleList()
        
        scale = 4
        k_size = (1, scale * 2 + 1)
        padding = (0, scale)
        stretch = Stretch4()
        conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
        conv.weight.data.fill_(1. / k_size[1])
        self.up_layers.append(stretch)
        self.up_layers.append(conv)

        scale = 4
        k_size = (1, scale * 2 + 1)
        padding = (0, scale)
        stretch = Stretch4()
        conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
        conv.weight.data.fill_(1. / k_size[1])
        self.up_layers.append(stretch)
        self.up_layers.append(conv)

        scale = 16
        k_size = (1, scale * 2 + 1)
        padding = (0, scale)
        stretch = Stretch16()
        conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
        conv.weight.data.fill_(1. / k_size[1])
        self.up_layers.append(stretch)
        self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, 512:-512]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad: int, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate):
        super( WaveRNN, self).__init__()
        self.pad = pad
        self.n_classes = 30
        self.training = False
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        
        self.I = torch.jit.trace( nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims), example_inputs=torch.rand(1, 2, feat_dims+self.aux_dims+1) )
        self.fc1 = torch.jit.trace( nn.Linear(rnn_dims + self.aux_dims, fc_dims), example_inputs=torch.rand(1, 2, rnn_dims+self.aux_dims) )
        self.fc2 = torch.jit.trace( nn.Linear(fc_dims + self.aux_dims, fc_dims), example_inputs=torch.rand(1, 2, fc_dims+self.aux_dims) )
        self.fc3 = torch.jit.trace( nn.Linear(fc_dims, self.n_classes), example_inputs=torch.rand(1, 2, fc_dims) )
        
        # self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        # self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        # self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        # self.fc3 = nn.Linear(fc_dims, self.n_classes)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        
        self.gru_cell1 = torch.nn.GRUCell(self.rnn1.input_size, self.rnn1.hidden_size, bias=True)
        self.gru_cell1.weight_hh.data = self.rnn1.weight_hh_l0.data
        self.gru_cell1.weight_ih.data = self.rnn1.weight_ih_l0.data
        self.gru_cell1.bias_hh.data = self.rnn1.bias_hh_l0.data
        self.gru_cell1.bias_ih.data = self.rnn1.bias_ih_l0.data
        # self.gru_cell1 = torch.jit.trace( self.gru_cell1, example_inputs=(torch.rand(1, self.rnn1.input_size), torch.zeros(1, rnn_dims))  )

        self.gru_cell2 = torch.nn.GRUCell(self.rnn2.input_size, self.rnn2.hidden_size)
        self.gru_cell2.weight_hh.data = self.rnn2.weight_hh_l0.data
        self.gru_cell2.weight_ih.data = self.rnn2.weight_ih_l0.data
        self.gru_cell2.bias_hh.data = self.rnn2.bias_hh_l0.data
        self.gru_cell2.bias_ih.data = self.rnn2.bias_ih_l0.data
        # self.gru_cell2 = torch.jit.trace( self.gru_cell2, example_inputs=(torch.rand(1, self.rnn2.input_size), torch.zeros(1, rnn_dims))  )
        self.rnn1 = None
        self.rnn2 = None
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

				# No grad does not work with PyTorch JIT
        for param in self.parameters():
          if param.requires_grad:
            param.requires_grad = False

    def forward( self, mels, batched, overlap, target ):
        outputs = []
        mels = mels.cuda()
        # wave_len = (mels.size(-1) - 1) * self.hop_length
        mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad)
        mels, aux = self.upsample(mels.transpose(1, 2))

        if batched:
            mels = self.fold_with_overlap(mels, target, overlap, cpu)
            aux = self.fold_with_overlap(aux, target, overlap, cpu)

        b_size, seq_len, _ = mels.size()
        nr_mix = self.n_classes//3
        temp = torch.log(- torch.log( torch.zeros((b_size, 1, nr_mix), dtype=torch.float32).uniform_(1e-5, 1.0 - 1e-5).cuda() ))
        one_hot = torch.zeros( (b_size,1, nr_mix), dtype=torch.float32 ).cuda()
        u = torch.zeros( (b_size, 1), dtype=torch.float32 ).uniform_(1e-5, 1.0 - 1e-5).cuda()
        u = (torch.log(u) - torch.log(1. - u))

        h1 = torch.zeros(b_size, self.rnn_dims).cuda()
        h2 = torch.zeros(b_size, self.rnn_dims).cuda()
        x = torch.zeros(b_size, 1).cuda()
        d = self.aux_dims
        aux_splits = []
        aux_splits.append( aux[:, :, :d])
        aux_splits.append( aux[:, :, d:d * 2])
        aux_splits.append( aux[:, :, d * 2:d * 3])
        aux_splits.append( aux[:, :, d * 3:d * 4])
        
        bb = torch.cat([ mels, aux_splits[0] ], dim=-1)
        for i in range(seq_len):
          
          #a1_t = aux_splits[0][:, i, :]
          # DO SOMETHING HERE?
          a2_t = aux_splits[1][:, i, :]
          a3_t = aux_splits[2][:, i, :]
          a4_t = aux_splits[3][:, i, :]
        
          #x = torch.cat([x, m_t, a1_t], dim=1)
          x = torch.cat( [x, bb[:,i,:]], dim=-1 )

          x = self.I(x)
          h1 = self.gru_cell1(x, h1)

          x = x + h1
          inp = torch.cat([x, a2_t], dim=1)
          h2 = self.gru_cell2(inp, h2)

          x = x + h2
          x = torch.cat([x, a3_t], dim=1)
          x = torch.nn.functional.relu(self.fc1(x))

          x = torch.cat([x, a4_t], dim=1)
          x = torch.nn.functional.relu(self.fc2(x))

          logits = self.fc3(x)
          
          #needed?
          b = logits.unsqueeze(0)
          sample = self.sample_from_discretized_mix_logistic( b, temp, one_hot, u )
          
          #avoid
          #output = sample.view(-1)
          
          #avoid?
          x = sample.transpose(0, 1).cuda()
          outputs.append(sample)

        wavs = torch.stack(outputs).transpose(0, 1)

        output = wavs[0]
        # output = self.xfade_and_unfold(wavs, target, overlap)

        # # Fade-out at the end to avoid signal cutting out suddenly
        # fade_out = np.linspace(1, 0, 10 * self.hop_length)
        # output = output[:wave_len]
        # output[-10 * self.hop_length:] *= fade_out

        return output


    @torch.jit.export
    def sample_from_discretized_mix_logistic( self, y, temp_const, one_hot, u ):
        nr_mix = 10
        # B x T x C
        logit_probs = y[:, :, :nr_mix]

        # sample mixture indicator from softmax
        temp = logit_probs.data - temp_const
        _, argmax = temp.max(dim=-1)

        one_hot.scatter_(len(argmax.size()), argmax.unsqueeze(-1), 1)

        # select logistic parameters
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        
        log_scales = torch.clamp( torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=-32.23619130191664)
        
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        # u = torch.zeros( means.size(), dtype=means.data.dtype ).uniform_(1e-5, 1.0 - 1e-5).cuda()
        x = means + torch.exp(log_scales) * u

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        return x


    def pad_tensor_after(self, x, pad: int):
        b, t, c = x.size()
        total = t + pad
        padded = torch.zeros(b, total, c).cuda()
        padded[:, :t, :] = x
        
        return padded

    def pad_tensor(self, x, pad: int):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad
        padded = torch.zeros(b, total, c).cuda()
        padded[:, pad:pad + t, :] = x
        
        return padded

    def fold_with_overlap(self, x, target, overlap, cpu=False):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor_after(x, padding)

        if cpu is False:
            folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()
        else:
            folded = torch.zeros(num_folds, target + 2 * overlap, features)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)
        linear = np.ones((silence_len), dtype=np.float64)
        
        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([linear, fade_out])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self) :
        return self.step.data.item()

    def checkpoint(self, model_dir, optimizer) :
        k_steps = self.get_step() // 1000
        self.save(model_dir.joinpath("checkpoint_%dk_steps.pt" % k_steps), optimizer)

    def log(self, path, msg) :
        with open(path, 'a') as f:
            print(msg, file=f)

    def load_for_infer( self, path ) :
        checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
        print(checkpoint.keys())
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            # Backwards compatibility
            self.load_state_dict(checkpoint["model"], strict=False)

    def load(self, path, optimizer) :
        checkpoint = torch.load(path)
        if "optimizer_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            # Backwards compatibility
            self.load_state_dict(checkpoint)

    def save(self, path, optimizer) :
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out :
            print('Trainable Parameters: %.3fM' % parameters)

