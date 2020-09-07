import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from vocoder.distribution import sample_from_discretized_mix_logistic
from vocoder.display import *
from vocoder.audio import *
#from sru import SRU

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@torch.jit.script
def sample_from_discretized_mix_logistic( y ):
  """
  Sample from discretized mixture of logistic distributions
  Args:
      y (Tensor): B x C x T
      log_scale_min (float): Log scale minimum value
  Returns:
      Tensor: sample in range of [-1, 1].
  """
  nr_mix = y.size(1) // 3

  # B x T x C
  y = y.transpose(1, 2)
  logit_probs = y[:, :, :nr_mix]

  # sample mixture indicator from softmax
  temp = torch.zeros(logit_probs.size(), dtype=torch.float32).uniform_(1e-5, 1.0 - 1e-5)
  if logit_probs.is_cuda:
    temp = temp.cuda()
  temp = logit_probs.data - torch.log(- torch.log(temp))
  _, argmax = temp.max(dim=-1)

  # (B, T) -> (B, T, nr_mix)
  one_hot = torch.zeros(argmax.size() + (nr_mix,), dtype=torch.float32)
  if argmax.is_cuda:
      one_hot = one_hot.cuda()
  one_hot.scatter_(len(argmax.size()), argmax.unsqueeze(-1), 1)
  
  # select logistic parameters
  means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
  
  vv = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
  log_scales = torch.clamp(vv, min=-32.23619130191664)
  
  # sample from logistic & clip to interval
  # we don't actually round to the nearest 8bit value when sampling
  u = torch.zeros( means.size(), dtype=means.data.dtype ).uniform_(1e-5, 1.0 - 1e-5).cuda()
  x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

  x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

  return x

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
        for f in self.layers: 
            x = f(x)
        x = self.conv_out(x)
        return x

# Repeats the features scales times in the specified direction
class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    #feat_dims = num_mels=80
    #compute_dims=128
    #res_out_dims=128
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: 
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='MOL'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW' :
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL' :
            self.n_classes = 30
        else :
            RuntimeError("Unknown model mode value - ", self.mode)

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()
    

    def forward(self, x, mels):
        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        h2 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


    def generate(self, mels, batched, target, overlap, mu_law, progress_callback=None):
        mu_law = mu_law if self.mode == 'RAW' else False
        if progress_callback is not None:
            progress_callback = self.gen_display

        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            mels = mels.cuda()
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()
            x = torch.zeros(b_size, 1).cuda()

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    b = logits.unsqueeze(0).transpose(1, 2)
                    sample = sample_from_discretized_mix_logistic(b)
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    x = sample.transpose(0, 1).cuda()

                elif self.mode == 'RAW' :
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)
        
                if progress_callback is not None:
                    if i % 100 == 0:
                        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                        progress_callback(i, seq_len, b_size, gen_rate)
                    

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)
        
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        #if hp.apply_preemphasis:
        #    output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        # fade_out = np.linspace(1, 0, 5 * self.hop_length)
        # output = output[:wave_len]
        # output[-5 * self.hop_length:] *= fade_out
        
        self.train()

        return output

    def to_one_hot( self, tensor, n, fill_with=1. ):
      # we perform one hot encore with respect to the last axis
      one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
      if tensor.is_cuda:
          one_hot = one_hot.cuda()
      one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
      return one_hot

    @torch.no_grad()
    def procs(self, x, i, aux_splits, mels):
      
      m_t = mels[:, i, :]

      a1_t = aux_splits[0][:, i, :]
      a2_t = aux_splits[1][:, i, :]
      a3_t = aux_splits[2][:, i, :]
      a4_t = aux_splits[3][:, i, :]

      x = torch.cat([x, m_t, a1_t], dim=1)
      x = self.I(x)
      h1 = rnn1(x, h1)

      x = x + h1
      inp = torch.cat([x, a2_t], dim=1)
      h2 = rnn2(inp, h2)

      x = x + h2
      x = torch.cat([x, a3_t], dim=1)
      x = F.relu(self.fc1(x))

      x = torch.cat([x, a4_t], dim=1)
      x = F.relu(self.fc2(x))

      logits = self.fc3(x)

      b = logits.unsqueeze(0).transpose(1, 2)
      
      assert b.size(1) % 3 == 0
      sample = sample_from_discretized_mix_logistic( b )
      output = sample.view(-1)
      
      x = sample.transpose(0, 1).cuda()

      return output
    

    def generate_from_mel(self, mels, batched, overlap, target, mu_law=False, cpu=False, apply_preemphasis=False,  progress_callback=None):
       # mels = torch.from_numpy( np.array([ mel ]) )
        mu_law = mu_law if self.mode == 'RAW' else False
        progress_callback = progress_callback or self.gen_display

        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            if cpu is False:
                mels = mels.cuda()
            mel_len = mels.size(-1)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, cpu=cpu, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))
            #print(mels.dtype)
            #print(aux.dtype)
            if batched:
                mels = self.fold_with_overlap(mels, target, overlap, cpu)
                aux = self.fold_with_overlap(aux, target, overlap, cpu)

            inds = []
            pieces = 16
            piece_size = int( 256/pieces )
            #for i in range(pieces):
            #    inds.append([k for j in range( 0, mel_len) for k in range((j*256) +(i*piece_size), (j*256) +(i*piece_size)+piece_size)])
            
            #for i in range(10):
            #    inds.append( [(i*2560)+j for j in range(2560)]  )
            inds = [ [] for i in range(16) ]
            for i in range(mels.size(1)):
                inds[ i%16 ].append(i)

            mels = torch.squeeze(mels[:,inds,:], 0)
            aux = torch.squeeze(aux[:,inds,:], 0)
            print(mels.shape)
            print(aux.shape)

            b_size, seq_len, _ = mels.size()

            if cpu is False:
                h1 = torch.zeros(b_size, self.rnn_dims, dtype=mels.dtype).cuda()
                h2 = torch.zeros(b_size, self.rnn_dims, dtype=mels.dtype).cuda()
                x = torch.zeros(b_size, 1, dtype=mels.dtype).cuda()
            else:
                h1 = torch.zeros(b_size, self.rnn_dims, dtype=mels.dtype)
                h2 = torch.zeros(b_size, self.rnn_dims, dtype=mels.dtype)
                x = torch.zeros(b_size, 1, dtype=mels.dtype)

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):
                m_t = mels[:, i, :]
                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)
                x = torch.cat([x, m_t, a1_t], dim=1)
                
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                b = logits.unsqueeze(0).transpose(1, 2)
                sample = sample_from_discretized_mix_logistic( b )
                output.append(sample.view(-1))
                # x = torch.FloatTensor([[sample]]).cuda()
                if cpu is False:
                    x = sample.transpose(0, 1).cuda()
                else:
                    x = sample.transpose(0, 1)

                
                #if (i+1)%piece_size==0:
                    # tile these to batch_size
                x = x[-1].repeat(pieces,1)
                #x = torch.zeros(pieces, 1).cuda()
                h1 = h1[-1].repeat(pieces,1)
                h2 = h2[-1].repeat(pieces,1)
                if i % 100 == 0:
                    gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                    progress_callback(i, seq_len, b_size, gen_rate)

        
        output = torch.stack(output)
        #print( "\n" )
        #print( output.size() )
        #output = output.flatten()
        nn = []
        print("\n")
        #print(seq_len)
        #print(pieces)
        #print(len(output))
        #for j2 in range(0, seq_len, piece_size):
        #    for i2 in range(pieces):
        #        nn.extend(output[i2][j2:j2+piece_size])
        #print(len(nn))    
        #output = np.array(nn, dtype=np.float64)
        if cpu is False:
            output = output.cpu().numpy().flatten().astype(np.float64)
        else:
            output = output.numpy().flatten().astype(np.float64)
        
        # if batched:
            # output = self.xfade_and_unfold(output, target, overlap)
        # else:
            # output = output[0]

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        if apply_preemphasis:
            output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        #fade_out = np.linspace(1, 0, 10 * self.hop_length)
        #output = output[:wave_len]
        #output[-10 * self.hop_length:] *= fade_out
        
        self.train()
        return output


    def gen_display(self, i, seq_len, b_size, gen_rate):
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

  
    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, cpu=False, side='both'):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        if cpu is False:
            padded = torch.zeros(b, total, c, dtype=x.dtype).cuda()
        else:
            padded = torch.zeros(b, total, c)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
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
            x = self.pad_tensor(x, padding, cpu=cpu, side='after')

        if cpu is False:
            folded = torch.zeros(num_folds, target + 2 * overlap, features, dtype=x.dtype).cuda()
        else:
            folded = torch.zeros(num_folds, target + 2 * overlap, features, dtype=x.dtype)

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
            self.load_state_dict(checkpoint["model_state"])
        else:
            # Backwards compatibility
            self.load_state_dict(checkpoint["model"])

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


