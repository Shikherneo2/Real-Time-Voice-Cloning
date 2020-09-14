import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys

from vocoder.distribution import sample_from_discretized_mix_logistic
from vocoder.display import *
from vocoder.audio import *

class SampleConditioningNetwork8( nn.Module ):
    def __init__( self, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( 3 ):
            self.layers.append( nn.Conv1d( indims, outdims, kernel_size=2, bias=False, dilation=2**i ) )
    
    def forward( self, x ):
        times = 8 if self.training else 7
        for i in range(times):
            a = x
            for j in self.layers:
                a = j(a)
            x = torch.cat([x[:,:,1:], a], dim=-1)
        return x

class SampleConditioningNetwork( nn.Module ):
    def __init__( self, num_layers, indims, outdims ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range( num_layers ):
            self.layers.append( nn.Conv1d( indims, outdims, padding=2, kernel_size=3, bias=False, dilation=2**i ) )
    def forward( self, x ):
        concat = x
        for f in self.layers: 
            x = f(x)
            concat = torch.cat( [concat,x], dim=-1 )
        return concat


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
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.scale_factor = 8 #16
        self.pad = pad
        if self.mode == 'RAW' :
            self.n_classes = 2 ** bit
        elif self.mode == 'MOL' :
            self.n_classes = 30
        else :
            RuntimeError("Unknown model mode value - ", self.mode)

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.condition_net_size = 18 #66

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        # self.I = nn.Linear(feat_dims + self.aux_dims + self.condition_net_size, rnn_dims)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.pred_condition_net = SampleConditioningNetwork8(1,1)
        # self.condition_samples = SampleConditioningNetwork( 3, 1, 1 )
        self.real_samples_probability = 0.25
        self.num_params()

    def get_orig_mask(self, size):
        # batch_size, seq_len, 1
        first_col = torch.ones( size[0], 1, size[2] )
        others = torch.cuda.FloatTensor( size[0], size[1]-1, size[2] ).uniform_() < self.real_samples_probability
        return torch.cat([first_col.cuda(), others.float()], dim=1)

    def forward(self, x, mels):
        self.step += 1
        bsize = x.size(0)

        scaled_bsize = bsize*self.scale_factor
        h1 = torch.zeros(1, scaled_bsize, self.rnn_dims).cuda()
        h2 = torch.zeros(1, scaled_bsize, self.rnn_dims).cuda()
        mels, aux = self.upsample(mels)

        mels = mels.reshape( bsize, -1, self.scale_factor, mels.size(-1) )
        aux = aux.reshape( bsize, -1, self.scale_factor, aux.size(-1) )
        mels = mels.transpose(2,1)
        aux = aux.transpose(2,1)

        mels = mels.reshape( scaled_bsize, -1, mels.size(-1) )
        aux = aux.reshape( scaled_bsize, -1, aux.size(-1) )

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = x.reshape( bsize, -1, self.scale_factor )
        # 70, 80, 16
        _,b,_ = x.size()

        # 70*16, 80, 1
        orig_x = x.transpose(2,1).reshape( scaled_bsize, b ).unsqueeze(-1)
        orig_mask = self.get_orig_mask( orig_x.size() )

        x2 = x.reshape( bsize*b, -1 )
        x = self.pred_condition_net( x2.unsqueeze(1) ).squeeze(1)
        x = x.reshape( bsize, b, -1 ).transpose( 2,1 )
        x = x.reshape( scaled_bsize, b).unsqueeze(-1)
        
        siz = x.size()
        x = torch.cat( [torch.zeros(siz[0],1,siz[2] ,dtype=torch.float32), x[:,:-1,:]], dim=1)
        x = (orig_mask*orig_x) + ((1-orig_mask)*x)
        # x = self.condition_samples( x.reshape( bsize*b, -1 ).unsqueeze(1) )
        # x = x.reshape( bsize, b, -1 )


        # torch.Size([70, 80, 66])
        # x = x.unsqueeze(1).repeat( 1, self.scale_factor, 1, 1 )
        # x = x.reshape( scaled_bsize, b, -1 )

        # x = x.transpose(2,1)
        # 70, 16, 80, 80
        # torch.Size([1120, 80, 80])

        # 70,80,16 -> 70,16,80 ->1120,80 -> 1120,80,1
        # x = x.transpose( 2, 1 ).reshape( scaled_bsize, b ).unsqueeze(-1)
        
        x = torch.cat([x, mels, a1], dim=2)
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
        x = self.fc3(x)
        
        po = x.size(-1)
        
        # 1120, 80, 30
        # 70, 16, 80, 30 -> 70, 80, 16, 30 -> 70, 1280, 30
        x = x.reshape(bsize, self.scale_factor, -1, 30).transpose(2,1).reshape(bsize, -1, po)

        return x


    def generate(self, mels, batched, target, overlap, mu_law, progress_callback=True):
        samples_in_seq = 8
        mu_law = mu_law if self.mode == 'RAW' else False
        if progress_callback is not None:
            progress_callback = self.gen_display

        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        start = time.time()
        with torch.no_grad():
            mels = mels.cuda()
            original_bsize = mels.size(0)
            scaled_bsize = original_bsize*self.scale_factor

            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            mels = mels.reshape( original_bsize, -1, self.scale_factor, mels.size(-1) )
            aux = aux.reshape( original_bsize, -1, self.scale_factor, aux.size(-1) )
            mels = mels.transpose(2,1)
            aux = aux.transpose(2,1)

            mels = mels.reshape( scaled_bsize, -1, mels.size(-1) )
            aux = aux.reshape( scaled_bsize, -1, aux.size(-1) )

            # if batched:
            #     mels = self.fold_with_overlap(mels, target, overlap)
            #     aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()
            # x = torch.zeros(b_size, self.condition_net_size).cuda()
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

                # Sample from discretized mixed logistic
                
                # torch.Size([16, 30])
                sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                # sample - torch.Size([1, 16])
                output.append(sample.view(-1))

                x = self.pred_condition_net( sample.unsqueeze(0) ).squeeze(1).t()

                # x = sample.t().cuda()
                # x = sample.transpose(0, 1).cuda()

                # (1,1,16)
                # x = sample.unsqueeze(0).cuda()
                # x = self.condition_samples( x ).squeeze(0)
                # x = x.repeat(b_size, 1)
                if progress_callback is not None:
                    if i % 100 == 0:
                        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                        progress_callback(i, seq_len, b_size, gen_rate)
        
        end = time.time()
        output = torch.stack(output).squeeze()
        print(output.shape)
        # po = output.size(-1)
        # bsize = output.size(0)
        
        # #  (7120, 1, 16)

        # output = output.reshape(bsize, self.scale_factor, -1).transpose(2,1).reshape(bsize, -1, po)
        output = torch.flatten( output )
        print( str(round(output.size(0)/1000/(end-start), 3))+" KHz" )
        output = output.cpu().numpy()
        output = output.astype(np.float32)
        
        self.train()

        return output

    def generate_from_mel(self, mel, batched, overlap, target, mu_law=False, cpu=False, apply_preemphasis=False,  progress_callback=None):
        mels = torch.from_numpy( np.array([ mel ]) )
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
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, cpu=cpu, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap, cpu)
                aux = self.fold_with_overlap(aux, target, overlap, cpu)

            b_size, seq_len, _ = mels.size()

            if cpu is False:
                h1 = torch.zeros(b_size, self.rnn_dims).cuda()
                h2 = torch.zeros(b_size, self.rnn_dims).cuda()
                x = torch.zeros(b_size, 1).cuda()
            else:
                h1 = torch.zeros(b_size, self.rnn_dims)
                h2 = torch.zeros(b_size, self.rnn_dims)
                x = torch.zeros(b_size, 1)

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
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    if cpu is False:
                        x = sample.transpose(0, 1).cuda()
                    else:
                        x = sample.transpose(0, 1)

                elif self.mode == 'RAW' :
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                if i % 100 == 0:
                    gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                    progress_callback(i, seq_len, b_size, gen_rate)

        output = torch.stack(output).transpose(0, 1)
        if cpu is False:
            output = output.cpu().numpy().astype(np.float64)
        else:
            output = output.numpy().astype(np.float64)
        
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        if apply_preemphasis:
            output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        
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
            padded = torch.zeros(b, total, c).cuda()
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

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

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