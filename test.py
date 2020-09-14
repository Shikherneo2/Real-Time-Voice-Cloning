import os
import time
import argparse
import numpy as np
from torch import optim
from pathlib import Path
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.display import stream, simple_table
from vocoder.gen_wavernn import gen_testset
# , gen_meltest
from torch.utils.data import DataLoader
import torch.nn.functional as F
import vocoder.hparams as hp


# Instantiate the model
print("Initializing the model...")
model = WaveRNN(
    rnn_dims=hp.voc_rnn_dims,
    fc_dims=hp.voc_fc_dims,
    bits=hp.bits,
    pad=hp.voc_pad,
    upsample_factors=hp.voc_upsample_factors,
    feat_dims=hp.num_mels,
    compute_dims=hp.voc_compute_dims,
    res_out_dims=hp.voc_res_out_dims,
    res_blocks=hp.voc_res_blocks,
    hop_length=hp.hop_length,
    sample_rate=hp.sample_rate,
    mode=hp.voc_mode
).cuda()
    
# Load the weights
model_dir = models_dir.joinpath(run_id)
model_dir.mkdir(exist_ok=True)
weights_fpath = weights_path
metadata_fpath = metadata_path

print("\nLoading weights at %s" % weights_fpath)
model.load(weights_fpath, optimizer)
print("WaveRNN weights loaded from step %d" % model.step)

# Initialize the dataset

dataset = VocoderDataset(metadata_fpath)

test_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True)

gen_testset( model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, model_dir )