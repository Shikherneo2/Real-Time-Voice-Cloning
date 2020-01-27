import os
import sys
import torch
import numpy as np
from librosa.output import write_wav
from models.fatchord_version import WaveRNN

sys.path.append("D:\Voice\Real-Time-Voice-Cloning")

import hparams as hp

use_cpu = True
sampling_rate = 16000
path = "D:\Voice\wavernn_models\checkpoints\checkpoint_490k_steps.pt"
# path = "C:\\Users\\hades\\Downloads\\best_model_16K.pth.tar"

mel_path = "D:\Voice\wavernn_models\mels\mels1\mel-1.npy"
# mel_path = "C:\\Users\\hades\\Downloads\\dat\\mel-5.npy"

mel = np.load(mel_path).T
mel = mel/hp.mel_max_abs_value

if use_cpu:
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
    )
else:
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

model.load_for_infer( path )
wav_path = mel_path.replace( os.path.basename(mel_path), os.path.basename(mel_path).replace("mel","wav").replace("npy", "wav"))
print("Saving to : "+wav_path)
wav = model.generate_from_mel( mel, batched=False, overlap=hp.voc_overlap, target=hp.voc_target, mu_law=True, cpu=use_cpu, apply_preemphasis=False )
wav = wav / np.abs(wav).max() * 0.9
write_wav( wav_path, wav.astype(np.float32), sr=sampling_rate)