import os
import sys
import time
import torch
import numpy as np
from librosa.output import write_wav
#from models.fatchord_version_subscale1 import WaveRNN
from models.best_working_fatchord_batched import WaveRNN
#from apex import amp
import hparams as hp

use_cpu = False
sampling_rate = 22050
path = "/home/sdevgupta/wavernn_testing_data/models/catheryn_big_mol_checkpoint_1089k_steps.pt"
# path = "/home/sdevgupta/mine/Real-Time-Voice-Cloning/experiments/fourth_run_gta/checkpoint_1004k_steps.pt"

mel_dir = sys.argv[1]
output_dir = sys.argv[2]
files = os.listdir( mel_dir )
files = [ os.path.join(mel_dir,i) for i in files if i[-3:]=="npy" if i.find("LJ")==-1]

# @torch.jit.script
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
).cuda()
model = torch.jit.script( model )

model.load_for_infer( path )
#model, _ = amp.initialize(model, [], opt_level="O3")

for i in range(files):
    print(files[i])
    mel = np.load(files[i]).T
#     mel = np.load(files[i]).T[:,:200]
    
    mel = torch.from_numpy( np.array([ mel ]) )
    # wav = model.generate_from_mel( mel, batched=False, overlap=100, target=5000, mu_law=True, cpu=use_cpu, apply_preemphasis=False )
    start = time.time()
#     wav = model.generate_from_mel( mel, batched=False, overlap=100, target=5000 )
#     wav = model(mel)
    wav = model( mel, batched=True, overlap=800, target=8000 )
    final = time.time()
    seq_len = mel.shape[-1]*256

    print( "Total time : "+ str(final-start) + ", "+ str( (final-start)*1000 )+"milliseconds" )
    print( "Rate : "+str( seq_len/(final-start) )+"Hz" )
    #wav_path = os.path.join( output_dir, os.path.basename(files[i]).replace(".npy","_990K.wav") )
    #write_wav( wav_path, wav.astype(np.float32), sr=sampling_rate )
