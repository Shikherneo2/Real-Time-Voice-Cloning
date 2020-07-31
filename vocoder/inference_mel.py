import os
import sys
import time
import torch
import numpy as np
from librosa.output import write_wav
from models.fatchord_version import WaveRNN

import hparams as hp

use_cpu = False
sampling_rate = 22050
path = "/home/sdevgupta/mine/Real-Time-Voice-Cloning/experiments/fifth_run_new_gta_mol/checkpoint_990k_steps.pt"
# path = "/home/sdevgupta/mine/Real-Time-Voice-Cloning/experiments/fourth_run_gta/checkpoint_1004k_steps.pt"

mel_dir = sys.argv[1]
output_dir = sys.argv[2]
files = os.listdir( mel_dir )
files = [sorted([ os.path.join(mel_dir,i) for i in files if i[-3:]=="npy"])[3]]

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

for file in files:
	mel = np.load(file).T
	#best performing -- 11000,550
	print(torch.backends.cudnn.enabled)
	mel = torch.from_numpy( np.array([ mel ]) )
	# wav = model.generate_from_mel( mel, batched=False, overlap=100, target=5000, mu_law=True, cpu=use_cpu, apply_preemphasis=False )
	start = time.time()
	wav = model( mel )
	final = time.time()
	seq_len = mel.shape[-1]
	
	print( "Total time : "+ str(final-start) + ", "+ str( (final-start)*1000 )+"milliseconds" )
	print( "Rate : "+str( seq_len/(final-start) )+"Hz" )
	wav_path = os.path.join( output_dir, os.path.basename(file).replace(".npy","_990K.wav") )
	write_wav( wav_path, wav.astype(np.float32), sr=sampling_rate )