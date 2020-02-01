import numpy as np
import os
from tqdm import tqdm
from synthesizer import audio, hparams

dir = "/home/sdevgupta/mine/Real-Time-Voice-Cloning/vocoder/mels"
ref_level_db = 20
files = [ os.path.join(dir, i) for i in os.listdir(dir) if i[-3:]=="npy"]

for file in tqdm(files):
	a = np.load(file)
	a = a*10
	a = a - ref_level_db
	a = audio._normalize(a, hparams.hparams)/4
	np.save( file, a )