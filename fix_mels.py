
import os
import librosa
import numpy as np
from tqdm import tqdm
from vocoder.utils import speech_utils
from synthesizer import audio, hparams
import matplotlib.pyplot as plt

max_mel_frames=900
rescaling_max = 0.9
sample_rate = 16000
utterance_min_duration = 1.6

txt = open("/home/sdevgupta/all_wavernn_vocoder_new.csv","r").read().split("\n")

filenames = [ [line.strip().split("|")[0].strip(),line.strip().split("|")[1].strip()] for line in txt if line.strip()!=""]

mel_basis_p = librosa.filters.mel(
          sr=sample_rate,
          n_fft=800,
          n_mels=80,
          htk=True,
          norm=None,
          fmax=None
      )

iter = 0
ref_level_db = 20
for filename in tqdm(filenames):
	iter+=1
	full_wav_filename = filename[0]
	wav, _ = librosa.load( full_wav_filename.replace("/home/sdevgupta/mine/data/wavs", "/home/sdevgupta/mine/Blizzard2013_Segmentation/segments").replace(".npy",".wav"), sr=sample_rate )
	wav = wav / np.abs(wav).max() * 0.9

	mel = speech_utils.get_speech_features( 
								wav,
								sample_rate,
								80, 
								features_type="mel",
								n_fft=800,
								hop_length=200,
								mag_power=1,
								feature_normalize=False,
								mean=0.,
								std=1.,
								data_min=1e-5,
								mel_basis=mel_basis_p)
	print(np.min(mel), np.max(mel))
	mel = mel - ref_level_db
	print(np.min(mel), np.max(mel))
	mel = audio._normalize(mel, hparams.hparams)/4
	print(np.min(mel), np.max(mel))
	plt.imshow(mel.T)
	plt.show()
	# np.save(filename[1], mel/15)
	break