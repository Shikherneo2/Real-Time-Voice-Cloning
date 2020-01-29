
import os
import librosa
import numpy as np
from tqdm import tqdm
from vocoder.utils import speech_utils

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
          norm=True,
          fmax=None
      )

for filename in tqdm(filenames):
    full_wav_filename = filename[0]
    wav, _ = librosa.load( full_wav_filename, sr=sample_rate )
    mel = speech_utils.get_speech_features( 
								wav,
								sample_rate,
								80, 
								features_type="mel",
								n_fft=800,
								hop_length=200,
								mag_power=2,
								feature_normalize=False,
								mean=0.,
								std=1.,
								data_min=1e-5,
								mel_basis=mel_basis_p)
    np.save(filename[1], mel/15)