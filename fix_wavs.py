import os
from tqdm import tqdm
import numpy as np
import librosa
from tqdm import tqdm
from synthesizer import audio, hparams
from vocoder import audio as audio_vocoder
import vocoder.hparams as hp

from vocoder.utils import speech_utils

max_mel_frames=900
rescaling_max = 0.9
sample_rate = 16000
utterance_min_duration = 1.6

txt = open("/home/sdevgupta/all_wavernn_vocoder_new.csv","r").read().split("\n")
op = "/home/sdevgupta/mine/data/wavs"
filenames = [ [line.strip().split("|")[0].strip(),line.strip().split("|")[1].strip()] for line in txt if line.strip()!=""]

mel_basis_p = librosa.filters.mel(
          sr=sample_rate,
          n_fft=800,
          n_mels=80,
          htk=True,
          norm=True,
          fmax=None
      )

for wav_path in filenames:
        wav_full_path = wav_path[0].replace("/home/sdevgupta/mine/data/wavs", "/home/sdevgupta/mine/Blizzard2013_Segmentation/segments").replace("npy","wav")
        
        if( wav_full_path[-3:]!="wav" ):
            print( "error " + wav_full_path )
        wav, _ = librosa.load( wav_full_path, 16000 )
#        mel = speech_utils.get_speech_features( 
#								wav,
#								sample_rate,
#								80, 
#								features_type="mel",
#								n_fft=800,
#								hop_length=200,
#								mag_power=2,
#								feature_normalize=False,
#								mean=0.,
#								std=1.,
#								data_min=1e-5,
#								mel_basis=mel_basis_p)
        
#       mel = mel.T/15
#      mel = np.clip(mel, -1, 1)
#     np.save( wav_path[1], mel )
        mel = np.load(wav_path[1])
        rescaling_max=0.9
        wav = wav / np.abs(wav).max() * rescaling_max
        wav = np.clip(wav, -1, 1)
	
        r_pad =  (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        
        wav = wav[:mel.shape[1] * hp.hop_length]
        wav = audio_vocoder.encode_mu_law(wav, mu=2 ** hp.bits)
        
        if(os.path.exists(os.path.join( op, wav_path[0].split("/")[-2] )) is False):
            os.mkdir( os.path.join( op, wav_path[0].split("/")[-2] ) )
        
        np.save( wav_path[0], wav )
