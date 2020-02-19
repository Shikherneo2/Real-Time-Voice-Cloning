import os
import librosa
import numpy as np
from tqdm import tqdm
from synthesizer import audio, hparams
from vocoder import audio as audio_vocoder
import vocoder.hparams as hp

from vocoder.utils import speech_utils

max_mel_frames=1250
rescaling_max = 0.9
sample_rate = 22050
utterance_min_duration = 1.6
mel_output_dir = "/home/sdevgupta/mine/data/mels"

read_metadata = open("/home/sdevgupta/all.csv", "r")

lines = [i.strip() for i in read_metadata.read().split("\n")]

read_metadata.close()

filenames = [ "/".join(i.split("|")[0].strip().split("/")[-2:])+".wav" for i in lines ]

filenames = filenames[:-1]

mel_basis_p = librosa.filters.mel(
          sr=sample_rate,
          n_fft=hp.n_fft,
          n_mels=80,
          htk=True,
          norm=None,
          fmax=None
      )

write_metadata = open("/home/sdevgupta/all_wavernn_vocoder.csv", "w")
iter = 0
for wav_filename in tqdm(filenames):
    full_wav_filename = os.path.join( "/home/sdevgupta/mine/data/wavs", wav_filename )

    iter+=1
    if os.path.exists( full_wav_filename.replace(".wav", ".npy") ) is False:
        continue
    
    else:
        # Get mel spectrogram
        full_wav_filename2 = full_wav_filename.replace("/home/sdevgupta/mine/data/wavs", "/home/sdevgupta/mine/Blizzard2013_Segmentation/segments")
        wav, _ = librosa.load( full_wav_filename2, sample_rate )
        mel = speech_utils.get_speech_features( 
								wav,
								sample_rate,
								80, 
								features_type="mel",
								n_fft=hp.n_fft,
								hop_length=hp.hop_length,
								mag_power=1,
								feature_normalize=False,
								mean=0.,
								std=1.,
								data_min=1e-5,
								mel_basis=mel_basis_p)
        mel_frames = mel.shape[0]
        if mel_frames > max_mel_frames:
            continue
        else:
            mel_filename = os.path.join( mel_output_dir, wav_filename.replace("/","_").replace(".npy", ".npy")  )
            np.save(mel_filename, mel.T)
            
            write_metadata.write( "|".join( [full_wav_filename, mel_filename] ) + "\n")


write_metadata.close()