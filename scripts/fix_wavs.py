import os

import librosa
import numpy as np
from tqdm import tqdm

import vocoder.hparams as hp
from synthesizer import audio, hparams
from vocoder import audio as audio_vocoder
from vocoder.utils import speech_utils

rescaling_max = 0.9
sample_rate = 22050
utterance_min_duration = 1.6

dir = "/home/sdevgupta/mine/data/mels"
wav_dir = dir.replace("/mels","/wavs_mol")

filenames = [ i for i in os.listdir(dir) ]

# write_metadata = open("/home/sdevgupta/new_train_list.txt","w")

for mel_path in tqdm(filenames):
        wav_full_path = os.path.join( "/home/sdevgupta/mine/Blizzard2013_Segmentation/segments", mel_path[:len(mel_path)-mel_path[::-1].find("_")-1],mel_path[len(mel_path)-mel_path[::-1].find("_"):] )
        full_mel_path = os.path.join( dir, mel_path )
        
        if( wav_full_path[-3:]!="wav" ):
            print( "error " + wav_full_path )
        wav, _ = librosa.load( wav_full_path, sr=sample_rate )
  
        mel = np.load( full_mel_path )  
       	rescaling_max = 0.9
        wav = wav / np.abs(wav).max() * rescaling_max
        wav = np.clip(wav, -1, 1)
	
        r_pad =  (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        
        wav = wav[:mel.shape[1] * hp.hop_length]
        wav = audio_vocoder.float_2_label(wav, bits=16)
        # wav = audio_vocoder.encode_mu_law(wav, mu=2 ** hp.bits)
        
        # if(os.path.exists(os.path.join( op, wav_path[0].split("/")[-2] )) is False):
        #     os.mkdir( os.path.join( op, wav_path[0].split("/")[-2] ) )
        output_wav_path = os.path.join( wav_dir, mel_path.replace(".wav",".npy") )
        np.save( output_wav_path, wav )
        # write_metadata.write("|".join([ output_wav_path, full_mel_path ])+"\n")

# write_metadata.close()
