import os
import librosa
import numpy as np
from tqdm import tqdm
from vocoder.utils import speech_utils

max_mel_frames=900
rescaling_max = 0.9
sample_rate = 16000
utterance_min_duration = 1.6
mel_output_dir = "/home/sdevgupta/mine/Real-Time-Voice-Cloning/vocoder/mels"

read_metadata = open("/home/sdevgupta/all.csv", "r")

lines = [i.strip() for i in read_metadata.read().split("\n")]

read_metadata.close()

filenames = [ "/".join(i.split("|")[0].strip().split("/")[-2:])+".wav" for i in lines ]

filenames = filenames[:-1]

mel_basis_p = librosa.filters.mel(
          sr=sample_rate,
          n_fft=800,
          n_mels=80,
          htk=True,
          norm=True,
          fmax=None
      )

write_metadata = open("/home/sdevgupta/all_wavernn_vocoder.csv", "w")
iter = 0

maxs = 0
mins = 1

for wav_filename in tqdm(filenames):
    full_wav_filename = os.path.join( "/home/sdevgupta/mine/Blizzard2013_Segmentation/segments", wav_filename )
    wav, _ = librosa.load( full_wav_filename, sample_rate )

    iter+=1
    if(iter==1000):
        break
    if len(wav) < utterance_min_duration * sample_rate:
        continue
    else:
        # Get mel spectrogram
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
        a = np.max(mel)
        b = np.min(mel)
        if a>maxs:
            maxs=a
        if b<mins:
            mins=b
        # mel_frames = mel.shape[0]
        # if mel_frames > max_mel_frames:
        #     continue
        # else:
        #     mel_filename = os.path.join( mel_output_dir, "mel-"+str(iter)+".npy" )
        #     np.save(mel_filename, mel)
        #     write_metadata.write( "|".join( [full_wav_filename, mel_filename] ) + "\n")
print(mins, maxs)