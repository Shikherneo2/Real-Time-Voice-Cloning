import os
import librosa
import numpy as np
from vocoder import audio

def encode_decode( wav, bits=9 ):
	enc = audio.encode_mu_law( wav, mu=2 ** bits )
	return audio.decode_mu_law( enc, mu=2**bits, from_labels = True )

dir = "/home/sdevgupta/mine/other_version/tests"

wav = librosa.core.load( "/home/sdevgupta/mine/other_version/tests/01-000013.wav", sr=22050 )[0]
librosa.output.write_wav( path=os.path.join( dir, "original.wav") , y=wav, sr=22050 )
librosa.output.write_wav( path=os.path.join( dir, "9_bits.wav" ), y=encode_decode(wav, 9), sr=22050 )
librosa.output.write_wav( path=os.path.join( dir, "10_bits.wav") , y=encode_decode(wav, 10), sr=22050 )
librosa.output.write_wav( path=os.path.join( dir, "11_bits.wav") , y=encode_decode(wav, 11), sr=22050 )
librosa.output.write_wav( path=os.path.join( dir, "12_bits.wav") , y=encode_decode(wav, 12), sr=22050 )
librosa.output.write_wav( path=os.path.join( dir, "13_bits.wav") , y=encode_decode(wav, 13), sr=22050 )