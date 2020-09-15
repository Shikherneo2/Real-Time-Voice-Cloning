from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
import torch
import scipy.io.wavfile
import numpy as np
from librosa.output import write_wav

_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
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
        mode=hp.voc_mode
    ).cuda()
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=False,  batched=False, target=8000, overlap=800, progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value

    mel = torch.from_numpy( mel )
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    return wav

mel = np.load("/home/sdevgupta/mine/OpenSeq2Seq/ljspeech_catheryn_logs/combined_mels/black_beauty_17-000053.npy")

load_model("/home/sdevgupta/mine/Real-Time-Voice-Cloning/experiments/run2_autoregressive_context/checkpoint_105k_steps.pt")
wav_raw = infer_waveform( np.array([mel] ))

wav = wav_raw.astype(np.int16)
scipy.io.wavfile.write( "/home/sdevgupta/Desktop/tests_wavernn_wav_datatype/test_int16.wav", 22050, wav)

wav = wav_raw
scipy.io.wavfile.write( "/home/sdevgupta/Desktop/tests_wavernn_wav_datatype/test_float32.wav", 22050, wav)

wav = wav_raw.astype(np.float32)
write_wav( "/home/sdevgupta/Desktop/tests_wavernn_wav_datatype/test_librosa.wav", wav, sr=22050  )
