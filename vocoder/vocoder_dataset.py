from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
import vocoder.hparams as hp
import numpy as np
import torch
import librosa

class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath):
        
        with open(metadata_fpath, "r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        wav_fpaths = [ x[0].strip() for x in metadata ]
        gta_fpaths = [ x[1].strip() for x in metadata ]
      
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        self.number_of_samples = len(wav_fpaths)        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def get_number_of_samples(self):
        return self.number_of_samples

    def __getitem__(self, index):  
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        # mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        mel = np.load(mel_path).astype(np.float32)
        wav = np.load(wav_path)
        
        wav = wav / np.abs(wav).max()

        # Fix for missing padding   # TODO: settle on whether this is any useful
        # r_pad =  (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        # wav = np.pad(wav, (0, r_pad), mode='constant')
    # assert len(wav) >= mel.shape[1] * hp.hop_length
        # wav = wav[:mel.shape[1] * hp.hop_length]
        # assert len(wav) % hp.hop_length == 0
        
        # Quantize the wav
        # if hp.voc_mode == 'RAW':
        #     if hp.mu_law:
        #         quant = audio.encode_mu_law(wav, mu=2 ** hp.bits)
        #     else:
        #         quant = audio.float_2_label(wav, bits=hp.bits)
        # elif hp.voc_mode == 'MOL':
            
        return mel, wav

    def __len__(self):
        return len(self.samples_fpaths)
        
        
def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]-64:sig_offsets[i] + hp.voc_seq_len ] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels)

    mels = torch.tensor(mels)

    quant = audio.float_2_label(labels, bits=16).astype(np.int64)
    labels = torch.tensor(quant).long()

    x = labels[:, :-16]
    y = labels[:, 64:]
    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = audio.label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL' :
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels
