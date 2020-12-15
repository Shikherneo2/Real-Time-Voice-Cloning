from synthesizer.hparams import hparams as _syn_hp


# Audio settings------------------------------------------------------------------------
# Match the values of the synthesizer
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

# bit depth of signal
bits = 11 #(16 in one model)
# Recommended to suppress noise if using raw bits in hp.voc_mode below
mu_law = True


# WAVERNN / VOCODER --------------------------------------------------------------------------------
# either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_mode = 'RAW'

# Needs to correctly factorise hop_length
voc_upsample_factors = (4, 8, 8)
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 70
voc_lr = 1e-5

# number of samples to generate at each checkpoint
voc_gen_at_checkpoint = 3
# this will pad the input so that the resnet can 'see' wider than input length
voc_pad = 2
# must be a multiple of hop_length
voc_seq_len = hop_length * 5
# (8 in one model) - probably only used in training

# Generating / Synthesizing
voc_gen_batched = True
voc_target = 8000
voc_overlap = 800