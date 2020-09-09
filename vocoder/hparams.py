# Audio settings------------------------------------------------------------------------
sample_rate = 22050
n_fft = 1024
num_mels = 80
hop_length = 256
win_length = 1024
fmin = 55
min_level_db = -100
ref_level_db = 20
mel_max_abs_value = 4.
preemphasis = 0.97
apply_preemphasis = False

# bit depth of signal - only applicable if mode is RAW
bits = 11
# Recommended to suppress noise if using raw bits in hp.voc_mode below
mu_law = False

# WAVERNN / VOCODER --------------------------------------------------------------------------------
# either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_mode = 'MOL'

# Needs to correctly factorise hop_length
voc_upsample_factors = (4, 8, 8)
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 70
voc_lr = 1e-4

# number of samples to generate at each checkpoint
voc_gen_at_checkpoint = 3
# this will pad the input so that the resnet can 'see' wider than input length
voc_pad = 2
# must be a multiple of hop_length
voc_seq_len = hop_length * 5

# Generating / Synthesizing
voc_gen_batched = False
voc_target = 8000
voc_overlap = 800