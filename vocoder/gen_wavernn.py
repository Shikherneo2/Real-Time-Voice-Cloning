from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *
import vocoder.hparams as hp
from synthesizer import audio as audio_synth, hparams as hparams_synth


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law, progress_callback=None)
        save_wav(wav, save_str)

def gen_meltest( model: WaveRNN, batched, target, overlap, save_path ):
	mel = []
	mel.append( np.load("/home/sdevgupta/mine/waveglow/outputs/waveglow_specs/mel-1.npy").T )
	mel.append( np.load("/home/sdevgupta/mine/waveglow/outputs/waveglow_specs/mel-3.npy").T )
	mel.append( np.load("/home/sdevgupta/mine/waveglow/outputs/waveglow_specs/mel-5.npy").T )
	
	k = model.get_step() // 1000
	for i,m in enumerate(mel):
		m = m - 20
		m = audio_synth._normalize(m, hparams_synth.hparams)/4
		wav = model.generate_from_mel( m, batched=False, overlap=hp.voc_overlap, target=hp.voc_target, mu_law=True, cpu=False, apply_preemphasis=False )
		#wav = wav / np.abs(wav).max() * 0.9
		save_str = save_path.joinpath( "mel-"+str(i+1)+"-steps-"+str(k)+"k.wav" )
		save_wav(wav, save_str)

