import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import vocoder.hparams as hp
from tensorboardX import SummaryWriter
from vocoder.display import simple_table, stream
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_meltest, gen_testset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder

from distributed import init_distributed, apply_gradient_allreduce

def train(num_gpus, rank, group_name, run_id: str, models_dir: Path, metadata_path:Path, weights_path:Path, ground_truth: bool, save_every: int, backup_every: int, force_restart: bool):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length
    
    seed=1234
    torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, "nccl", "tcp://localhost:54321")

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
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
       
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups: 
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    weights_fpath = weights_path
    metadata_fpath = metadata_path

    if force_restart:
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)
    
    # This might not work -- Mixed Precision
    if num_gpus>1:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # Initialize the dataset
    
    dataset = VocoderDataset(metadata_fpath)
    
	shuffle_param = False if num_gpus>1 else True
    if rank==0:
        test_loader = DataLoader(dataset,
                              batch_size=1,
                              shuffle=shuffle_param,
                              pin_memory=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    epoch_start = int( model.step*hp.voc_batch_size/dataset.get_number_of_samples() )
    epoch_end = 200
    
    log_path = os.path.join( models_dir, "logs" )
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    
    writer = SummaryWriter( log_path )
    print("Log path : " + log_path)

    print("Starting from epoch: "+str(epoch_start))
    
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    data_loader = DataLoader(dataset,
                                collate_fn=collate_vocoder,
                                batch_size=hp.voc_batch_size,
                                num_workers=2,
                                shuffle=shuffle_param,
                                train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
                                pin_memory=True)

    for epoch in range(epoch_start, epoch_start+epoch_end):
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(data_loader, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()
            
            # Forward pass
            y_hat = model(x, m)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)
            
            # Backward pass
            loss = loss_fu nc(y_hat, y)
            if num_gpus>1:
                reduced_loss = reduce_tensor( loss.data ).item()
            else:
                reduced_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += reduced_loss
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if rank==0 and backup_every != 0 and step % backup_every == 0 :
                model.checkpoint(model_dir, optimizer)
                
            # if save_every != 0 and step % save_every == 0 :
            #     model.save(weights_fpath, optimizer)
            
            if rank==0 and step%500 == 0:
                writer.add_scalar('Loss/train', avg_loss, round(step/1000,1))
                msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                    f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                    f"steps/s | Step: {k}k | "
                print(msg, flush=True)

            if rank==0 and step%15000 == 0:
                gen_testset( model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, model_dir )
                gen_meltest( model, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap,model_dir )
