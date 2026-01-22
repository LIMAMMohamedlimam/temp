# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.utils.data
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import SeqDataset, paired_collate_fn
from trainer import train


def main():
    parser = argparse.ArgumentParser(description='main_train.py')

    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=64)

    Complétter ici .. 
    .....
    parser.add_argument('-no_cuda', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    # ========= Loading Dataset ========= #
    data = torch.load(args.data, weights_only=False)
    args.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, args)

    args.src_vocab_size = training_data.dataset.src_vocab_size
    args.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # ========= Preparing Model ========= #
    if args.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(args)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    transformer = Transformer(
        Compléter ici .. 
        ....
        dropout=args.dropout).to(device)

    args_optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    train(compléter ici ... , args)


def prepare_dataloaders(???, args):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=??,
        batch_size=??,
        collate_fn=??,
        shuffle=??)

    valid_loader = torch.utils.data.DataLoader(
        SeqDataset(???),
        ???)
    return train_loader, ??


if __name__ == '__main__':
    main()
