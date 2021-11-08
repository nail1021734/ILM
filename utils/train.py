import json
import os
import random
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from utils.data_processor import create_data_loader


def save_model(model, optimizer, save_path, iteration):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f'checkpoint-{iteration}.pt')
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(save_path, f'optimizer-{iteration}.pt')
    )


def create_warm_up_function(
    config: Dict,
):
    total_step = config.epoch_num * (config.dataset_size // (
        config.batch_size * config.accumulate_step)
    )

    warm_up_step = int(total_step * config.warm_up_step_rate)

    def warm_up_function(step):
        m = (step + 1) / \
            warm_up_step if step < warm_up_step else 1
        return m
    return warm_up_function


def save_config(
    config: Dict,
    model_config: Dict,
):
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    json.dump(config, open(
        os.path.join(config['save_path'], 'config.json'), 'w'))
    json.dump(model_config, open(
        os.path.join(config['save_path'], 'model_config.json'), 'w'))


def train(
    config: Dict,
    model_config: Dict
):
    r'''Use to training ILM model.

    Parameters
    ==========
    config: Dict
        Include 2 part.
        1. Training config.
        max_length: int,
        tokenizer_name: str,
        save_ckpt_step: int,
        log_step: int,
        exp_name: str,
        dataset_name: str,

        2. Hyperparameters.
        seed: int,
        lr: float,
        epoch_num: int,
        batch_size: int,
        accumulate_step: int,
        warm_up_step_rate: float,
        weight_decay: float,

    model_config: Dict
        Use to create `GPT2Config` object.
    '''
    # Save experiment config.
    save_config(
        config=config,
        model_config=model_config,
    )
    config = Namespace(**config)

    # Set training on cuda or cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initial tensorboard writer.
    writer = SummaryWriter(config.save_path)

    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Get data loader.
    data_loader = create_data_loader(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
    )

    # Initial model.
    model = GPT2LMHeadModel(
        GPT2Config.from_dict(model_config)
    )
    model = model.to(device)

    # Load from pretrained.
    if config.ckpt_path:
        model.load_state_dict(torch.load(config.ckpt_path))

    # Set bias and LayerNorm no weight dacay.
    no_decay = ['bias', 'ln']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': config.weight_decay,
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Initial optimizer.
    optimizer = torch.optim.AdamW(
        optim_group_params,
        lr=config.lr,
    )
    if config.optimizer_path:
        optimizer.load_state_dict(torch.load(config.optimizer_path))

    warm_up_function = create_warm_up_function(
        config=config,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_up_function)

    criterion = CrossEntropyLoss(ignore_index=config.padding_idx)
    iteration = 1
    total_loss = 0
    for epoch in range(config.epoch_num):
        epoch_iter = tqdm(
            data_loader,
            desc=f'epoch: {epoch}, loss: {0:.6f}'
        )

        for batch_inputs in epoch_iter:
            batch_inputs['input_ids'] = batch_inputs['input_ids'].to(device)
            batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to(
                device)
            outputs = model(
                input_ids=batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask']
            )

            # Calaulate loss.
            shift_logits = outputs.logits[..., :-1, :].contiguous()

            # Start from second word.
            mask = batch_inputs['answer_mask'][..., 1:].to(device)
            padding_id = torch.zeros_like(mask) + config.padding_idx
            padding_id = padding_id * (mask == False)
            padding_id = padding_id.to(device)
            shift_labels = batch_inputs['input_ids'][..., 1:].contiguous(
            ) * mask + padding_id

            # Flatten the tokens.
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            total_loss += loss.item()

            # Modify progress bar.
            epoch_iter.set_description(
                f'epoch: {epoch}, loss: {loss.item():.6f}'
            )
            loss.backward()

            if iteration % config.accumulate_step == 0:
                # Update model.
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if iteration % config.log_step == 0:
                # Update tensorboard.
                avg_loss = total_loss / config.log_step
                writer.add_scalar('loss', avg_loss, iteration)
                total_loss = 0

            if iteration % config.save_ckpt_step == 0:
                save_model(
                    model=model,
                    optimizer=optimizer,
                    save_path=config.save_path,
                    iteration=iteration
                )
            iteration += 1
    save_model(
        model=model,
        optimizer=optimizer,
        save_path=config.save_path,
        iteration=iteration
    )
