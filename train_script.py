import json
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from ray import tune


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
    train_config,
    hp_config,
):
    total_step = hp_config.epoch_num * (train_config.dataset_size // (
        hp_config.batch_size * hp_config.accumulate_step)
    )

    warm_up_step = int(total_step * hp_config.warm_up_step_rate)

    def warm_up_function(step):
        m = (step + 1) / \
            warm_up_step if step < warm_up_step else 1
        return m
    return warm_up_function


def save_config(train_config, hp_config, model_config, save_path):
    json.dump(train_config, open(os.path.join(
        save_path, 'train_config.json'), 'w'))
    json.dump(hp_config, open(os.path.join(
        save_path, 'hp_config.json'), 'w'))
    json.dump(model_config.to_dict(), open(os.path.join(
        save_path, 'model_config.json'), 'w'))


def train(
    hp_config,
    train_config=None,
    model_config=None,
    data_loader_creater=None,
    save_path=None,
    progress_bar=False,
):
    # tune.utils.wait_for_gpu()
    # Get save path.
    if not save_path:
        save_dir = tune.get_trial_dir()
    else:
        save_dir = save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    # Save experiment config.
    save_config(
        train_config=train_config,
        hp_config=hp_config,
        model_config=model_config,
        save_path=save_dir
    )
    train_config = Namespace(**train_config)
    hp_config = Namespace(**hp_config)

    # Set training on cuda or cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initial tensorboard writer.
    writer = SummaryWriter(save_dir)

    # Set random seed.
    random.seed(hp_config.seed)
    np.random.seed(hp_config.seed)
    torch.manual_seed(hp_config.seed)

    # Get data loader.
    data_loader = data_loader_creater(
        batch_size=hp_config.batch_size,
        tokenizer_name=train_config.tokenizer_name,
        max_length=train_config.max_length
    )

    # Initial model.
    model = GPT2LMHeadModel(model_config)
    model = model.to(device)

    # Initial optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp_config.lr)
    warm_up_function = create_warm_up_function(
        train_config=train_config,
        hp_config=hp_config
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_up_function)

    criterion = CrossEntropyLoss(ignore_index=train_config.padding_idx)
    iteration = 1
    total_loss = 0
    for epoch in range(hp_config.epoch_num):
        if progress_bar:
            epoch_iter = tqdm(
                data_loader,
                desc=f'epoch: {epoch}, loss: {0:.6f}'
            )
        for batch_inputs in data_loader:
            batch_inputs['input_ids'] = batch_inputs['input_ids'].to(device)
            batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to(
                device)
            outputs = model(**batch_inputs)

            # Calaulate loss.
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch_inputs['input_ids'][..., 1:].contiguous()
            # Flatten the tokens
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

            # Modify progress bar.
            if progress_bar:
                epoch_iter.set_description(
                    f'epoch: {epoch}, loss: {loss.item():.6f}'
                )
            loss.backward()

            if iteration % hp_config.accumulate_step == 0:
                # Update model.
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if iteration % train_config.log_step == 0:
                # Update tensorboard.
                avg_loss = total_loss / train_config.log_step
                writer.add_scalar('loss', avg_loss, iteration)
                tune.report(loss=avg_loss)
                total_loss = 0

            if iteration % train_config.save_ckpt_step == 0:
                save_model(
                    model=model,
                    optimizer=optimizer,
                    save_path=save_dir,
                    iteration=iteration
                )
            iteration += 1
    save_model(
        model=model,
        optimizer=optimizer,
        save_path=save_dir,
        iteration=iteration
    )
