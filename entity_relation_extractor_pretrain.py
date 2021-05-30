from collections import defaultdict
import os
import random
import logging
import json

import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from models.pretrain_models.joint_relation_extraction_pretrained_model import JointREPretrainedModel
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def step(model, batch_inputs, device):
    fields = [
        'tokens_id', 'tokens_index', 'masked_index', 'tokens_label', 'ent_mention_label', 'confused_tokens_id',
        'confused_tokens_index', 'origin_tokens_id', 'origin_tokens_index'
    ]

    for field in fields:
        if field in batch_inputs:
            batch_inputs[field] = torch.LongTensor(batch_inputs[field])
            if device > -1:
                batch_inputs[field] = batch_inputs[field].cuda(device=device, non_blocking=True)

    outputs = model(batch_inputs, batch_inputs['pretrain_task'])
    return outputs['loss']


def train(cfg, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(), param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [param for name, param in parameters if not any(item in name for item in no_decay)],
        'weight_decay_rate':
        cfg.adam_weight_decay_rate
    }, {
        'params': [param for name, param in parameters if any(item in name for item in no_decay)],
        'weight_decay_rate':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = 320000
    num_warmup_steps = 32000
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)

    step_cnt = 0
    model.zero_grad()

    tasks = ['masked_entity_token_prediction', 'masked_entity_typing', 'entity_mention_permutation', 'confused_context']

    data_file_path = {
        'masked_entity_typing': 'data/pretrain/wikipedia_masked_entity_typing_instance.json',
        'masked_entity_token_prediction': 'data/pretrain/wikipedia_masked_entity_token_prediction_instance.json',
        'entity_mention_permutation': 'data/pretrain/wikipedia_entity_mention_permutation_instance.json',
        'confused_context': 'data/pretrain/wikipedia_confused_context_instance.json'
    }

    data_file = {
        'masked_entity_typing': open(data_file_path['masked_entity_typing'], 'r'),
        'masked_entity_token_prediction': open(data_file_path['masked_entity_token_prediction'], 'r'),
        'entity_mention_permutation': open(data_file_path['entity_mention_permutation'], 'r'),
        'confused_context': open(data_file_path['confused_context'], 'r')
    }

    batch_size = {
        'masked_entity_typing': 9,
        'masked_entity_token_prediction': 9,
        'entity_mention_permutation': 9,
        'confused_context': 5
    }

    fields = {
        'masked_entity_typing': {
            'tokens_id': 'tokens_id',
            'tokens_index': 'tokens_index',
            'ent_spans': 'spans',
            'ent_labels': 'labels'
        },
        'masked_entity_token_prediction': {
            'tokens_id': 'tokens_id',
            'masked_index': 'masked_index',
            'tokens_label': 'tokens_label'
        },
        'entity_mention_permutation': {
            'tokens_id': 'tokens_id',
            'tokens_index': 'tokens_index',
            'ent_mention': 'spans',
            'ent_mention_label': 'labels'
        },
        'confused_context': {
            'confused_tokens_id': 'confused_tokens_id',
            'confused_tokens_index': 'confused_tokens_index',
            'confused_ent_mention': 'confused_spans',
            'origin_tokens_id': 'origin_tokens_id',
            'origin_tokens_index': 'origin_tokens_index',
            'origin_ent_mention': 'origin_spans'
        }
    }

    no_pad_namespace = [
        'ent_spans', 'ent_labels', 'ent_mention', 'ent_mention_label', 'confused_ent_mention', 'origin_ent_mention'
    ]

    model.train()

    while True:
        step_cnt += 1

        for task in tasks:
            batch = defaultdict(list)
            batch['pretrain_task'] = task
            for _ in range(batch_size[task]):
                try:
                    line = next(data_file[task])
                except StopIteration:
                    data_file[task].close()
                    data_file[task] = open(data_file_path[task], 'r')
                    line = next(data_file[task])
                sent = json.loads(line.strip())
                for field, raw_field in fields[task].items():
                    batch[field].append(sent[raw_field])
            for field in fields[task]:
                if field not in no_pad_namespace:
                    lens = [len(item) for item in batch[field]]
                    batch[field + '_lens'] = lens
                    max_len = max(lens)
                    for i in range(len(lens)):
                        batch[field][i].extend([0] * (max_len - lens[i]))

            loss = step(model, batch, cfg.device)

            if step_cnt % 16 == 0:
                logger.info("Step: {} {}: {}".format(step_cnt // 16, task, loss.item()))

            loss = loss * batch_size[task] / 512
            loss.backward()

        if step_cnt % 16 == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if step_cnt % 16000 == 0:
            torch.save(model.state_dict(), open(cfg.pretrained_model_path + '_' + str(step_cnt // 16000) + 'k', "wb"))

        if step_cnt == 320000:
            torch.save(model.state_dict(), open(cfg.pretrained_model_path, "wb"))
            logger.info("Pretraining Completed!")
            break


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    # joint model
    model = JointREPretrainedModel(cfg)

    # continue training
    if cfg.continue_training and os.path.exists(cfg.pretrained_model_path):
        state_dict = torch.load(open(cfg.pretrained_model_path, 'rb'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loading last training model {} successfully.".format(cfg.pretrained_model_path))

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    train(cfg, model)


if __name__ == '__main__':
    main()
