import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.token_embedders.bert_encoder import BertEncoder
from modules.token_embedders.bert_encoder import BertLinear
from modules.span_extractors.cnn_span_extractor import CNNSpanExtractor
from modules.decoders.decoder import VanillaSoftmaxDecoder
from utils.nn_utils import batched_index_select, gelu, clone_weights

logger = logging.getLogger(__name__)


class JointREPretrainedModel(nn.Module):
    """This class contains entity typing, masked token prediction, entity mention permutation prediction, confused entity mention context rank loss, four pretrained tasks in total.
    """
    def __init__(self, cfg):
        """This function decides `JointREPretrainedModel` components

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
        """

        super().__init__()
        self.ent_output_size = cfg.ent_output_size
        self.context_output_size = cfg.context_output_size
        self.output_size = cfg.ent_mention_output_size
        self.ent_batch_size = cfg.ent_batch_size
        self.permutation_batch_size = cfg.permutation_batch_size
        self.permutation_samples_num = cfg.permutation_samples_num
        self.confused_batch_size = cfg.confused_batch_size
        self.confused_samples_num = cfg.confused_samples_num
        self.activation = gelu
        self.device = cfg.device

        self.bert_encoder = BertEncoder(bert_model_name=cfg.bert_model_name,
                                        trainable=cfg.fine_tune,
                                        output_size=cfg.bert_output_size,
                                        activation=self.activation)

        self.entity_span_extractor = CNNSpanExtractor(input_size=self.bert_encoder.get_output_dims(),
                                                      num_filters=cfg.entity_cnn_output_channels,
                                                      ngram_filter_sizes=cfg.entity_cnn_kernel_sizes,
                                                      dropout=cfg.dropout)

        if self.ent_output_size > 0:
            self.ent2hidden = BertLinear(input_size=self.entity_span_extractor.get_output_dims(),
                                         output_size=self.ent_output_size,
                                         activation=self.activation,
                                         dropout=cfg.dropout)
        else:
            self.ent_output_size = self.entity_span_extractor.get_output_dims()
            self.ent2hidden = lambda x: x

        self.context_span_extractor = CNNSpanExtractor(input_size=self.bert_encoder.get_output_dims(),
                                                       num_filters=cfg.context_cnn_output_channels,
                                                       ngram_filter_sizes=cfg.context_cnn_kernel_sizes,
                                                       dropout=cfg.dropout)

        if self.context_output_size > 0:
            self.context2hidden = BertLinear(input_size=self.context_span_extractor.get_output_dims(),
                                             output_size=self.context_output_size,
                                             activation=self.activation,
                                             dropout=cfg.dropout)
        else:
            self.context_output_size = self.context_span_extractor.get_output_dims()
            self.context2hidden = lambda x: x

        if self.output_size > 0:
            self.mlp = BertLinear(input_size=2 * self.ent_output_size + 3 * self.context_output_size,
                                  output_size=self.output_size,
                                  activation=self.activation,
                                  dropout=cfg.dropout)
        else:
            self.output_size = 2 * self.ent_output_size + 3 * self.context_output_size
            self.mlp = lambda x: x

        self.entity_pretrained_decoder = VanillaSoftmaxDecoder(hidden_size=self.ent_output_size, label_size=18)

        self.masked_token_mlp = BertLinear(input_size=self.bert_encoder.get_output_dims(),
                                           output_size=self.bert_encoder.get_output_dims(),
                                           activation=self.activation)

        self.token_vocab_size = self.bert_encoder.bert_model.embeddings.word_embeddings.weight.size()[0]
        self.masked_token_decoder = nn.Linear(self.bert_encoder.get_output_dims(), self.token_vocab_size, bias=False)
        self.masked_token_decoder.weight.data.normal_(mean=0.0, std=0.02)
        self.masked_token_decoder_bias = nn.Parameter(torch.zeros(self.token_vocab_size))

        clone_weights(self.masked_token_decoder, self.bert_encoder.bert_model.embeddings.word_embeddings)

        self.masked_token_loss = nn.CrossEntropyLoss()

        self.permutation_decoder = VanillaSoftmaxDecoder(hidden_size=self.output_size, label_size=120)

        self.confused_context_decoder = nn.Linear(self.output_size, 1)
        self.confused_context_decoder.weight.data.normal_(mean=0.0, std=0.02)
        self.confused_context_decoder.bias.data.zero_()

        self.entity_mention_index_tensor = torch.LongTensor([2, 0, 3, 1, 4])
        if self.device > -1:
            self.entity_mention_index_tensor = self.entity_mention_index_tensor.cuda(device=self.device,
                                                                                     non_blocking=True)

    def forward(self, batch_inputs, pretrain_task=''):
        """This function propagates forwardly

        Arguments:
            batch_inputs {dict} -- batch inputs

        Keyword Arguments:
            pretrain_task {str} -- pretraining task (default: {''})
        
        Returns:
            dict -- results
        """

        if pretrain_task == 'masked_entity_typing':
            return self.masked_entity_typing(batch_inputs)
        elif pretrain_task == 'masked_entity_token_prediction':
            return self.masked_entity_token_prediction(batch_inputs)
        elif pretrain_task == 'entity_mention_permutation':
            return self.permutation_prediction(batch_inputs)
        elif pretrain_task == 'confused_context':
            return self.confused_context_prediction(batch_inputs)

    def seq_decoder(self, seq_inputs, seq_mask=None, seq_labels=None):
        results = {}
        seq_outpus = self.masked_token_decoder(seq_inputs) + self.masked_token_decoder_bias
        seq_log_probs = F.log_softmax(seq_outpus, dim=2)
        seq_preds = seq_log_probs.argmax(dim=2)
        results['predict'] = seq_preds

        if seq_labels is not None:
            if seq_mask is not None:
                active_loss = seq_mask.view(-1) == 1
                active_outputs = seq_outpus.view(-1, self.token_vocab_size)[active_loss]
                active_labels = seq_labels.view(-1)[active_loss]
                no_pad_avg_loss = self.masked_token_loss(active_outputs, active_labels)
                results['loss'] = no_pad_avg_loss
            else:
                avg_loss = self.masked_token_loss(seq_outpus.view(-1, self.token_vocab_size), seq_labels.view(-1))
                results['loss'] = avg_loss

        return results

    def masked_entity_typing(self, batch_inputs):
        """This function pretrains masked entity typing task.
        
        Arguments:
            batch_inputs {dict} -- batch inputs
        """

        seq_wordpiece_tokens_reprs, _ = self.bert_encoder(batch_inputs['tokens_id'])
        batch_inputs['seq_wordpiece_tokens_reprs'] = seq_wordpiece_tokens_reprs
        batch_inputs['seq_tokens_reprs'] = batched_index_select(seq_wordpiece_tokens_reprs,
                                                                batch_inputs['tokens_index'])

        all_ents = []
        all_ents_labels = []
        all_seq_tokens_reprs = []
        for ent_spans, ent_labels, seq_tokens_reprs in zip(batch_inputs['ent_spans'], batch_inputs['ent_labels'],
                                                           batch_inputs['seq_tokens_reprs']):
            all_ents.extend([span[0], span[1] - 1] for span in ent_spans)
            all_ents_labels.extend(ent_label for ent_label in ent_labels)
            all_seq_tokens_reprs.extend(seq_tokens_reprs for _ in range(len(ent_spans)))

        if self.ent_batch_size > 0:
            all_entity_typing_loss = []
            for idx in range(0, len(all_ents), self.ent_batch_size):
                batch_ents_tensor = torch.LongTensor(all_ents[idx:idx + self.ent_batch_size]).unsqueeze(1)
                if self.device > -1:
                    batch_ents_tensor = batch_ents_tensor.cuda(device=self.device, non_blocking=True)

                batch_seq_tokens_reprs = torch.stack(all_seq_tokens_reprs[idx:idx + self.ent_batch_size])

                batch_ents_feature = self.ent2hidden(
                    self.entity_span_extractor(batch_seq_tokens_reprs, batch_ents_tensor).squeeze(1))

                batch_ents_labels = torch.LongTensor(all_ents_labels[idx:idx + self.ent_batch_size])
                if self.device > -1:
                    batch_ents_labels = batch_ents_labels.cuda(device=self.device, non_blocking=True)

                entity_typing_outputs = self.entity_pretrained_decoder(batch_ents_feature, batch_ents_labels)

                all_entity_typing_loss.append(entity_typing_outputs['loss'])

            if len(all_entity_typing_loss) != 0:
                entity_typing_loss = sum(all_entity_typing_loss) / len(all_entity_typing_loss)
            else:
                zero_loss = torch.Tensor([0])
                zero_loss.requires_grad = True
                if self.device > -1:
                    zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)
                entity_typing_loss = zero_loss
        else:
            all_ents_tensor = torch.LongTensor(all_ents).unsqueeze(1)
            if self.device > -1:
                all_ents_tensor = all_ents_tensor.cuda(device=self.device, non_blocking=True)
            all_seq_tokens_reprs = torch.stack(all_seq_tokens_reprs)
            all_ents_feature = self.entity_span_extractor(all_seq_tokens_reprs, all_ents_tensor).squeeze(1)

            all_ents_feature = self.ent2hidden(all_ents_feature)

            all_ents_labels = torch.LongTensor(all_ents_labels)
            if self.device > -1:
                all_ents_labels = all_ents_labels.cuda(device=self.device, non_blocking=True)

            entity_typing_outputs = self.entity_pretrained_decoder(all_ents_feature, all_ents_labels)

            entity_typing_loss = entity_typing_outputs['loss']

        outputs = {}
        outputs['loss'] = entity_typing_loss

        return outputs

    def masked_entity_token_prediction(self, batch_inputs):
        """This function pretrains masked entity tokens prediction task.
        
        Arguments:
            batch_inputs {dict} -- batch inputs
        """

        masked_seq_wordpiece_tokens_reprs, _ = self.bert_encoder(batch_inputs['tokens_id'])
        masked_seq_wordpiece_tokens_reprs = self.masked_token_mlp(masked_seq_wordpiece_tokens_reprs)

        if batch_inputs['masked_index'].sum() != 0:
            masked_entity_token_outputs = self.seq_decoder(seq_inputs=masked_seq_wordpiece_tokens_reprs,
                                                           seq_mask=batch_inputs['masked_index'],
                                                           seq_labels=batch_inputs['tokens_label'])
            masked_entity_token_loss = masked_entity_token_outputs['loss']
        else:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.device > -1:
                zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)
            masked_entity_token_loss = zero_loss

        outputs = {}
        outputs['loss'] = masked_entity_token_loss
        return outputs

    def permutation_prediction(self, batch_inputs):
        """This function pretrains entity mention permutaiton prediction task.
        
        Arguments:
            batch_inputs {dict} -- batch inputs
        """

        all_permutation_feature = self.get_entity_mention_feature(batch_inputs['tokens_id'],
                                                                  batch_inputs['tokens_index'],
                                                                  batch_inputs['ent_mention'],
                                                                  batch_inputs['tokens_index_lens'])
        permutation_outputs = self.permutation_decoder(all_permutation_feature, batch_inputs['ent_mention_label'])
        permutation_loss = permutation_outputs['loss']

        outputs = {}
        outputs['loss'] = permutation_loss
        return outputs

    def confused_context_prediction(self, batch_inputs):
        """This function pretrains confused context prediction task.
        
        Arguments:
            batch_inputs {dict} -- batch inputs
        """

        all_confused_context_feature = self.get_entity_mention_feature(batch_inputs['confused_tokens_id'],
                                                                       batch_inputs['confused_tokens_index'],
                                                                       batch_inputs['confused_ent_mention'],
                                                                       batch_inputs['confused_tokens_index_lens'])
        all_truth_context_feature = self.get_entity_mention_feature(batch_inputs['origin_tokens_id'],
                                                                    batch_inputs['origin_tokens_index'],
                                                                    batch_inputs['origin_ent_mention'],
                                                                    batch_inputs['origin_tokens_index_lens'])
        confused_context_score = self.confused_context_decoder(all_confused_context_feature)
        truth_context_score = self.confused_context_decoder(all_truth_context_feature)
        rank_loss = torch.mean(torch.relu(5.0 - torch.abs(confused_context_score - truth_context_score)))

        outputs = {}
        outputs['loss'] = rank_loss
        return outputs

    def get_entity_mention_feature(self, batch_wordpiece_tokens, batch_wordpiece_tokens_index, batch_entity_mentions,
                                   batch_seq_lens):
        """This function extracts entity mention feature using CNN.
        
        Arguments:
            batch_wordpiece_tokens {tensor} -- batch wordpiece tokens
            batch_wordpiece_tokens_index {tensor} -- batch wordpiece tokens index
            batch_entity_mentions {list} -- batch entity mentions
            batch_seq_lens {list} -- batch sequence length list
        
        Returns:
            tensor -- entity mention feature
        """

        batch_seq_reprs, _ = self.bert_encoder(batch_wordpiece_tokens)
        batch_seq_reprs = batched_index_select(batch_seq_reprs, batch_wordpiece_tokens_index)

        entity_spans = []
        context_spans = []
        for entity_mention, seq_len in zip(batch_entity_mentions, batch_seq_lens):
            entity_spans.append([[entity_mention[0][0], entity_mention[0][1]],
                                 [entity_mention[1][0], entity_mention[1][1]]])
            context_spans.append([[0, entity_mention[0][0]], [entity_mention[0][1], entity_mention[1][0]],
                                  [entity_mention[1][1], seq_len]])

        entity_spans_tensor = torch.LongTensor(entity_spans)
        if self.device > -1:
            entity_spans_tensor = entity_spans_tensor.cuda(device=self.device, non_blocking=True)

        context_spans_tensor = torch.LongTensor(context_spans)
        if self.device > -1:
            context_spans_tensor = context_spans_tensor.cuda(device=self.device, non_blocking=True)

        entity_feature = self.entity_span_extractor(batch_seq_reprs, entity_spans_tensor)
        context_feature = self.context_span_extractor(batch_seq_reprs, context_spans_tensor)

        entity_feature = self.ent2hidden(entity_feature)
        context_feature = self.context2hidden(context_feature)

        entity_mention_feature = torch.cat([
            context_feature[:, 0, :], entity_feature[:, 0, :], context_feature[:, 1, :], entity_feature[:, 1, :],
            context_feature[:, 2, :]
        ],
                                           dim=-1).view(len(batch_wordpiece_tokens), -1)

        entity_mention_feature = self.mlp(entity_mention_feature)

        return entity_mention_feature
