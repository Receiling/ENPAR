import json
import os
from concurrent import futures
from collections import defaultdict
import json
import random
import itertools

import spacy
import fire
from transformers import BertTokenizer

from utils.entity_chunking import get_entity_span


def entity_tagger(source_file, target_file):
    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_md")
    print("Load `en_core_web_md` model successfully.")

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    print("Load bert tokenizer successfully.")

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file, 'w', encoding='utf-8') as fout:
        sentId = 0
        for line in fin:
            line = line.strip()
            if len(line) <= 0:
                continue
            sent = nlp(line)
            tokens = list(map(str, sent))

            ent_labels = []
            for token in sent:
                if token.ent_iob_ == 'O':
                    ent_labels.append(str(token.ent_iob_))
                else:
                    ent_labels.append(str(token.ent_iob_) + '-' + str(token.ent_type_))

            span2ent = get_entity_span(ent_labels)

            if len(span2ent) < 1:
                continue

            wordpiece_tokens = ['[CLS]']
            wordpiece_tokens_index = []
            cur_pos = 1
            for token in tokens:
                tokenized_token = list(bert_tokenizer.tokenize(token))
                wordpiece_tokens.extend(tokenized_token)
                wordpiece_tokens_index.append([cur_pos, cur_pos + len(tokenized_token)])
                cur_pos += len(tokenized_token)
            wordpiece_tokens.append('[SEP]')

            ents = []
            for span, ent_label in span2ent.items():
                wordpiece_text = ' '.join(
                    wordpiece_tokens[wordpiece_tokens_index[span[0]][0]:wordpiece_tokens_index[span[-1] - 1][1]])
                ents.append({'wordpieceText': wordpiece_text, 'offset': span, 'label': ent_label})
            sent = json.dumps({
                'sentText': ' '.join(tokens),
                'wordpieceSentText': ' '.join(wordpiece_tokens),
                'wordpieceTokensIndex': wordpiece_tokens_index,
                'entityMentions': ents,
            })
            print(sent, file=fout)

            sentId += 1
            if sentId % 1000 == 0:
                print("processing {} sentences of file {}.".format(sentId, source_file))

    print("File {} is preprocessed completely, processing all {} sentences.".format(source_file, sentId))


def record_entity2tokens(source_file, entity2tokens_file):
    entity2tokens = defaultdict(list)
    entity_num_stat = defaultdict(int)

    with open(source_file, 'r') as fin:
        cnt = 0
        for line in fin:
            sent = json.loads(line)
            sentence_wordpiece_tokens = sent['wordpieceSentText'].split(' ')
            for ent in sent['entityMentions']:
                entity_wordpiece_tokens = sentence_wordpiece_tokens[sent['wordpieceTokensIndex'][
                    ent['offset'][0]][0]:sent['wordpieceTokensIndex'][ent['offset'][1] - 1][1]]
                entity2tokens[len(entity_wordpiece_tokens)].append(' '.join(entity_wordpiece_tokens))
            cnt += 1

    json.dump(entity2tokens, open(entity2tokens_file, 'w'))
    print("Processed {} sentences.".format(cnt))
    print('Save entity2tokens success.')


def get_random_replace_entity_tokens(raw_text, entity2tokens):
    label = str(len(raw_text.split(' ')))
    replace_entity_idx = random.randint(0, len(entity2tokens[label]) - 1)
    cnt = 0
    while True:
        random_replace_entity_text = entity2tokens[label][replace_entity_idx]
        if random_replace_entity_text != raw_text:
            return random_replace_entity_text.split(' ')
        cnt += 1
        if cnt > 10:
            return raw_text.split(' ')


def permutate_tokens(permutation, entity_mention_parts, wordpiece_tokens):
    permutated_wordpiece_tokens = ['[CLS]']
    permutated_wordpiece_tokens_index = []
    token_cnt = 0
    permutated_ent_spans = []
    for idx in permutation:
        if len(entity_mention_parts[idx]) == 0:
            continue
        if idx in [1, 3]:
            permutated_ent_spans.append((token_cnt, token_cnt + len(entity_mention_parts[idx])))
        for span in entity_mention_parts[idx]:
            st = len(permutated_wordpiece_tokens)
            permutated_wordpiece_tokens.extend(wordpiece_tokens[span[0]:span[-1]])
            permutated_wordpiece_tokens_index.append([st, len(permutated_wordpiece_tokens)])
        token_cnt += len(entity_mention_parts[idx])
    permutated_wordpiece_tokens.append('[SEP]')
    return permutated_wordpiece_tokens, permutated_wordpiece_tokens_index, permutated_ent_spans


def left_shift_entity_tokens(wordpiece_tokens, wordpiece_tokens_index, ent_span, left_shift):
    shifted_wordpiece_tokens = wordpiece_tokens[:wordpiece_tokens_index[ent_span[0] - left_shift][0]]
    shifted_wordpiece_tokens_index = wordpiece_tokens_index[:ent_span[0] - left_shift]
    shifted_wordpiece_tokens.extend(
        wordpiece_tokens[wordpiece_tokens_index[ent_span[0]][0]:wordpiece_tokens_index[ent_span[-1] - 1][1]])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[0]:ent_span[-1]])
    shifted_wordpiece_tokens.extend(
        wordpiece_tokens[wordpiece_tokens_index[ent_span[0] - left_shift][0]:wordpiece_tokens_index[ent_span[0]][0]])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[0] - left_shift:ent_span[0]])
    shifted_wordpiece_tokens.extend(wordpiece_tokens[wordpiece_tokens_index[ent_span[-1] - 1][1]:])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[-1]:])
    return shifted_wordpiece_tokens, shifted_wordpiece_tokens_index


def right_shift_entity_tokens(wordpiece_tokens, wordpiece_tokens_index, ent_span, right_shift):
    shifted_wordpiece_tokens = wordpiece_tokens[:wordpiece_tokens_index[ent_span[0]][0]]
    shifted_wordpiece_tokens_index = wordpiece_tokens_index[:ent_span[0]]
    shifted_wordpiece_tokens.extend(
        wordpiece_tokens[wordpiece_tokens_index[ent_span[-1] - 1][1]:wordpiece_tokens_index[ent_span[-1] + right_shift -
                                                                                            1][1]])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[-1]:ent_span[-1] + right_shift])
    shifted_wordpiece_tokens.extend(
        wordpiece_tokens[wordpiece_tokens_index[ent_span[0]][0]:wordpiece_tokens_index[ent_span[-1] - 1][1]])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[0]:ent_span[-1]])
    shifted_wordpiece_tokens.extend(wordpiece_tokens[wordpiece_tokens_index[ent_span[-1] + right_shift - 1][1]:])
    shifted_wordpiece_tokens_index.extend(wordpiece_tokens_index[ent_span[-1] + right_shift:])
    return shifted_wordpiece_tokens, shifted_wordpiece_tokens_index


def get_masked_entity_typing_instance(source_file, target_file, entity2tokens_file, vocab_file, entity_labels_file):
    with open(source_file, 'r') as fin, open(target_file, 'w') as fout, open(vocab_file, 'r') as vocab_fin, open(
            entity_labels_file, 'r') as entity_label_fin, open(entity2tokens_file, 'r') as entity2tokens_fin:
        entity2tokens = json.load(entity2tokens_fin)
        entity_labels = json.load(entity_label_fin)
        vocab = json.load(vocab_fin)
        processed_cnt = 0
        word_cnt = 0
        entity_cnt = 0
        for line in fin:
            sent = json.loads(line)
            num_entity = len(sent['entityMentions'])
            entity_cnt += num_entity
            word_cnt += len(sent['sentText'].split(' '))

            if num_entity < 2:
                continue

            wordpiece_tokens = sent['wordpieceSentText'].split(' ')
            if len(wordpiece_tokens) > 256:
                continue

            wordpiece_tokens_index = sent['wordpieceTokensIndex']
            tokens_id = [vocab[token] for token in wordpiece_tokens]
            spans = []
            lables = []

            for entity in sent['entityMentions']:
                wordpiece_tokens_range = (wordpiece_tokens_index[entity['offset'][0]][0],
                                          wordpiece_tokens_index[entity['offset'][1] - 1][1])
                spans.append(entity['offset'])
                lables.append(entity_labels[entity['label']])

                rd = random.random()
                if rd < 0.15:
                    for i in range(wordpiece_tokens_range[0], wordpiece_tokens_range[1]):
                        tokens_id[i] = vocab['[MASK]']
                elif rd < 0.2:
                    replace_entity_tokens = get_random_replace_entity_tokens(entity['wordpieceText'], entity2tokens)

                    j = 0
                    for i in range(wordpiece_tokens_range[0], wordpiece_tokens_range[1]):
                        tokens_id[i] = vocab.get(replace_entity_tokens[j], vocab['[UNK]'])
                        j += 1

            instance = {}
            instance['tokens_id'] = tokens_id
            instance['tokens_index'] = [
                wordpiece_tokens_span[0] for wordpiece_tokens_span in sent['wordpieceTokensIndex']
            ]
            instance['spans'] = spans
            instance['labels'] = lables

            instance = json.dumps(instance)
            print(instance, file=fout)

            processed_cnt += 1
            if processed_cnt % 10000 == 0:
                print("Processed {} sentences.".format(processed_cnt))

    print("Completed, processed {} sentences overall. word cnt {}, entity cnt {}".format(
        processed_cnt, word_cnt, entity_cnt))


def get_masked_entity_token_prediction_instance(source_file, target_file, entity2tokens_file, vocab_file):
    with open(source_file, 'r') as fin, open(target_file,
                                             'w') as fout, open(vocab_file,
                                                                'r') as vocab_fin, open(entity2tokens_file,
                                                                                        'r') as entity2tokens_fin:
        entity2tokens = json.load(entity2tokens_fin)
        vocab = json.load(vocab_fin)
        processed_cnt = 0
        for line in fin:
            sent = json.loads(line)
            num_entity = len(sent['entityMentions'])
            if num_entity < 2:
                continue

            wordpiece_tokens = sent['wordpieceSentText'].split(' ')
            if len(wordpiece_tokens) > 256:
                continue

            tokens_id = [vocab[token] for token in wordpiece_tokens]
            wordpiece_tokens_index = sent['wordpieceTokensIndex']
            masked_index = [0] * len(tokens_id)

            max_masked_subwords = 20

            masked_subwords_prob = 0.15

            num_mask_subwords = min(max_masked_subwords, max(1,
                                                             int(round(masked_subwords_prob * len(wordpiece_tokens)))))

            entity_idx = list(range(num_entity))
            random.shuffle(entity_idx)
            subwords_cnt = 0

            for idx in entity_idx:
                if subwords_cnt >= num_mask_subwords:
                    break
                ent = sent['entityMentions'][idx]
                if subwords_cnt + len(ent['wordpieceText'].split(' ')) > num_mask_subwords:
                    continue

                wordpiece_tokens_range = (wordpiece_tokens_index[ent['offset'][0]][0],
                                          wordpiece_tokens_index[ent['offset'][1] - 1][1])

                subwords_cnt += (wordpiece_tokens_range[1] - wordpiece_tokens_range[0])

                for i in range(wordpiece_tokens_range[0], wordpiece_tokens_range[1]):
                    masked_index[i] = 1

                if random.random() < 0.8:
                    for i in range(wordpiece_tokens_range[0], wordpiece_tokens_range[1]):
                        tokens_id[i] = vocab['[MASK]']
                else:
                    if random.random() < 0.5:
                        replace_entity_tokens = get_random_replace_entity_tokens(ent['wordpieceText'], entity2tokens)

                        j = 0
                        for i in range(wordpiece_tokens_range[0], wordpiece_tokens_range[1]):
                            tokens_id[i] = vocab.get(replace_entity_tokens[j], vocab['[UNK]'])
                            j += 1

            instance = {}
            instance['tokens_id'] = tokens_id
            instance['masked_index'] = masked_index
            instance['tokens_label'] = [vocab[token] for token in wordpiece_tokens]

            instance = json.dumps(instance)
            print(instance, file=fout)

            processed_cnt += 1
            if processed_cnt % 10000 == 0:
                print("Processed {} sentences.".format(processed_cnt))


def get_entity_mention_permutation_instance(source_file, target_file, vocab_file):
    permutation_labels = list(itertools.permutations([0, 1, 2, 3, 4], 5))
    permutations = list(range(1, 120))
    processed_cnt = 0

    with open(source_file, 'r') as fin, open(target_file, 'w') as fout, open(vocab_file, 'r') as vocab_fin:
        vocab = json.load(vocab_fin)
        for line in fin:
            sent = json.loads(line)
            num_entity = len(sent['entityMentions'])
            if num_entity < 2:
                continue

            wordpiece_tokens = sent['wordpieceSentText'].split(' ')
            if len(wordpiece_tokens) > 256:
                continue

            wordpiece_tokens_index = sent['wordpieceTokensIndex']

            entity_idx = list(range(num_entity))
            random.shuffle(entity_idx)

            for i in range(min(num_entity, 5)):
                for j in range(i + 1, min(num_entity, 5)):
                    ent1_idx = min(entity_idx[i], entity_idx[j])
                    ent2_idx = max(entity_idx[i], entity_idx[j])

                    ent1_span = sent['entityMentions'][ent1_idx]['offset']
                    ent2_span = sent['entityMentions'][ent2_idx]['offset']

                    entity_mention_parts = [
                        wordpiece_tokens_index[:ent1_span[0]], wordpiece_tokens_index[ent1_span[0]:ent1_span[1]],
                        wordpiece_tokens_index[ent1_span[1]:ent2_span[0]],
                        wordpiece_tokens_index[ent2_span[0]:ent2_span[1]], wordpiece_tokens_index[ent2_span[1]:]
                    ]

                    random.shuffle(permutations)

                    for label in [0] + permutations[:4]:
                        permutation = permutation_labels[label]
                        permutated_wordpiece_tokens, permutated_wordpiece_tokens_index, permutated_ent_spans = permutate_tokens(
                            permutation, entity_mention_parts, wordpiece_tokens)
                        instance = {}
                        instance['tokens_id'] = [vocab[token] for token in permutated_wordpiece_tokens]
                        instance['tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in permutated_wordpiece_tokens_index
                        ]
                        instance['spans'] = [permutated_ent_spans[0], permutated_ent_spans[1]]
                        instance['labels'] = label

                        instance = json.dumps(instance)
                        print(instance, file=fout)

            processed_cnt += 1
            if processed_cnt % 10000 == 0:
                print("Processed {} sentences.".format(processed_cnt))


def get_confused_context_instance(source_file, target_file, vocab_file):
    with open(source_file, 'r') as fin, open(target_file, 'w') as fout, open(vocab_file, 'r') as vocab_fin:
        vocab = json.load(vocab_fin)
        processed_cnt = 0
        instance_cnt = 0
        for line in fin:
            sent = json.loads(line)
            num_entity = len(sent['entityMentions'])
            if num_entity < 2:
                continue

            wordpiece_tokens = sent['wordpieceSentText'].split(' ')
            if len(wordpiece_tokens) > 256:
                continue

            origin_tokens_id = [vocab[token] for token in wordpiece_tokens]
            origin_tokens_index = [wordpiece_tokens_span[0] for wordpiece_tokens_span in sent['wordpieceTokensIndex']]
            wordpiece_tokens_index = sent['wordpieceTokensIndex']

            entity_idx = list(range(num_entity))
            random.shuffle(entity_idx)

            for i in range(min(num_entity, 8)):
                for j in range(i + 1, min(num_entity, 8)):
                    instance_cnt += 1
                    ent1_idx = min(entity_idx[i], entity_idx[j])
                    ent2_idx = max(entity_idx[i], entity_idx[j])

                    ent1 = sent['entityMentions'][ent1_idx]
                    ent2 = sent['entityMentions'][ent2_idx]

                    ent1_span = ent1['offset']
                    ent2_span = ent2['offset']

                    # random repalece entity tokens

                    if instance_cnt % 5 == 1:
                        for ent in [ent1, ent2]:
                            confuse_context_wordpiece_tokens = list(wordpiece_tokens)
                            for p in range(wordpiece_tokens_index[ent['offset'][0]][0],
                                           wordpiece_tokens_index[ent['offset'][1] - 1][1]):
                                confuse_context_wordpiece_tokens[p] = '[MASK]'

                                instance = {}
                                instance['confused_tokens_id'] = [
                                    vocab[token] for token in confuse_context_wordpiece_tokens
                                ]
                                instance['confused_tokens_index'] = [
                                    wordpiece_tokens_span[0] for wordpiece_tokens_span in wordpiece_tokens_index
                                ]
                                instance['confused_spans'] = [ent1_span, ent2_span]
                                instance['origin_tokens_id'] = origin_tokens_id
                                instance['origin_tokens_index'] = origin_tokens_index
                                instance['origin_spans'] = [ent1_span, ent2_span]

                                instance = json.dumps(instance)
                                print(instance, file=fout)

                    # exchange entity tokens of entity pair
                    if instance_cnt % 5 == 2:
                        entity_mention_parts = [
                            wordpiece_tokens_index[:ent1_span[0]], wordpiece_tokens_index[ent1_span[0]:ent1_span[1]],
                            wordpiece_tokens_index[ent1_span[1]:ent2_span[0]],
                            wordpiece_tokens_index[ent2_span[0]:ent2_span[1]], wordpiece_tokens_index[ent2_span[1]:]
                        ]
                        permutated_wordpiece_tokens, permutated_wordpiece_tokens_index, permutated_ent_spans = permutate_tokens(
                            [0, 3, 2, 1, 4], entity_mention_parts, wordpiece_tokens)

                        instance = {}
                        instance['confused_tokens_id'] = [vocab[token] for token in permutated_wordpiece_tokens]
                        instance['confused_tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in permutated_wordpiece_tokens_index
                        ]
                        instance['confused_spans'] = [permutated_ent_spans[0], permutated_ent_spans[1]]
                        instance['origin_tokens_id'] = origin_tokens_id
                        instance['origin_tokens_index'] = origin_tokens_index
                        instance['origin_spans'] = [ent1_span, ent2_span]

                        instance = json.dumps(instance)
                        print(instance, file=fout)

                    # left/right shift entity tokens
                    if instance_cnt % 5 == 3 and ent1_span[0] != 0:
                        left_shift = min(5, ent1_span[0])
                        confuse_context_wordpiece_tokens, confuse_context_wordpiece_tokens_index \
                            = left_shift_entity_tokens(wordpiece_tokens, wordpiece_tokens_index, ent1_span, left_shift)
                        instance = {}
                        instance['confused_tokens_id'] = [vocab[token] for token in confuse_context_wordpiece_tokens]
                        instance['confused_tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in confuse_context_wordpiece_tokens_index
                        ]
                        instance['confused_spans'] = [[ent1_span[0] - left_shift, ent1_span[-1] - left_shift],
                                                      ent2_span]
                        instance['origin_tokens_id'] = origin_tokens_id
                        instance['origin_tokens_index'] = origin_tokens_index
                        instance['origin_spans'] = [ent1_span, ent2_span]

                        instance = json.dumps(instance)
                        print(instance, file=fout)

                    if instance_cnt % 5 == 4 and ent1_span[-1] != ent2_span[0]:
                        shift = min(5, ent2_span[0] - ent1_span[-1])
                        confuse_context_wordpiece_tokens, confuse_context_wordpiece_tokens_index \
                            = right_shift_entity_tokens(wordpiece_tokens, wordpiece_tokens_index, ent1_span, shift)
                        instance = {}
                        instance['confused_tokens_id'] = [vocab[token] for token in confuse_context_wordpiece_tokens]
                        instance['confused_tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in confuse_context_wordpiece_tokens_index
                        ]
                        instance['confused_spans'] = [[ent1_span[0] + shift, ent1_span[-1] + shift], ent2_span]
                        instance['origin_tokens_id'] = origin_tokens_id
                        instance['origin_tokens_index'] = origin_tokens_index
                        instance['origin_spans'] = [ent1_span, ent2_span]

                        instance = json.dumps(instance)
                        print(instance, file=fout)

                        confuse_context_wordpiece_tokens, confuse_context_wordpiece_tokens_index \
                            = left_shift_entity_tokens(wordpiece_tokens, wordpiece_tokens_index, ent2_span, shift)
                        instance = {}
                        instance['confused_tokens_id'] = [vocab[token] for token in confuse_context_wordpiece_tokens]
                        instance['confused_tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in confuse_context_wordpiece_tokens_index
                        ]
                        instance['confused_spans'] = [ent1_span, [ent2_span[0] - shift, ent2_span[-1] - shift]]
                        instance['origin_tokens_id'] = origin_tokens_id
                        instance['origin_tokens_index'] = origin_tokens_index
                        instance['origin_spans'] = [ent1_span, ent2_span]

                        instance = json.dumps(instance)
                        print(instance, file=fout)

                    if instance_cnt % 5 == 0 and ent2_span[-1] != len(wordpiece_tokens_index):
                        right_shift = min(5, len(wordpiece_tokens_index) - ent2_span[-1])
                        confuse_context_wordpiece_tokens, confuse_context_wordpiece_tokens_index = right_shift_entity_tokens(
                            wordpiece_tokens, wordpiece_tokens_index, ent2_span, right_shift)
                        instance = {}
                        instance['confused_tokens_id'] = [vocab[token] for token in confuse_context_wordpiece_tokens]
                        instance['confused_tokens_index'] = [
                            wordpiece_tokens_span[0] for wordpiece_tokens_span in confuse_context_wordpiece_tokens_index
                        ]
                        instance['confused_spans'] = [
                            ent1_span, [ent2_span[0] + right_shift, ent2_span[-1] + right_shift]
                        ]
                        instance['origin_tokens_id'] = origin_tokens_id
                        instance['origin_tokens_index'] = origin_tokens_index
                        instance['origin_spans'] = [ent1_span, ent2_span]

                        instance = json.dumps(instance)
                        print(instance, file=fout)

            processed_cnt += 1
            if processed_cnt % 10000 == 0:
                print("Processed {} sentences.".format(processed_cnt))


if __name__ == '__main__':
    fire.Fire({
        'entity_tagger': entity_tagger,
        'entity2tokens': record_entity2tokens,
        'masked_entity_token_prediction_instance': get_masked_entity_token_prediction_instance,
        'masked_entity_typing_instance': get_masked_entity_typing_instance,
        'entity_mention_permutation_instance': get_entity_mention_permutation_instance,
        'confused_context_instance': get_confused_context_instance,
    })
