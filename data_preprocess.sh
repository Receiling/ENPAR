#!/bin/bash

python -m spacy download en_core_web_sm

PYTHONPATH=$(pwd) python inputs/preprocess.py entity_tagger data/pretrain/wikipedia_raw.txt data/pretrain/wikipedia_labeled.json

PYTHONPATH=$(pwd) python inputs/preprocess.py entity2tokens data/pretrain/wikipedia_labeled.json data/pretrain/entity2tokens.json

PYTHONPATH=$(pwd) python inputs/preprocess.py masked_entity_typing_instance data/pretrain/wikipedia_labeled.json data/pretrain/wikipedia_masked_entity_typing_instance.json data/pretrain/entity2tokens.json data/pretrain/bert_base_cased_vocab.json data/pretrain/entity_labels.json

PYTHONPATH=$(pwd) python inputs/preprocess.py masked_entity_token_prediction_instance data/pretrain/wikipedia_labeled.json data/pretrain/wikipedia_masked_entity_token_prediction_instance.json data/pretrain/entity2tokens.json data/pretrain/bert_base_cased_vocab.json

PYTHONPATH=$(pwd) python inputs/preprocess.py entity_mention_permutation_instance data/pretrain/wikipedia_labeled.json data/pretrain/wikipedia_entity_mention_permutation_instance.json data/pretrain/bert_base_cased_vocab.json

PYTHONPATH=$(pwd) python inputs/preprocess.py confused_context_instance data/pretrain/wikipedia_labeled.json data/pretrain/wikipedia_confused_context_instance.json data/pretrain/bert_base_cased_vocab.json