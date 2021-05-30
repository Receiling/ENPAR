def print_predictions(outputs, file_path, vocab, sequence_label_domain):
    """This fucntion prints prediction results
    
    Arguments:
        outputs {list} -- prediction outputs
        file_path {str} -- output file path
        vocab {Vocabulary} -- vocabulary
        sequence_label_domain {str} -- sequence label domain
    """

    with open(file_path, 'w') as fout:
        for sent_output in outputs:
            seq_len = sent_output['seq_len']
            tokens = []
            if 'tokens' in sent_output and 'sequence_labels' in sent_output and 'sequence_label_preds' in sent_output:
                for token_idx, true_sequence_label, pred_sequence_label in zip(
                        sent_output['tokens'][:seq_len], sent_output['sequence_labels'][:seq_len],
                        sent_output['sequence_label_preds'][:seq_len]):
                    token = vocab.get_token_from_index(token_idx, 'tokens')
                    tokens.append(token)
                    true_sequence_label = vocab.get_token_from_index(true_sequence_label, sequence_label_domain)
                    pred_sequence_label = vocab.get_token_from_index(pred_sequence_label, sequence_label_domain)
                    print("{}\t{}\t{}".format(token, true_sequence_label, pred_sequence_label), file=fout)

            if 'span2ent' in sent_output:
                for span, ent in sent_output['span2ent'].items():
                    ent = vocab.get_token_from_index(ent, 'span2ent')
                    assert ent != 'None', "true relation can not be `None`."

                    print("Ent-True\t{}\t{}\t{}".format(ent, span, ' '.join(tokens[span[0]:span[1]])), file=fout)

            if 'all_ent_preds' in sent_output:
                for span, ent in sent_output['all_ent_preds'].items():
                    # ent = vocab.get_token_from_index(ent, 'span2ent')
                    if ent == 'None':
                        continue

                    print("Ent-Pred\t{}\t{}\t{}".format(ent, span, ' '.join(tokens[span[0]:span[1]])), file=fout)

            if 'span2rel' in sent_output:
                for (span1, span2), rel in sent_output['span2rel'].items():
                    rel = vocab.get_token_from_index(rel, 'span2rel')
                    assert rel != 'None', "true relation can not be `None`."

                    if rel[-1] == '<':
                        span1, span2 = span2, span1
                    print("Rel-True\t{}\t{}\t{}\t{}\t{}".format(rel[:-2], span1, span2,
                                                                ' '.join(tokens[span1[0]:span1[1]]),
                                                                ' '.join(tokens[span2[0]:span2[1]])),
                          file=fout)

            if 'all_rel_preds' in sent_output:
                for (span1, span2), rel in sent_output['all_rel_preds'].items():
                    # rel = vocab.get_token_from_index(rel, 'span2rel')
                    if rel == 'None':
                        continue

                    if rel[-1] == '<':
                        span1, span2 = span2, span1
                    print("Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(rel[:-2], span1, span2,
                                                                ' '.join(tokens[span1[0]:span1[1]]),
                                                                ' '.join(tokens[span2[0]:span2[1]])),
                          file=fout)

            print(file=fout)
