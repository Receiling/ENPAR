from inputs.fields.field import Field
import logging

logger = logging.getLogger(__name__)


class CharTokenField(Field):
    """This class splits each token into chars
    """
    def __init__(self, namespace, vocab_namespace, source_key, is_counting=True):
        """This function sets namespace of field, vocab namespace for indexing, dataset source key

        Arguments:
            namespace {str} -- namespace of field, counter namespace if constructing a counter
            vocab_namespace {str} -- vocab namespace for indexing
            source_key {str} -- indicate key in text data
        
        Keyword Arguments:
            is_counting {bool} -- decide constructing a counter or not (default: {True})
        """

        super().__init__()
        self.namespace = str(namespace)
        self.counter_namespace = str(namespace)
        self.vocab_namespace = str(vocab_namespace)
        self.source_key = str(source_key)
        self.is_counting = is_counting

    def count_vocab_items(self, counter, sentences):
        """This function counts token's char in sentences,
        then updates counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        if self.is_counting:
            for sentence in sentences:
                for token in sentence[self.source_key]:
                    for char in token:
                        counter[self.counter_namespace][str(char)] += 1

            logger.info("Count sentences {} to update counter namespace {} successfully.".format(
                self.source_key, self.counter_namespace))

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary,
        then update instance

        Arguments:
            instance {dict} -- numerical representation of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            token_num_repr = []
            for token in sentence[self.source_key]:
                token_num_repr.append(
                    [vocab.get_token_index(char, self.vocab_namespace) for char in token])
            instance[self.namespace].append(token_num_repr)

        logger.info("Index sentences {} to construct instance namespace {} successfully.".format(
            self.source_key, self.namespace))
