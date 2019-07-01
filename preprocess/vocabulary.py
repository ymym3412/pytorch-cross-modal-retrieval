from tqdm import tqdm
from collections import Counter


class Vocabulary():
    """
    Create vocabulary from dataset
    """

    def __init__(self, tokenizer, max_vocab, min_count, pad_token='@@PADDING@@', unk_token='@@UNKNOWN@@'):
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.pad_token = pad_token
        self.unk_token = unk_token

    def __len__(self):
        return len(self._index2vocab)

    def create_vocabulary(self, datasets):
        """
        Create vocabulary from torchvision.datasets.flickr.Flickr30k
        """
        print('Create vocabulary from {} dataset...'.format(len(datasets)))
        counter = Counter()
        for dataset in datasets:
            for _, captions in tqdm(dataset):
                for caption in captions:
                    tokens = self.tokenizer.tokenize(caption)
                    counter.update(tokens)

        self._index2vocab = [self.pad_token, self.unk_token]
        for token, freq in counter.most_common():
            if freq < self.min_count:
                break
            if len(self._index2vocab) >= self.max_vocab:
                break
            self._index2vocab.append(token)
        self._vocab2index = {token: i for i, token in enumerate(self._index2vocab)}

    def vocab2index(self, token):
        return self._vocab2index[token] if token in self._vocab2index else self._vocab2index['@@UNKNOWN@@']

    def index2vocab(self, idx):
        return self._index2vocab[idx]

    def save_vocab(self, file_path):
        with open(file_path, 'w') as f:
            for token in self._index2vocab:
                f.write(token + '\n')

    def load_vocab(self, file_path):
        with open(file_path) as f:
            self._index2vocab = [line.replace('\n', '') for line in f]

        self._vocab2index = {token: i for i, token in enumerate(self._index2vocab)}
