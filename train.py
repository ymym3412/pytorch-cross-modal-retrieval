from torchvision.datasets.flickr import Flickr30k
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
import datetime
import yaml
from cnn_finetune import make_model
import torch

from preprocess.tokenizer import EnglishTokenizer
from preprocess.vocabulary import Vocabulary
from preprocess.transform import Tokenizer, VocabToIndex, ToLongTensor
from dataset.dataset import TripletFlickrDataset
from dataset.dataset_util import collate_triplet_wrapper
from model.text_encoder import TextEncoder
from model.cross_modal_model import TripletLossModel
from model.model_util import save_model


def decay_learning_rate(init_lr, optimizer, epoch):
    """
    decay learning late every 4 epoch
    """
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # Load hyperparameters
    with open('parameters.yaml') as f:
        parameters = yaml.safe_load(f)

    # init dataset
    flickr_root = os.getenv('FLICKR_DATA_ROOT')
    flickr_train = os.path.join(flickr_root, 'train')
    flickr_val = os.path.join(flickr_root, 'validation')
    flickr_test = os.path.join(flickr_root, 'test')

    train_data = Flickr30k(flickr_train, os.path.join(flickr_train, 'train_caption.txt'))
    val_data = Flickr30k(flickr_val, os.path.join(flickr_val, 'val_caption.txt'))
    test_data = Flickr30k(flickr_test, os.path.join(flickr_test, 'test_caption.txt'))

    # create vocabulary
    english_tokenizer = EnglishTokenizer()
    if parameters['vocab']['load_vocab']:
        vocabulary = Vocabulary(None, 0, 0)
        vocabulary.load_vocab(parameters['vocab']['vocab_file'])
    else:
        vocabulary = Vocabulary(tokenizer=english_tokenizer,
                                max_vocab=parameters['vocab']['max_vocab'],
                                min_count=parameters['vocab']['min_count'])
        vocabulary.create_vocabulary([train_data, val_data])
        vocabulary.save_vocab('vocab')

    # prepare transform
    image_size = parameters['image']['image_size']
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    caption_transform = transforms.Compose([
        Tokenizer(english_tokenizer),
        VocabToIndex(vocabulary),
        ToLongTensor()
    ])

    # create dataset
    triplet_train_data = TripletFlickrDataset(train_data, image_transform, caption_transform)

    # create dataloader

    train_loader = DataLoader(triplet_train_data,
                              batch_size=parameters['data']['batch_size'],
                              shuffle=True,
                              collate_fn=collate_triplet_wrapper)

    # load pretrained cnn model
    image_model = make_model(parameters['image']['pretrained_model'],
                             num_classes=parameters['cross_modal']['shared_dim'],
                             pretrained=True,
                             input_size=(image_size, image_size))

    # create text model
    text_model = TextEncoder(vocab_size=len(vocabulary),
                             embed_dim=parameters['text']['embed_dim'],
                             hidden_size=parameters['text']['hidden_dim'],
                             output_size=parameters['text']['output_dim'],
                             dropout=parameters['text']['dropout'],
                             num_layers=parameters['text']['num_layers'],
                             use_abs=parameters['text']['use_abs'])

    # Triplet Loss
    criterion = nn.TripletMarginLoss(margin=parameters['learning']['margin'], p=2)

    use_cuda = parameters['learning']['use_cuda']
    init_lr = parameters['learning']['init_lr']
    model = TripletLossModel(image_model,
                             text_model,
                             criterion,
                             fineturn=True,
                             use_cuda=use_cuda,
                             grad_clip=parameters['learning']['grad_clip'],
                             lr=init_lr)

    model.cpu()
    if use_cuda and torch.cuda.is_available():
        model.cuda()

    model.train_start()
    for epoch in range(12):
        decay_learning_rate(init_lr, model.optimizer, epoch)

        for i_batch, batch in enumerate(train_loader):
            image_triple, caption_triple = batch
            if use_cuda:
                image_triple = image_triple.cuda()
                caption_triple = caption_triple.cuda()

            loss = model.forward(image_triple, caption_triple)
            print(f'epoch: {epoch}\titeration: {i_batch}\tLoss: {loss}')

    now = datetime.datetime.now()
    save_model(model.state_dict(), '{}.model'.format(now.strftime("%Y%m%d%H%M")))


if __name__ == '__main__':
    train()
