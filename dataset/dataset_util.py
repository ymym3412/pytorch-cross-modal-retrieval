from .dataset import ImageBatch, CaptionBatch, TripleBatch
import torch
from torch.nn.utils.rnn import pad_sequence


def create_image_batch(image_list):
    return ImageBatch(torch.stack(image_list))


def create_caption_batch(caption_list):
    caption_lengths = torch.LongTensor([sentence_tensor.shape[0] for sentence_tensor in caption_list])
    caption_batch = pad_sequence(caption_list, batch_first=True)
    return CaptionBatch(caption_batch, caption_lengths)


def collate_triplet_wrapper(batch):
    image_triple, cap_triple = [data[0] for data in batch], [data[1] for data in batch]
    img_triple_img = create_image_batch([t[0] for t in image_triple])
    img_triple_cap = create_caption_batch([t[1] for t in image_triple])
    img_triple_neg_cap = create_caption_batch([t[2] for t in image_triple])

    cap_triple_cap = create_caption_batch([t[0] for t in cap_triple])
    cap_triple_img = create_image_batch([t[1] for t in cap_triple])
    cap_triple_neg_img = create_image_batch([t[2] for t in cap_triple])

    return TripleBatch(img_triple_img, img_triple_cap, img_triple_neg_cap), \
           TripleBatch(cap_triple_cap, cap_triple_img, cap_triple_neg_img)
