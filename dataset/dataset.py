from torch.utils.data import Dataset
import torch
import random


# Dataset for triplet loss
class TripletFlickrDataset(Dataset):
    def __init__(self, flickr_dataset, transform=None, target_transform=None):
        self.dataset = flickr_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get Negative sample
        while True:
            negative_img_id = random.randint(0, len(self.dataset) - 1)
            if negative_img_id != idx:
                break

        while True:
            negative_cap_id = random.randint(0, len(self.dataset) - 1)
            if negative_cap_id != idx:
                break

        image, captions = self.dataset[idx]
        # choose a sample from captions
        caption = captions[random.randint(0, len(captions) - 1)]
        negative_image = self.dataset[negative_img_id][0]
        negative_captions = self.dataset[negative_cap_id][1]
        negative_caption = negative_captions[random.randint(0, len(negative_captions) - 1)]

        if self.transform:
            image = self.transform(image)
            negative_image = self.transform(negative_image)
        if self.target_transform:
            caption = self.target_transform(caption)
            negative_caption = self.target_transform(negative_caption)

        # image_triple, caption_triple
        return (image, caption, negative_caption), (caption, image, negative_image)


class ImageBatch:
    def __init__(self, image):
        assert isinstance(image, torch.FloatTensor), 'Not FloatTensor'
        self.image_batch = image

    def cuda(self):
        self.image_batch = self.image_batch.cuda()
        return self

    def cpu(self):
        self.image_batch = self.image_batch.cpu()
        return self

    def get_batch(self):
        return self.image_batch


class CaptionBatch:
    def __init__(self, caption, caption_length):
        assert isinstance(caption, torch.LongTensor), 'caption Not LongTensor'
        assert isinstance(caption_length, torch.LongTensor), 'caption_length Not LongTensor'
        self.caption_batch = caption
        self.caption_length = caption_length

    def cuda(self):
        self.caption_batch = self.caption_batch.cuda()
        self.caption_length = self.caption_length.cuda()
        return self

    def cpu(self):
        self.caption_batch = self.caption_batch.cpu()
        self.caption_length = self.caption_length.cpu()
        return self

    def get_batch(self):
        return self.caption_batch, self.caption_length


class TripleBatch:
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def cuda(self):
        self.anchor = self.anchor.cuda()
        self.positive = self.positive.cuda()
        self.negative = self.negative.cuda()
        return self

    def cpu(self):
        self.anchor = self.anchor.cpu()
        self.positive = self.positive.cpu()
        self.negative = self.negative.cpu()
        return self

    def get_batch(self):
        return self.anchor.get_batch(), self.positive.get_batch(), self.negative.get_batch()
