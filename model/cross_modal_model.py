from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam, SGD


class TripletLossModel:
    def __init__(self, img_encoder, text_encoder, criterion, fineturn=True, use_cuda=False, grad_clip=2, lr=1e-2):
        self.image_encoder = img_encoder
        self.text_encoder = text_encoder
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.grad_clip = grad_clip

        params = list(self.text_encoder.parameters())
        if fineturn:
            params += list(self.image_encoder._classifier[0].parameters())
            params += list(self.image_encoder._classifier[3].parameters())
            params += list(self.image_encoder._classifier[6].parameters())

        self.params = params
        self.optimizer = Adam(self.params, lr=lr, weight_decay=5e-4)
        self.optim_dict = {'Adam': Adam, 'SGD': SGD}

    def cuda(self):
        """switch cuda
        """
        self.image_encoder.cuda()
        self.text_encoder.cuda()

    def cpu(self):
        """switch cpu
        """
        self.image_encoder.cpu()
        self.text_encoder.cpu()

    def state_dict(self):
        state_dict = [self.image_encoder.state_dict(), self.text_encoder.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.image_encoder.train()
        self.text_encoder.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.image_encoder.eval()
        self.text_encoder.eval()

    def forward(self, image_triple, caption_triple):
        # execute image_triple
        image, pos_cap, neg_cap = image_triple.get_batch()
        image_encoded = self.image_encoder(image)
        pos_text_encoded = self.text_encoder(pos_cap[0], pos_cap[1])
        neg_text_encoded = self.text_encoder(neg_cap[0], neg_cap[1])
        image_triple_loss = self.criterion(image_encoded, pos_text_encoded, neg_text_encoded)

        # execute caption_triple
        caption, pos_img, neg_img = caption_triple.get_batch()
        text_encoded = self.text_encoder(caption[0], caption[1])
        pos_image_encoded = self.image_encoder(pos_img)
        neg_image_encoded = self.image_encoder(neg_img)
        caption_triple_loss = self.criterion(text_encoded, pos_image_encoded, neg_image_encoded)

        loss = image_triple_loss + caption_triple_loss

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return loss.item()
