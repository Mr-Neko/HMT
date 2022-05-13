import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
import clip
from models.augmentation import ImageAugmentation

ia = ImageAugmentation()
class Hook:
    def __init__(self, module, backward=False):

        self.input = [0]
        self.output = [0]
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.input[0] = input
        self.output[0] = output

    def close(self):
        self.hook.remove()

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx

        clips, _ = clip.load('RN101')
        self.visual = clips.visual
        self.pooling = nn.AdaptiveMaxPool2d((7, 7))
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

        for name, module in self.visual.named_modules():

            if name == 'layer2':
                self.hook_layer2 = Hook(module)
            if name == 'layer3':
                self.hook_layer3 = Hook(module)

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, train=False, *args):

        with torch.no_grad():
            images = self.visual(images)
        layer2 = self.pooling(self.hook_layer2.output[0]).view(-1, 512, 49)
        layer3 = self.pooling(self.hook_layer3.output[0]).view(-1, 1024, 49)
        images = torch.cat((layer2, layer3), dim=1).permute(0, 2, 1)

        images = images.detach()

        if train:
            images = ia.randnChooseOne4(images)
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:

                with torch.no_grad():
                    visual = self.visual(visual)
                layer2 = self.hook_layer2.output[0]
                layer3 = self.hook_layer3.output[0]
                layer2 = self.pooling(layer2).view(-1, 512, 49)
                layer3 = self.pooling(layer3).view(-1, 1024, 49)
                visual = torch.cat((layer2, layer3), dim=1).permute(0, 2, 1)

                visual = visual.detach()
                visual = ia.randnChooseOne4(visual)
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
