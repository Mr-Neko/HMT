import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .attention import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att_layer4 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.enc_att_layer3 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.enc_att_layer2 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1=nn.Dropout(dropout)
        self.lnorm1=nn.LayerNorm(d_model)

        self.dropout2_layer2=nn.Dropout(dropout)
        self.lnorm2_layer2=nn.LayerNorm(d_model)
        self.pwff_layer2 = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout2_layer3=nn.Dropout(dropout)
        self.lnorm2_layer3=nn.LayerNorm(d_model)
        self.pwff_layer3 = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout2_layer4=nn.Dropout(dropout)
        self.lnorm2_layer4=nn.LayerNorm(d_model)
        self.pwff_layer4 = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        #MHA+AddNorm
        self_att = self.self_att(input, input, input, None, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        enc_att_layer2 = self.enc_att_layer2(self_att, enc_output[0], enc_output[0], None, mask_enc_att)
        enc_att_layer2 = self.lnorm2_layer2(self_att + self.dropout2_layer2(enc_att_layer2))
        enc_att_layer2 = enc_att_layer2 * mask_pad
        # FFN+AddNorm
        ff_layer2 = self.pwff_layer2(enc_att_layer2)
        ff_layer2 = ff_layer2 * mask_pad

        enc_att_layer3 = self.enc_att_layer3(self_att, enc_output[1], enc_output[1], None, mask_enc_att)
        enc_att_layer3 = self.lnorm2_layer3(self_att + self.dropout2_layer3(enc_att_layer3))
        enc_att_layer3 = enc_att_layer3 * mask_pad
        # FFN+AddNorm
        ff_layer3 = self.pwff_layer3(enc_att_layer3)
        ff_layer3 = ff_layer3 * mask_pad

        enc_att_layer4 = self.enc_att_layer4(self_att, enc_output[2], enc_output[2], None, mask_enc_att)
        enc_att_layer4 = self.lnorm2_layer4(self_att + self.dropout2_layer4(enc_att_layer4))
        enc_att_layer4 = enc_att_layer4 * mask_pad
        # FFN+AddNorm
        ff_layer4 = self.pwff_layer4(enc_att_layer4)
        ff_layer4 = ff_layer4 * mask_pad

        score_layer2 = (ff_layer2 * input).sum(dim=-1).unsqueeze(dim=-1)
        score_layer3 = (ff_layer3 * input).sum(dim=-1).unsqueeze(dim=-1)
        score_layer4 = (ff_layer4 * input).sum(dim=-1).unsqueeze(dim=-1)

        score_matrix = torch.cat((score_layer2, score_layer3, score_layer4), dim=-1)
        _, index = torch.max(score_matrix, dim=-1)

        index = index.unsqueeze(dim=-1)
        ff = ff_layer2 * (index == 0) + ff_layer3 * (index == 1) + ff_layer4 * (index == 2)

        return ff


class TransformerDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

        self.weight_linear = nn.Linear(d_model, 1)

    def forward(self, input, encoder_output, mask_encoder, pos=None):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        '''
        weight_input = torch.cat(
            (outputs[0].mean(dim=1).unsqueeze(dim=1), outputs[1].mean(dim=1).unsqueeze(dim=1), outputs[2].mean(dim=1).unsqueeze(dim=1)), dim=1)
        weight_input = torch.nn.ReLU()(self.weight_linear(weight_input)).squeeze(dim=-1)
        weight_input = torch.nn.Softmax(dim=-1)(weight_input)

        out = weight_input[:, 0].unsqueeze(dim=-1).unsqueeze(dim=-1) * outputs[0]\
              + weight_input[:, 1].unsqueeze(dim=-1).unsqueeze(dim=-1) * outputs[1]\
              + weight_input[:, 2].unsqueeze(dim=-1).unsqueeze(dim=-1) * outputs[2]
        '''


        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
