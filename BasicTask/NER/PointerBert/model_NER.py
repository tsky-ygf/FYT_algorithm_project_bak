#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/09 15:52
# @Author  : Czq
# @File    : model_NER.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


class PointerNERBERT(nn.Module):
    def __init__(self, args):
        super(PointerNERBERT, self).__init__()

        self.bert = BertModel.from_pretrained(args.model)
        # self.bert = BertModel(args.bert_config)
        self.num_labels = len(args.labels)
        self.linear_hidden = nn.Linear(args.bert_emb_size, args.hidden_size)
        self.linear_start = nn.Linear(args.hidden_size, self.num_labels)
        self.linear_end = nn.Linear(args.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        bert_emb = self.bert(**inputs)
        bert_out, bert_pool = bert_emb[0], bert_emb[1]

        hidden = self.linear_hidden(bert_out)
        start_logits = self.linear_start(self.gelu(hidden))
        end_logits = self.linear_end(self.gelu(hidden))
        # delete cls and sep
        start_logits = start_logits[:, 1:-1]
        end_logits = end_logits[:, 1:-1]
        start_prob = self.sigmoid(start_logits)
        end_prob = self.sigmoid(end_logits)
        # start_prob = self.softmax(start_logits)
        # end_prob = self.softmax(end_logits)
        return start_prob, end_prob
        # return start_logits, end_logits


class PointerNERBERT2(nn.Module):
    def __init__(self, args):
        super(PointerNERBERT2, self).__init__()

        self.bert = BertModel.from_pretrained(args.model)
        # self.bert = BertModel(args.bert_config)
        self.num_labels = len(args.labels)
        self.num_labels_aux = len(args.labels_aux)
        self.linear_hidden = nn.Linear(args.bert_emb_size, args.hidden_size)
        self.linear_start_aux = nn.Linear(args.hidden_size, self.num_labels_aux)
        self.linear_end_aux = nn.Linear(args.hidden_size, self.num_labels_aux)
        self.linear_start = nn.Linear(args.hidden_size+self.num_labels_aux, self.num_labels)
        self.linear_end = nn.Linear(args.hidden_size+self.num_labels_aux, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        bert_emb = self.bert(**inputs)
        bert_out, bert_pool = bert_emb[0], bert_emb[1]

        hidden = self.linear_hidden(bert_out)

        start_logits_aux = self.linear_start_aux(self.gelu(hidden))
        end_logits_aux = self.linear_end_aux(self.gelu(hidden))

        start_logits = self.linear_start(self.gelu(torch.cat([hidden,start_logits_aux], dim=-1)))
        end_logits = self.linear_end(self.gelu(torch.cat([hidden, end_logits_aux],dim=-1)))
        # delete cls and sep
        start_logits = start_logits[:, 1:-1]
        end_logits = end_logits[:, 1:-1]
        start_prob = self.sigmoid(start_logits)
        end_prob = self.sigmoid(end_logits)

        start_logits_aux = start_logits_aux[:,1:-1]
        end_logits_aux = end_logits_aux[:,1:-1]
        start_prob2 = self.sigmoid(start_logits_aux)
        end_prob2 = self.sigmoid(end_logits_aux)
        return start_prob, end_prob, start_prob2, end_prob2


class PointerNERBERT_softmax(nn.Module):
    def __init__(self, args):
        super(PointerNERBERT_softmax, self).__init__()

        self.bert = BertModel.from_pretrained(args.model)
        # self.bert = BertModel(args.bert_config)
        self.num_labels = len(args.labels)
        self.linear_hidden = nn.Linear(args.bert_emb_size, args.hidden_size)
        self.linear_start = nn.Linear(args.hidden_size, self.num_labels)  # +1 because of softmax
        self.linear_end = nn.Linear(args.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        bert_emb = self.bert(**inputs)
        bert_out, bert_pool = bert_emb[0], bert_emb[1]

        hidden = self.linear_hidden(bert_out)
        start_logits = self.linear_start(self.gelu(hidden))
        end_logits = self.linear_end(self.gelu(hidden))
        # delete cls and sep
        start_logits = start_logits[:, 1:-1]
        end_logits = end_logits[:, 1:-1]
        # start_prob = self.sigmoid(start_logits)
        # end_prob = self.sigmoid(end_logits)
        start_prob = self.softmax(start_logits)
        end_prob = self.softmax(end_logits)
        return start_prob, end_prob


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)


# Multi-class Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: [N, C]
        :param target: [N, ]
        :return:
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, ignore_index=self.ignore_index, reduction='sum')
        return loss


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, ):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs = self.bert(**input_ids)
        sequence_output = outputs[0]
        #
        sequence_output = sequence_output[:, 1:-1]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1) - 2
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions, 1)
            else:
                label_logits = start_positions.float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            attention_mask = attention_mask[:, 1:-1].unsqueeze(-1).repeat(1, 1, self.num_labels)
            attention_mask = attention_mask.contiguous().view(-1, self.num_labels)

            active_loss = attention_mask == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]
            active_loss = active_loss.view(-1)
            active_start_labels = start_positions.contiguous().view(-1)[active_loss]
            active_end_labels = end_positions.contiguous().view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits.float(), active_start_labels.long())
            end_loss = loss_fct(active_end_logits.float(), active_end_labels.long())
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x
