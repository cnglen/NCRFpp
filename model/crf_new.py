#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CRF"""

__author__ = "Wang Gang"

import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.autograd as autograd
import numpy as np
import pandas as pd
import logging


def log_sum_exp(t, dim=1):
    """Compute log-sum-exp in a numerically stable way
      1) exp
      2) sum over `dim`
      3) log

    Input:
       t (M, N, P): input tensor
       dim: along which dim to sum

    Return:
        (M, P) if `dim`=1 else (M, N)

    Note:
       - take max_score=0 to under the log-exp-sum: log( sum( exp(), dim=`dim`))
       - max_score is to Guarantee numerically stable
    """
    _, idx_max = torch.max(t, dim=dim, keepdim=True)  # dim=1: (M, 1, P)
    max_score = torch.gather(t, dim=dim, index=idx_max)  # dim=1: (M, 1, P)
    result = max_score.squeeze(dim=dim) + \
        torch.log(torch.sum(torch.exp(t - max_score.expand_as(vec)), dim=1, keepdim=False))
    return result


class CRF(nn.Module):
    """CRF
    """

    def __init__(self, tag2id, id2tag=None, bos_token="<BOS>"):
        """pass
        """
        super().__init__()
        self.tag2id = tag2id
        if id2tag:
            self.id2tag = id2tag
        else:
            pass                # todo
        self.bos_token = bos_token

    def _calculate_logZ(self, feats, mask):
        pass

    def _score(self, scores, y, mask):
        """
        Calculate the F(y, x), which is ofter referred to as the join  discriminant function

        Input:
          scores: (seq_len, batch_size, tag_size, tag_size)
          y: (batch_size, seq_len)
          mask:  (batch_size, seq_len)

        Output:
          (batch_size)

        Note:

           - In the `scores`, seq_len is without <bos> and <eos>, but tag_size is with <bos> and <eos>
             - when we encode bigram information:
               - <bos> -> ? is encoded => NO necessar of start_energy
               - ? -> <eos> is NOT encoded => ncessary of end_energy
        """

        logging.info("  get seq_len, batch_size, tag_size")
        logging.debug("  note: seq_len: without <bos> and <eos>; tag_size: with <bos> and <eos>")
        seq_len, batch_size, tag_size, _tag_size = scores.size()
        assert(tag_size == _tag_size)
        del(_tag_size)

        logging.info("  encode bigram information ...")
        y_encoded = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        idx = 0
        y_encoded[:, idx] = self.tag2id[self.bos_token] * tag_size + y[:, idx]  # start -> first
        for idx in np.arange(start=1, stop=seq_len):
            y_encoded[:, idx] = y[:, idx - 1] * tag_size + y[:, idx]

        # transition energy:  ? -> STOP_TAG
        end_transition = self.transitions[:, self.tag2id[self.eos_token]] \
                             .contiguous() \
                             .view(1, tag_size) \
                             .expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1, keepdim=True) \
                           .long()  # length for batch,  last word position = length - 1
        end_ids = torch.gather(y, dim=1, index=length_mask - 1)  # index the label id of last word
        end_energy = torch.gather(end_transition, dim=1, index=end_ids).sum()

        y_encoded = y_encoded.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)

        # need convert tags id to search from 400 positions of scores
        tgt_energy = torch.gather(scores.view(seq_len, batch_size, -1), dim=2, index=y_encoded) \
                          .squeeze(dim=2) \
                          .masked_select(mask.transpose(1, 0)) \
                          .sum()

        gold_score = tgt_energy + end_energy
        return gold_score

    def forward(self, features):
        """
        Find the best path and score

        """
        pass

    def NLL_loss(self, features, mask, y):
        """negative log likelihood loss function

        Input:
          features: (batch_size, seq_len, T): output of lstm + hidden2tag

        Output:
          - log exp(F(x, y))/Z(x) = log(Z(x)) - F(x, y)


        features:
          (batch_size, seq_len, tag_size)
          -> (batch_size, seq_len, 1, tag_size)
          -> (batch_size, seq_len, tag_size, tag_size)
          -> (seq_len, batch_size, tag_size, tag_size)

        self.transitions:
          (tag_size, tag_size)
          -> (1, 1, tag_size, tag_size)
          -> (seq_len, batch_size, tag_size, tag_size)

        calculate log Z(x), where Z(x) is the so-called partition function which
          - sum over the scores of all possible sequence y
          - acts as a normalizer

        """

        # step0: combine the features and self.transitions
        scores = features.unsqueeze(dim=2).expand(-1, -1, tag_size, -1).transpose(0, 1) + \
            self.transitions.unsqueeze(dim=0).unsqueeze(dim=0).expand(seq_len, batch_size, -1, -1)

        # step1: calcualte F(x, y)
        gold_score = self._scores(scores, y, mask)

        # step2: calculate the log Z(x)
        batch_size, seq_len, tag_size, _tag_size = features.size
        assert(tag_size == _tag_size)
        del(_tag_size)

        mask = mask.transpose(1, 0).contiguous()  # (seq_len, batch_size)

        score_iterator = enumerate(scores)
        _, init_values = next(score_iterator)

        #START_TAG = self.tag2id[self.bos_token]
        partition = init_values[:, START_TAG, :].clone().view(batch_size, tag_size)
        for location, current_score in score_iterator:
            _tmp = current_score + partition.view(batch_size, tag_size, 1) \
                .expand(batch_size, tag_size, tag_size)
            current_partition = log_sum_exp(_tmp)  # without considering mask, cumulative to current
            mask_idx = mask[location, :].view(batch_size, 1).expand(batch_size, tag_size)
            current_partition_masked = current_partition.masked_select(mask_idx)
            partition.masked_scatter_(mask=mask_idx, source=current_partition_masked)

        # using the score: ? -> STOP
        _tmp = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + \
            partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        current_partition = log_sum_exp(_tmp)
        partition = current_partition[:, STOP_TAG]  # only take STOP_TAG, since the last label must be STOP_TAG

        logZ = partition.sum()
        return logZ - gold_score

    def _viterbi_decode_nbest(self, features):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
