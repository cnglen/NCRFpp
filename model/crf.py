# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-16 16:57:39
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
START_TAG = -2
STOP_TAG = -1


# def log_sum_exp(vec, m_size):
#     """
#     Calculate log of exp sum
#

#     args:
#         vec (batch_size, vanishing_dim, hidden_dim) : input tensor
#         m_size : hidden_dim

#     return:
#         batch_size, hidden_dim
#     """
#     # print("=======>", vec.shape, vec.shape[2], m_size, vec.shape[2] == m_size)
#     _, idx = torch.max(vec, 1)  # B * 1 * M
#     max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)) \
#                      .view(-1, 1, m_size)  # B * M

#     # # new
#     # _, idx = torch.max(input=vec, dim=1, keepdim=True)  # B * 1 * M
#     # max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)) \
#     #                  .view(-1, 1, m_size)  # B * M

#     # B * M
#     return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)) \
#                                              .view(-1, m_size)


def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm

      - exp(vec)
      - sum over dim=1
      - log

    args:
       vec (batch_size, vanishing_dim, hidden_dim): input tensor

    return:
           (batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, dim=1, keepdim=True)
    max_score = torch.gather(vec, 1, idx)
    result = max_score.squeeze(dim=1) + \
        torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), dim=1, keepdim=False))
    return result


class CRF(nn.Module):

    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        print("build CRF ...")
        self.gpu = gpu
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # NOTE: (i,j): score from i -> j
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        init_transitions[:, START_TAG] = -10000.0
        init_transitions[STOP_TAG, :] = -10000.0
        init_transitions[:, 0] = -10000.0  # ??
        init_transitions[0, :] = -10000.0  # ??
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

        # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size+2, self.tagset_size+2))
        # self.transitions.data.zero_()

    def _calculate_PZ(self, feats, mask):
        """
        Calculate Z(x)

        input:
            feats: (batch_size, seq_len, self.tag_size+2): output of lstm+linear network
            masks: (batch_size, seq_len)

        output:
            Z(x):
            scores: (seq_len, batch_size, tag_size, tag_size)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        print("batch_size={}, seq_len={}, tag_size={}".format(batch_size, seq_len, tag_size))
        assert(tag_size == self.tagset_size + 2)
        mask = mask.transpose(1, 0).contiguous()  # (seq_len, batch_size)

        # version2: feats, self.transitions -> scores
        # Input:
        #   feats: score of (B/M/E/S) @ (batch_size, seq_len)
        #   self.transitions: score of transition of each pair: i->j
        #
        # Output:
        #   scores: score of transition of each pair @ (batch_size, seq_len)
        #
        # Implemnation:
        # feats:
        #   (batch_size, seq_len, tag_size)
        #   -> (batch_size, seq_len, 1, tag_size)
        #   -> (batch_size, seq_len, tag_size, tag_size)
        #   -> (seq_len, batch_size, tag_size, tag_size)
        # self.transitions:
        #   (tag_size, tag_size)
        #   -> (1, 1, tag_size, tag_size)
        #   -> (seq_len, batch_size, tag_size, tag_size)
        scores = feats.unsqueeze(dim=2).expand(-1, -1, tag_size, -1).transpose(0, 1) + \
            self.transitions.unsqueeze(dim=0).unsqueeze(dim=0).expand(seq_len, batch_size, -1, -1)

        # # version1: feats, self.transitions -> scores
        # ins_num = seq_len * batch_size
        # # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        # # feats(t, i, j): score(i->j) @ t
        # feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # # need to consider start
        # scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        # scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.next()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)

        # add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values)  # batch_size, tag_size
            # print cur_partition.data

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            # effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            # let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            # replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask=mask_idx, source=masked_cur_partition)
            print("idx={}, cur_values.shape={}, partition.shape={}".format(idx, cur_values.shape, partition.shape))
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size) \
                                     .expand(batch_size, tag_size, tag_size) + \
            partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values)
        final_partition = cur_partition[:, STOP_TAG]  # give up the STOP_TAG columns

        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        orig_feats = feats.clone()
        orig_mask = mask.clone()

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size + 2)
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        # reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = seq_iter.next()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            # print "cur value:", cur_values.size()
            partition, cur_bp = torch.max(cur_values, 1)
            # print "partsize:",partition.size()
            # exit(0)
            # print partition
            # print cur_bp
            # print "one best, ",idx
            partition_history.append(partition)
            # cur_bp: (batch_size, tag_size) max source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # exit(0)

        # add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()  # (batch_size, seq_len. tag_size)

        # get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)

        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
            self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        # print(last_values.shape)
        # print("last_bp", last_bp)

        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        # select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()  # (batch_size, seq_len, tag_size)a
        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(dim=1, index=last_position, src=insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1, 0).contiguous()

        # decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], dim=1, index=pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)

        score_v2, best_path_v2 = self._viterbi_decode_v2(orig_feats, orig_mask)
        print("---------------------------------------------------")
        print(((best_path_v2 - decode_idx).abs() > 0).sum().data.cpu().numpy()[0])

        return score_v2, best_path_v2

    def _viterbi_decode_v2(self, features, mask):
        """
        viterbi decode

        Input:
          features: (batch_size, seq_len, tag_size_to)
          mask:     (batch_size, seq_len)

        Output:
          best_path:  (batch_size, seq_len)
          best_score: (batch_size)
        """
        batch_size, seq_len, tag_size = features.size()

        scores = features.unsqueeze(dim=2).expand(-1, -1, tag_size, -1).transpose(0, 1) + \
            self.transitions.unsqueeze(dim=0).unsqueeze(dim=0).expand(seq_len, batch_size, -1, -1)

        # location iterator
        seq_mask = mask.transpose(1, 0).contiguous()  # (seq_len, batch_size)
        padded_mask = (1 - seq_mask.long()).byte()      # 1~padded, 0~not padded

        # partition_history[j][batch][tag]: for each location `j`, each `batch`: best score(cumulative) ended in `tag`
        # back_point_history[j][batch][tag]: for each location `j`, each `batch`: the previous path node of "best path score ended in `tag`"
        # both element of partition_history and back_point_history: (batch_size, tag_size_to)
        partition_history = []
        back_point_history = []
        sequence_iterator = enumerate(scores)
        _, init_values = next(sequence_iterator)
        # START_TAG = self.tag2id[self.bos_token]

        partition = init_values[:, START_TAG, :].clone().view(batch_size, tag_size)  # (batch_size, tag_size_to)
        partition_history.append(partition)

        for location, current_score in sequence_iterator:
            # score(prev_tag -> current_tag) + score(->prev_tag)
            _tmp = current_score + partition.view(batch_size, tag_size, 1) \
                .expand(-1, -1, tag_size)      # (batch_size, tag_size_from, tag_size_to)
            partition, back_point = torch.max(_tmp, dim=1, keepdim=False)
            special_value = 0
            back_point.masked_fill_(mask=padded_mask[location, :].view(batch_size, 1).expand(-1, tag_size),
                                    value=special_value)
            partition_history.append(partition)
            back_point_history.append(back_point)

        # ? -> STOP:
        # step1: 每一个batch, 每一个location, to_tag_size的累计最好分数
        partition_history = torch.stack(partition_history, dim=0) \
            .transpose(1, 0) \
            .contiguous()  # (batch_size, seq_len, tag_size)
        # step2: get the last position for each setences, and select the last partitions using gather()
        length_mask = torch.sum(mask.long(), dim=1, keepdim=True)  # (batch_size, 1)
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, dim=1, index=last_position)\
                              .view(batch_size, tag_size)
        # step3: get partition, back_point
        _tmp = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + \
            last_partition.view(batch_size, tag_size, 1).expand(-1, -1, tag_size)
        partition, back_point = torch.max(_tmp, dim=1, keepdim=False)
        best_score = partition[:, STOP_TAG].squeeze()  # batch_size

        # pad zero to back_point_history to save the last path node (Note: for no padding sequence)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_point_history.append(pad_zero)
        back_point_history = torch.stack(back_point_history, dim=0).contiguous()  # seq_len, batch_size, tag_size

        # move the ending tags(expand to tag_size) to the corresponding position of back_point_history to replace the 0 values
        ending_tag = back_point[:, STOP_TAG].contiguous().view(batch_size, 1).expand(batch_size, tag_size).unsqueeze(0)
        _last_position = length_mask.view(1, batch_size, 1).expand(1, batch_size, tag_size) - 1
        back_point_history.scatter_(dim=0, index=_last_position, src=ending_tag)

        best_path = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            best_path = best_path.cuda()

        # for padded sequence, will alse insert a wrong path, but will be corrected by mask
        best_path[-1, :] = back_point[:, STOP_TAG]

        if self.gpu:
            back_point_history = back_point_history.cuda()

        for idx in np.arange(start=back_point_history.size(0) - 2, stop=-1, step=-1):
            best_path[idx, :] = torch.gather(back_point_history[idx], dim=1,
                                             index=best_path[idx + 1, :].view(batch_size, 1))

        best_path = best_path.transpose(0, 1) * mask.long()
        return best_score, best_path

    def forward(self, feats, mask):
        """
        get the path_score, best_path
        """
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Calculate the gold score for gold `tags`

        input:
             scores: (seq_len, batch_size, tag_size, tag_size)
             mask: (batch_size, seq_len)
             tags: (batch_size, seq_len) gold tags
        output:
             score: sum of score for gold sequences `tags` within whole batch
        """

        # Gives the score of a provided tag sequence
        seq_len = scores.size(0)
        batch_size = scores.size(1)
        tag_size = scores.size(2)

        # convert `tag` value into a new format, recorded label bigram information to index
        # new_value=previous_tag * tag_size + current_tag:  (new_value//tag_size, new_value % tag_size) <==> (previous_tag, current_tag)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                # new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
                new_tags[:, 0] = (START_TAG % tag_size) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        # transition for label to STOP_TAG
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        # length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1, keepdim=True).long()
        # index the label id of last word
        end_ids = torch.gather(tags, dim=1, index=length_mask - 1)

        # index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, dim=1, index=end_ids)

        # convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)

        # need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), dim=2, index=new_tags) \
            .view(seq_len, batch_size)
        # mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        # add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        return forward_score - gold_score

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size + 2)
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        # reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = seq_iter.next()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone()  # bat_size * to_target_size
        # initial partition [batch_size, tag_size]
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(
                    batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + \
                    partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                # compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            # cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            # print partition[:,0,:]
            # print cur_bp[:,0,:]
            # print "nbest, ",idx
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            # print partition
            # exit(0)
            # partition: (batch_size * to_target * nbest)
            # cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            # cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            # mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            # print cur_bp[0]
            back_points.append(cur_bp)
        # add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, tag_size, nbest).transpose(
            1, 0).contiguous()  # (batch_size, seq_len, nbest, tag_size)
        # get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + \
            self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        # end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        # select end ids in STOP_TAG
        pointer = end_bp[:, STOP_TAG, :]  # (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1, 0).contiguous()
        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last[0]
        # exit(0)
        # copy the ids of last position:insert_last to back_points, though the last_position index
        # last_position includes the length of batch sentences
        # print "old:", back_points[9,0,:,:]
        back_points.scatter_(1, last_position, insert_last)
        # back_points: [batch_size, seq_length, tag_size, nbest]
        # print "new:", back_points[9,0,:,:]
        # exit(0)
        # print pointer[2]
        '''
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        '''

        back_points = back_points.transpose(1, 0).contiguous()
        # print back_points[0]
        # back_points: (seq_len, batch, tag_size, nbest)
        # decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data / nbest
        # print "pointer-1:",pointer[2]
        # exit(0)
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            # print "pointer: ",idx,  pointer[3]
            # print "back:",back_points[idx][3]
            # print "mask:",mask[idx+1,3]
            new_pointer = torch.gather(back_points[idx].view(
                batch_size, tag_size * nbest), 1, pointer.contiguous().view(batch_size, nbest))
            decode_idx[idx] = new_pointer.data / nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = new_pointer + pointer.contiguous().view(batch_size, nbest) * \
                mask[idx].view(batch_size, 1).expand(batch_size, nbest).long()

        # exit(0)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        # decode_idx: [batch, seq_len, nbest]
        # print decode_idx[:,:,0]
        # print "nbest:",nbest
        # print "diff:", decode_idx[:,:,0]- decode_idx[:,:,4]
        # print decode_idx[:,0,:]
        # exit(0)

        # calculate probability for each sequence
        scores = end_partition[:, :, STOP_TAG]
        # scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        # path_score: [batch_size, nbest]
        # exit(0)
        return path_score, decode_idx
