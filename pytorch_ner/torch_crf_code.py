__version__ = '0.7.2'

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.
    条件随机场

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    这个模块实现了一个条件随机字段。前向计算定标签序列和发射矩阵得分的对数似然。
    这个类还有一个`~CRF.decode`查找的方法，使用“维特比算法”找出发射矩阵得分的最佳标签序列。

    Args:
        num_tags: 标签的类别
        batch_first: 第1个维度是否对应着批次

    Attributes:
        start_transitions (`~torch.nn.Parameter`): 起始位置分数张量的大小
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): 结束位置分数张量的大小
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): 过渡分数张量的大小
            ``(num_tags, num_tags)``.

    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # 初始状态概率向量
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        # 转移概率矩阵
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化转换参数

        参数将从均匀分布中随机初始化
        取值介于-0.1和0.1之间。
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        根据发射矩阵的得分计算与标签之间的对数似然

        Args:
            emissions: 发射矩阵的张量大小。
            如果batch_first为False，则为大小为(seq_length, batch_size, num_tags)，
            否则为(batch_size, seq_length, num_tags)。
            
            tags: 标签序列张量大小。
            如果batch_first 为False，则为大小为(seq_length, batch_size)，
            否则为(batch_size, seq_length)。
            
            mask: Mask张量大小 
            如果 batch_first 为 False, 则大小为(seq_length, batch_size)
            否则 (batch_size, seq_length)。

            reduction: 指定要应用于输出的数据收缩计算方法,取值包括（none|sum|mean|token_mean）
            none: 不会进行任何收缩
            sum: 计算每批次总和
            mean: 计算每个批次的均值 
            token_mean: 计算每个token的均值

        Returns:
            对数似然。如果reduce 为none，则大小为(batch_size,)。
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            # 矩阵转置，batch_size后移 
            # (batch_size,seq_len,feature_size) -> (seq_len,batch_size,feature_size)
            emissions = emissions.transpose(0, 1) 
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 计算分子 shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # 计算分母 shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # (log likelihood) shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """
        使用维特比算法找到最可能的标签序列
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            包含每个批次中最佳标签序列的列表。
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        # 检测发射矩阵维度必须为3
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # 得分 = 初始转移概率值 + 发射矩阵中初始状态值
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]] # 当前批次中的初始状态值
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # 计算到下个标签的转移概率，通过mask统计是否要把概率值与score加和
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # 发射矩阵中概率与score加和，同样通过mask控制
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # score + 结束转移矩阵
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        # 序列长度
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)

        # score = 起始转移矩阵得分 + 发射概率矩阵，初始时间步的得分(所有状态的得分)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # 广播得分对应着当前状态转移到其它类标签的概率
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # 广播发射得分对应着发射概率矩阵当前状态概率
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # 对于每个样本，第i行和第j列的条目存储的所有可能的标签序列的分数总和
            # 从标签i转换到标签j的发射概率值
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # 对所有可能的当前标签求和，log-sum-exp
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # 设置到下个状态的得分
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # 加上结束转移状态的得分
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # 所有标签概率的得分 logZ(x)
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # 初始始转移概率 + 发射矩阵第0个状态
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score表示的张量大小为(batch_size, num_tags)
        # 标签存储的是：到目前为止以到标签i结束的最佳标签序列的分数 
        # history 记录的时保存最佳候选标签的前一个位置。以便在追溯最佳标签序列时使用

        # 维特比算法递归思想:为每个可能的下一个标签计算最佳标签序列的分数
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # 为每个可能的转移的下一个状态创建的‘广播维特比得分’
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # 为每个可能的当前状态标签创建的‘广播发射概率得分’
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # 分数张量的大小为 (batch_size，num_tags，num_tags)
            # 对于每个样本，行列存储的是迄今为止最佳状态标签序列的分数
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # 查找所有可能的当前标签的最高分数
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # 如果此时间步长有效(mask == 1)，则将分数设置为下一个分数，并保存产生下一个分数的索引
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # 结尾转移状态的得分
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # 计算每个样本的最佳路径
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # 找到在最后一个时间步长得分最高的状态标签，这是最后一步的标签
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # 我们追溯最佳的最后一个标签来自哪里，将它附加到我们的最佳标签序列中，
            # 然后再次追溯，以此类推
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # 颠倒顺序，因为我们从最后一个时间步长开始
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
