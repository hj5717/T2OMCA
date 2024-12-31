import torch.nn as nn
import torch.nn.functional as F
import torch


# 这是一个辅助函数,用于对矩阵进行掩码操作
def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
        对方阵的上三角部分(包括对角线)进行掩码操作,将其设置为指定值(默认为0)
        :param matrices: 输入的方阵,形状为 [batch_size, seq_len, seq_len]
        :param maskval: 掩码值,默认为0.0
        :param mask_diagonal: 是否对对角线元素进行掩码,默认为True
        :return: 进行了掩码操作后的矩阵
    """
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()
        """
        多头自注意力机制层
        :param emb: 输入特征向量的维度
        :param heads: 注意力头的数量,默认为8
        :param mask: 是否需要对注意力进行掩码,默认为False
        """
        self.emb = emb
        self.heads = heads
        self.mask = mask
        # 计算查询向量,键向量和值向量,对应公式中的 Q,K,V
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, mask):

        h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q

        # get keys, queries, values
        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(k).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t_q, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(
            self,
            emb,
            heads,
            mask,
            ff_hidden_mult=4,
            dropout=0.0
    ):
        super().__init__()
        """
        Transformer 块,包含了多头自注意力和前馈神经网络
        :param emb: 输入特征向量的维度
        :param heads: 注意力头的数量
        :param mask: 是否需要对注意力进行掩码
        :param ff_hidden_mult: 前馈神经网络的隐藏层维度是输入维度的多少倍,默认为4
        :param dropout: dropout 比率,默认为 0.0
         """
        self.attention = MultiHeadAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),  # 前馈全连接层,映射到更高维度
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(ff_hidden_mult * emb, emb)  # 前馈全连接层,映射回输入维度
        )  # 前馈神经网络模块

        self.do = nn.Dropout(dropout)  # dropout层,防止过拟合

    def forward(self, q_k_mask):
        """
        前向传播计算
        :param q_k_mask: 元组,包含 (查询向量,键向量,掩码张量)
        :return: 经过 Transformer 块计算后的向量,以及更新后的键向量和掩码张量
        """
        q, k, mask = q_k_mask

        attended = self.attention(q, k, mask)

        x = self.norm1(attended + q)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, k, mask


class Transformer(nn.Module):

    def __init__(
            self,
            emb,  # 嵌入维度
            heads,  # 注意力头数
            depth,  # TransformerBlock的数量
            ff_hidden_mult=4,  # 前馈隐藏层倍数
            dropout=0.0  # dropout比率，默认为0
    ):
        super().__init__()

        # transformer blocks
        tblocks = []
        for _ in range(depth):  # 创建transformer块
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    mask=False,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout
                ))

        self.tblocks = nn.Sequential(*tblocks)  # Transforme快堆叠，用于依次串联多个神经网络层或子模块，形成一个简单的前向神经网络结构

    def forward(self, q, k, mask=None):
        """
        前向传播函数
        q: 查询张量
        k: 键张量
        mask: 可选的掩码张量
        """
        x, k, mask = self.tblocks((q, k, mask))

        return x
