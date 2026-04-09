import math

import torch
from torch import nn
from torch.nn import init

class M_head_t(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask, adj):
        super(M_head_t, self).__init__()
        self.wq = nn.Linear(input_dim, hidden_dim)
        self.wk = nn.Linear(input_dim, hidden_dim)
        self.wv = nn.Linear(input_dim, hidden_dim)
        self.mask = mask
        self.adj = adj

    def forward(self, x):
        q = self.wq(x) # [B, T, N, D]
        k = self.wk(x) # [B, T, N, D]
        v = self.wv(x)# [B, T, N, D]
        B, T, N, D = q.shape  # 从张量本身提取维度信息

        q = q.permute(0, 2, 1, 3).reshape(B * N, T, D)
        k = k.permute(0, 2, 1, 3).reshape(B * N, T, D)
        v = v.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_scores = torch.einsum('btd,bsd->bts', q, k)  # [B*N, T, T]

        attn_scores = attn_scores / math.sqrt(D)

        if self.mask == 'time':
            mask = torch.tril(torch.ones(T, T)).to(attn_scores.device)  # 掩码，一半为0，一半为1
            mask = mask.unsqueeze(0)  # [1, T, T]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        elif self.mask == 'space':
            adj_mask = self.adj.unsqueeze(0).to(attn_scores.device)  # [1, T, T], 把邻接矩阵扩展为与 attn_scores 对应的形状
            attn_scores = attn_scores * adj_mask  # 将邻接矩阵应用到注意力分数上

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B*N, T, T]

        out = torch.einsum('bts,bsd->btd', attn_probs, v)  # [B*N, T, D]
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # → [B, T, N, D]
        return out


class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, node_num):
        super().__init__()
        self.wz = nn.Parameter(torch.randn(input_dim + hidden_dim, hidden_dim),
                               requires_grad=True)
        init.xavier_uniform_(self.wz)
        self.bz = nn.Parameter(torch.randn(node_num, hidden_dim), requires_grad=True)
        init.xavier_uniform_(self.bz)

        self.wr = nn.Parameter(torch.randn(input_dim + hidden_dim, hidden_dim),
                               requires_grad=True)
        init.xavier_uniform_(self.wr)
        self.br = nn.Parameter(torch.randn(node_num, hidden_dim), requires_grad=True)
        init.xavier_uniform_(self.br)

        self.wh = nn.Parameter(torch.randn(input_dim + hidden_dim, hidden_dim),
                               requires_grad=True)
        init.xavier_uniform_(self.wh)
        self.bh = nn.Parameter(torch.randn(node_num, hidden_dim), requires_grad=True)
        init.xavier_uniform_(self.bh)

    def forward(self, x, ht):
        z = torch.sigmoid(torch.matmul(torch.cat((x, ht), dim=-1), self.wz) + self.bz)
        r = torch.sigmoid(torch.matmul(torch.cat((x, ht), dim=-1), self.wr) + self.br)
        ht_ = torch.tanh(torch.matmul(torch.cat((x, r * ht), dim=-1), self.wh) + self.bh)
        ht = (1 - z) * ht + z * ht_
        return ht


class Model(nn.Module):
    def __init__(self, laplace, train_shape, pre_step, device):
        super(Model, self).__init__()
        self.device = device
        laplace = torch.softmax(laplace, dim=-1)
        self.laplace = laplace.to(device)
        self.pre_step = int(pre_step/5)
        self.adj = (self.laplace != 0).bool()

        self.head_count = 6
        self.head_out = 8

        self.head_t_ahead = nn.ModuleList([M_head_t(train_shape[-1], self.head_out, mask='time', adj=self.adj) for _ in range(self.head_count)])  # range(5)5个多头注意力机制
        self.head_g_ahead = nn.ModuleList([M_head_t(train_shape[-1], self.head_out, mask='time', adj=self.adj) for _ in range(self.head_count)])  # range(5)5个多头注意力机制
        self.head_s_ahead = nn.ModuleList([M_head_t(train_shape[-1], self.head_out, mask='space', adj=self.adj) for _ in range(self.head_count)])
        self.head_st_ahead = nn.ModuleList([M_head_t(train_shape[-1], self.head_out, mask='space', adj=self.adj) for _ in range(self.head_count)])

        self.conv = nn.Conv2d(in_channels=train_shape[-1], out_channels=self.head_out * len(self.head_t_ahead),
                              kernel_size=[1, int(pre_step / 5) + 11], padding=0)
        self.conv1 = nn.Conv2d(in_channels=train_shape[-1], out_channels=self.head_out * len(self.head_t_ahead),
                              kernel_size=[1, int(pre_step / 5) + 12], padding=0)
        self.pre_laplace_linre = nn.Linear(in_features=1, out_features=self.head_out * len(self.head_t_ahead))
        self.pre_x_laplace_linre = nn.Linear(in_features=1, out_features=self.head_out * len(self.head_t_ahead))

        self.out_proj_t = nn.Linear(self.head_out * len(self.head_t_ahead), self.head_out * len(self.head_t_ahead))
        self.out_proj_g = nn.Linear(self.head_out * len(self.head_g_ahead), self.head_out * len(self.head_g_ahead))
        self.out_proj_s = nn.Linear(self.head_out * len(self.head_s_ahead), self.head_out * len(self.head_s_ahead))
        self.out_proj_st = nn.Linear(self.head_out * len(self.head_st_ahead), self.head_out * len(self.head_st_ahead))
        self.residual_proj = nn.Linear(train_shape[-1], self.head_out * len(self.head_t_ahead))

        self.xhr_GRU = nn.ModuleList(
            [CustomGRUCell(2 * self.head_out * len(self.head_t_ahead), self.head_out * len(self.head_t_ahead), train_shape[1]) for i in range(12)]
        )

        self.last_GRU = nn.ModuleList(
            [CustomGRUCell(self.head_out * len(self.head_t_ahead), self.head_out * len(self.head_t_ahead), train_shape[1]) for i in range(pre_step)]
        )
        self.out = nn.Linear(self.head_out * len(self.head_t_ahead), 1)
        print("初始化完成")

    def forward(self, x, T):
        fx_1 = x

        time_pre_x = torch.cat((T.new_zeros(T.shape[0], T.shape[1], T.shape[2], 1), T), dim=-1)
        time_pre_x = torch.cat((fx_1, time_pre_x), dim=1)
        time_pre_x = torch.cat((time_pre_x, time_pre_x[:, :self.pre_step - 1, :, :]), dim=1)
        pre_laplace_x = self.conv1(time_pre_x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        att_t, att_g, att_s, att_st = [], [], [], []
        for head_t, head_l, head_s, head_st in zip(self.head_t_ahead, self.head_g_ahead, self.head_s_ahead, self.head_st_ahead):
            att_t.append(head_t(fx_1))
            att_st.append(head_st(fx_1.permute(0, 2, 1, 3)).permute(0, 2, 1, 3))
        att_t_cat = torch.cat(att_t, dim=-1)
        att_st_cat = torch.cat(att_st, dim=-1)
        att_t_out = self.out_proj_t(att_t_cat)
        att_st_out = self.out_proj_st(att_st_cat)
        att_stt_out = torch.cat((att_t_out, att_st_out), dim=-1)

        ht = att_t_out.new_zeros(att_t_out[:, 0, :, :].shape)
        for i in range(1, att_stt_out.shape[1]):
            ht = self.xhr_GRU[i](att_stt_out[:, i, :, :], ht)

        out = pre_laplace_x.new_zeros(pre_laplace_x.shape)
        for i in range(T.shape[1]):
            ht = self.last_GRU[i](pre_laplace_x[:, i, :, :], ht)
            out[:, i, :, :] = ht
        out = self.out(out)
        return out