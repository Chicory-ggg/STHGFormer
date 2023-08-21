import torch
import torch.nn as nn
import numpy as np
from G_degree import num_node,two_degree

# get node num and two degrees of traffic graph
node_num = num_node()
G_ID, G_OD = two_degree()

# temporal embedding
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

# Spatial ScaledDotProduct
class SpatialScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(SpatialScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_bias):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores = scores + attn_bias
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

# Temporal ScaledDotProduct
class TemporalScaledDotProduct(nn.Module):
    def __init__(self):
        super(TemporalScaledDotProduct, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        # B,H,T,N,C -> B,H,N,T,C
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        # B,H,N,T,C * B,H,N,C,T -> B,H,N,T,T
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        context = context.transpose(2, 3)  # B,H,T,N,C
        return context

# Spatial Attention
class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V, attn_bias):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = SpatialScaledDotProductAttention()(Q, K, V, attn_bias)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

# Temporal Attention
class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = TemporalScaledDotProduct()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

# SpaFormer
class SpaFormer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(SpaFormer, self).__init__()
        self.D_S = adj.to('cuda:0')
        self.heads = heads
        self.embed_size = embed_size
        self.in_degree_encoder = nn.Embedding(node_num, embed_size)
        self.out_degree_encoder = nn.Embedding(node_num, embed_size)
        self.rel_pos_encoder = nn.Embedding(4, heads)
        self.attention_s = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ID_embedding = nn.Embedding(node_num, embed_size)
        self.Label_embedding = nn.Embedding(2, embed_size)
        self.feed_forward_s = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query):
        # value, key, query: [N, T, C]  [B, N, T, C]
        B, N, T, C = query.shape
        # HSE - att
        L_T_ID = torch.arange(node_num).to('cuda:0')
        ID_embedding = self.ID_embedding(L_T_ID)  # (node_num, embed_size)
        ID_embedding = ID_embedding.unsqueeze(1)  # (B,1,C)
        ID_embedding = ID_embedding.unsqueeze(0)  # (1,B,1,C)
        ID_embedding = ID_embedding.expand(B, N, T, C)
        # HSE - sig
        in_degree = self.in_degree_encoder(G_ID.int().to('cuda:0'))  # (N,C) (72,64)
        in_degree = in_degree.unsqueeze(1)  # (N,1,C)
        in_degree = in_degree.expand(B, N, T, C)  # (B,N,T,C)
        out_degree = self.out_degree_encoder(G_OD.int().to('cuda:0'))  # (N,C) (72,64)
        out_degree = out_degree.unsqueeze(1)  # (N,1,C)
        out_degree = out_degree.expand(B, N, T, C)  # (B,N,T,C)
        # HSE - rel
        G = self.D_S
        rel_pos_bias = self.rel_pos_encoder(G.long())  # (N,N,heads)
        rel_pos_bias = rel_pos_bias.unsqueeze(2)  # (N,N,1,h)
        rel_pos_bias = rel_pos_bias.expand(N, N, T, self.heads)  # (N,N,T,h)
        rel_pos_bias = rel_pos_bias.permute(3, 2, 0, 1)  # (h,t,N,N)
        rel_pos_bias = rel_pos_bias.unsqueeze(0)  # (1,h,t,N,N)
        rel_pos_bias = rel_pos_bias.expand(B, self.heads, T, N, N)  # (b,h,t,N,N)
        # HSE
        X_tildeS = query + ID_embedding + in_degree + out_degree
        attention_s = self.attention_s(X_tildeS, X_tildeS, X_tildeS, rel_pos_bias)  # (B, N, T, C)
        X_S = self.dropout(self.norm1(attention_s + X_tildeS))
        forward = self.feed_forward_s(X_S)
        out = self.dropout(self.norm2(forward + X_S))
        return out  # (B, N, T, C)

# TempFormer
class TempFormer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(TempFormer, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.W_T_embedding = nn.Linear(2, embed_size)
        self.attention_t = TMultiHeadAttention(embed_size, heads)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.feed_forward_t = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, w_t, origin_input):
        # T
        B, N, T, C = query.shape
        # Temporal Embedding
        D_T = get_sinusoid_encoding_table(T, C).to('cuda:0')
        D_T = D_T.expand(B, N, T, C)
        D_W_T = self.W_T_embedding(w_t)
        D_W_T = D_W_T.unsqueeze(1)  # (B,1,C) (32,1,32)
        D_W_T = D_W_T.unsqueeze(1)  # (B,1,1,C) (32,1,1,32)
        D_W_T = D_W_T.expand(B, N, T, C)
        X_tildeT = query + D_T + D_W_T

        attention_t = self.attention_t(X_tildeT, X_tildeT, X_tildeT)  # (B, N, T, C)
        X_T = self.norm3(attention_t + origin_input)
        forward = self.feed_forward_t(X_T)
        out = self.dropout(self.norm4(forward + X_T))
        return out  # (B, N, T, C)

# STBlock
class STBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(STBlock, self).__init__()
        self.STransformer = SpaFormer(embed_size, heads, adj, dropout, forward_expansion)
        self.TTransformer = TempFormer(embed_size, heads, adj, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.FC = nn.Sequential(
            nn.Linear(in_features=node_num, out_features=node_num),
            nn.BatchNorm1d(node_num),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=node_num, out_features=node_num),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, value, key, query, t, w_t):
        # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        origin_input = query
        # AST
        x_abs = torch.abs(origin_input) # (B,N,T,C)
        gap = self.global_average_pool(x_abs) # (B,N,1,1)
        gap = self.flatten(gap) # (B,N)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        threshold = torch.unsqueeze(threshold, 2) # (B,C,1,1)
        sub = x_abs - threshold # (B,C,N,T)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x_Soft = torch.mul(torch.sign(query), n_sub)

        s_x = self.norm1(self.STransformer(origin_input) + origin_input)  # (B, N, T, C)
        t_x = self.dropout(self.norm2(self.TTransformer(s_x, w_t, x_Soft)) + s_x)
        return t_x

# Encoder
class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t, w_t):
        # x: [N, T, C]  [B, N, T, C]
        # out = self.dropout(x)
        out = x
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, t, w_t)
        return out

## STHGFormer: Total Model
class STHGFormer(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            time_num,
            num_layers,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout=0,
            device="cuda:0"
    ):
        super(STHGFormer, self).__init__()

        self.forward_expansion = forward_expansion
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            dropout
        )
        self.relu = nn.ReLU()
        self.Llinear = nn.Linear(in_features=embed_size, out_features=1)
        self.Llinear1 = nn.Linear(in_features=T_dim, out_features=output_T_dim)
        self.Tlinear = nn.Linear(in_features=embed_size, out_features=1)
        self.Tlinear1 = nn.Linear(in_features=T_dim, out_features=output_T_dim)

    def forward(self, x, w_t):
        # input x shape[ C, N, T]
        # C: num of channels  N:num of nodes T: num of Step
        input_w_t_r = w_t[:, 0, :]  # (B,2)
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        # input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.encoder(input_Transformer, self.forward_expansion, input_w_t_r)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        # split road segments and intersection turns
        L_out = output_Transformer[:, :, 0:277, :]
        T_out = output_Transformer[:, :, 277:546, :]
        L_out = self.relu(self.Llinear(L_out))
        T_out = self.relu(self.Tlinear(T_out))
        L_out = L_out.permute(0, 3, 2, 1)
        T_out = T_out.permute(0, 3, 2, 1)
        L_out = self.relu(self.Llinear1(L_out))
        T_out = self.relu(self.Tlinear1(T_out))
        out = torch.cat((L_out, T_out), dim=2)
        out = out.squeeze(1)
        return out  # [B, N, output_dim]






