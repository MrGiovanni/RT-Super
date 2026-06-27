# 3D version of Transformer decoder; Copyright Johns Hopkins University
#  Modified from Mask2former

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from fvcore.common.registry import Registry

from .position_encoding import PositionEmbeddingSine
from torch.cuda.amp import autocast

from torch.utils.checkpoint import checkpoint


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0] 
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.ls1 = nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]  
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]

        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(self.ls1(tgt2))

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, use_layer_scale=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(self.ls1(tgt))
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(self.ls1(tgt2))
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
from typing import List, Optional, Tuple
import torch
from torch import nn


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder3d(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
        self,
        #R-Super additions
        feature_in_channels: List[int], #number of channels for each feature level we will cross attend to
        kernel_channels: List[Tuple[int, int]], # channel in - channel out for each conv layer we will create
        num_classes_segmentation: int, 
        report_token_dims = [1,1,1,10,10,30,10], #tumor_organ_name, tumor_count, known_tumor_count, tumor_attenuation, tumor_malignancy, tumor_diameters, tumor_volumes
        use_report = True, 
        max_spatial_size = 32, #gives you a resolution of 4 mm if input is 128x128x128. This already catches any tumor in our 130K dataset.
        #previous standards
        hidden_dim = 264,
        nheads = 8,
        dim_feedforward = 2048,
        dec_layers = 8,
        pre_norm = False,
        enforce_input_project = False,
        is_mhsa_float32 = False,
        no_max_hw_pe = False, 
        use_layer_scale = False, 
        generate_kernels = 'all',
        #this is to generate 3x3x3 conv
        use_conv3_bank = False,
		conv3_num_experts = 16,
		conv3_temperature = 1.0,
	    conv3_gate_dropout = 0.1,
		conv3_use_noisy_gating = False,
		conv3_noisy_gating_std = 1.0,
		conv3_init_scale = 0.02,
        level_modes = None, #list of strings, "conv3", "conv1", "both", "none" per level
        generate_kernels_t_3x3 = None, #list of bool per level, whether to generate 3x3x3 kernels or not
        longitudinal = False,
        kernels_3x3 = None
    ):
        super().__init__()
        
        self.use_conv3_bank = bool(use_conv3_bank)
        self.conv3_num_experts = int(conv3_num_experts)
        self.conv3_temperature = float(conv3_temperature)
        self.conv3_gate_dropout = float(conv3_gate_dropout)
        self.conv3_use_noisy_gating = bool(conv3_use_noisy_gating)
        self.conv3_noisy_gating_std = float(conv3_noisy_gating_std)
        self.conv3_init_scale = float(conv3_init_scale)
        self.longitudinal = bool(longitudinal)
        
        
        if isinstance(generate_kernels,str):
            assert generate_kernels == 'all', "generate_kernels string option only supports 'all' currently"
            generate_kernels = [True]*len(feature_in_channels)
        else:
            #assert list of bool with the same size as feature_in_channels
            assert isinstance(generate_kernels,list), "generate_kernels must be a string or a list of bools"
            assert len(generate_kernels) == len(feature_in_channels), "generate_kernels list must have the same length as feature_in_channels"
            assert all(isinstance(x, bool) for x in generate_kernels), "generate_kernels list must contain only bools"
        
        num_feature_levels = len(kernel_channels)
        self.num_feature_levels = num_feature_levels
        
        if self.use_conv3_bank:
            #generate_kernels_t_3x3 is mandatory if we want to generate 3x3x3 kernels
            assert generate_kernels_t_3x3 is not None
            assert len(generate_kernels_t_3x3) == self.num_feature_levels
            assert all(isinstance(x, bool) for x in generate_kernels_t_3x3)
        
        if level_modes is None:
            # backward compatible default: everything produces conv1 (and conv3 if enabled)
            level_modes = ["both"] * num_feature_levels
        assert len(level_modes) == num_feature_levels
        assert all(m in ("conv3","conv1","both","none") for m in level_modes)
        self.level_modes = level_modes
        
        assert len(feature_in_channels) == num_feature_levels, f"feature_in_channels has {len(feature_in_channels)} levels but kernel_channels has {num_feature_levels}, we want to create kernel for each feature level"
        
        self.no_max_hw_pe = no_max_hw_pe
        
        out_channels_per_convolution = [out_c for (_, out_c) in kernel_channels]
        in_channels_per_convolution = [in_c for (in_c, _) in kernel_channels]

        # positional encoding
        assert hidden_dim % 6 == 0, f'hidden_dim should be divisible by 6, got {hidden_dim}'
        N_steps = hidden_dim // 3
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        if use_report:
            self.transformer_cross_attention_report_layers = nn.ModuleList()
        if self.longitudinal:
            self.transformer_cross_attention_layers_other_time = nn.ModuleList()
            if use_report:
                self.transformer_cross_attention_report_layers_other_time = nn.ModuleList()

        #asser that num_layers is divisible by num_feature_levels
        assert self.num_layers%num_feature_levels == 0, "num_layers must be divisible by num_feature_levels"
        self.layers_per_level = []
        #slice layer indices for each feature level
        layers_per_level = self.num_layers // num_feature_levels
        for i in range(num_feature_levels):
            self.layers_per_level.append(slice(i*layers_per_level, (i+1)*layers_per_level))
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )
            
            if self.longitudinal:
                self.transformer_cross_attention_layers_other_time.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                        is_mhsa_float32=is_mhsa_float32,
                        use_layer_scale=use_layer_scale,
                    )
                )
             
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )
            
            if use_report:
                self.transformer_cross_attention_report_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=max(2, nheads//4), # use fewer heads for report cross attention
                        dropout=0.0,
                        normalize_before=pre_norm,
                        is_mhsa_float32=is_mhsa_float32,
                        use_layer_scale=use_layer_scale,
                    )
                )
                if self.longitudinal:
                    self.transformer_cross_attention_report_layers_other_time.append(
                        CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=max(2, nheads//4),
                            dropout=0.0,
                            normalize_before=pre_norm,
                            is_mhsa_float32=is_mhsa_float32,
                            use_layer_scale=use_layer_scale,
                        )
                    )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    use_layer_scale=use_layer_scale,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = int(sum(out_channels_per_convolution))+1#include an extra memory token to help carry information across levels
        if use_conv3_bank:
            self.num_queries += num_feature_levels
        
        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        # learnable query p.e.
        self.query_positional = nn.Embedding(self.num_queries, hidden_dim)

        self.num_feature_levels = num_feature_levels 
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        if self.longitudinal:
            # one vector per feature level, used ONLY for other_time memories
            self.other_time_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            nn.init.normal_(self.other_time_level_embed.weight, mean=0.0, std=0.02)
        self.query_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.kernel_heads = nn.ModuleList()
        self.bias_heads = nn.ModuleList()
        
        for i in range(self.num_feature_levels):
            if feature_in_channels[i] != hidden_dim or enforce_input_project:
                proj = nn.Conv3d(feature_in_channels[i], hidden_dim, kernel_size=1)
                self.input_proj.append(proj)
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Identity())
            if generate_kernels[i]:
                self.kernel_heads.append(MLP(hidden_dim, hidden_dim, in_channels_per_convolution[i], 3))
                self.bias_heads.append(MLP(hidden_dim, hidden_dim, 1, 3))
            else:
                #dummy heads that will not be used
                self.kernel_heads.append(nn.Identity())
                self.bias_heads.append(nn.Identity())
                
        
        self.use_report = use_report
        if self.use_report:
            # ---- report tokens (heterogeneous dims, fixed count) ----
            self.report_token_dims = list(report_token_dims)     # e.g. [D0, D1, ...]
            self.num_report_tokens = len(self.report_token_dims)

            self.report_token_proj = nn.ModuleList([
                nn.Linear(d, hidden_dim, bias=True) for d in self.report_token_dims
            ])
            self.report_token_norm = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(self.num_report_tokens)
            ])

            # token identity for each report token type
            self.report_token_type_embed = nn.Parameter(torch.zeros(self.num_report_tokens, hidden_dim))
            nn.init.normal_(self.report_token_type_embed, mean=0.0, std=0.02)

            # ---- mask tokens (one per mask channel) ----
            self.num_mask_channels = int(num_classes_segmentation)  # Cm

            # project scalar mask avg (1 value) -> hidden_dim (shared across channels)
            self.mask_scalar_proj = nn.Linear(1, hidden_dim, bias=True)
            self.mask_scalar_norm = nn.LayerNorm(hidden_dim)

            # channel identity embedding for each mask channel
            self.mask_channel_embed = nn.Parameter(torch.zeros(self.num_mask_channels, hidden_dim))
            nn.init.normal_(self.mask_channel_embed, mean=0.0, std=0.02)
            
            #positional embedding for report tokens and mask tokens
            self.report_mask_positional = nn.Embedding(self.num_report_tokens + self.num_mask_channels, hidden_dim)
            

        self.kernel_channels = list(kernel_channels)
        self.out_channels_per_convolution = list(out_channels_per_convolution)
        self.in_channels_per_convolution = list(in_channels_per_convolution)
        self.feature_in_channels = list(feature_in_channels)
        self.queries_per_level = []
        start = 0
        for cout in out_channels_per_convolution:
            end = start + int(cout)
            self.queries_per_level.append(slice(start, end))
            start = end
        self.max_spatial_size = max_spatial_size # we will limit the spatial size to this. If larger, we will reduce with avgpooling
        # ---- precompute level id per query (excluding memory token) ----
        Q_seg = int(sum(out_channels_per_convolution))
        Q_bank = int(self.num_feature_levels) if self.use_conv3_bank else 0
        self.Q_seg = Q_seg
        self.Q_bank = Q_bank
        Q_wo_mem = self.num_queries - 1  # exclude memory token
        level_ids = torch.empty(Q_wo_mem, dtype=torch.long)

        for lvl, sl in enumerate(self.queries_per_level):
            level_ids[sl] = lvl
            
        # bank queries: one per level, appended after seg queries
        if self.use_conv3_bank:
            self.bank_query_index = []  # list[int], length=num_feature_levels
            for lvl in range(self.num_feature_levels):
                qi = self.Q_seg + lvl
                self.bank_query_index.append(qi)
                level_ids[qi] = lvl
        else:
            self.bank_query_index = None
            

        # register as buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer("query_level_ids", level_ids, persistent=True)

        # Optional sanity check
        assert self.query_level_ids.numel() == self.num_queries - 1, \
            f"query_level_ids has {self.query_level_ids.numel()} but expected {self.num_queries - 1}"
        
        self.memory_query_level = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.memory_query_level, std=0.02)  
        
        # ---- NEW: conv3 expert banks + gate heads (per level) ----
        if self.use_conv3_bank:
            K = self.conv3_num_experts
            self.conv3_weight_bank = nn.ParameterList()
            self.conv3_bias_bank = nn.ParameterList()
            self.conv3_gate_heads = nn.ModuleList()
            self.conv3_logits_dropout = nn.Dropout(self.conv3_gate_dropout) if self.conv3_gate_dropout > 0 else nn.Identity()
            self.conv3_level_to_bank_idx = [-1] * self.num_feature_levels

            bank_idx = 0
            for lvl in range(self.num_feature_levels):    
                if generate_kernels_t_3x3[lvl]:
                    if kernels_3x3 is None:
                        Cin3 = int(feature_in_channels[lvl])
                        Cout3 = Cin3
                    else:
                        Cin3, Cout3 = kernels_3x3[lvl]
                    self.conv3_level_to_bank_idx[lvl] = bank_idx

                    w = nn.Parameter(torch.empty(K, Cout3, Cin3, 3, 3, 3))
                    b = nn.Parameter(torch.empty(K, Cout3))
                    nn.init.normal_(w, mean=0.0, std=self.conv3_init_scale)
                    nn.init.zeros_(b)

                    self.conv3_weight_bank.append(w)
                    self.conv3_bias_bank.append(b)
                    self.conv3_gate_heads.append(MLP(hidden_dim, hidden_dim, K, 3))
                    bank_idx += 1
                else:
                    # no params registered for this level
                    self.conv3_gate_heads.append(nn.Identity())
                    
        
    def forward_conv3_bank_heads(self, bank_query: torch.Tensor, level: int):
        """
        bank_query: [1, B, C] (a single query token)
        Returns:
          conv3_kernel: [B, Cout, Cin, 3,3,3]
          conv3_bias:   [B, Cout]
        """
        if isinstance(self.conv3_gate_heads[level], nn.Identity):
            raise ValueError(f'No conv3 head for level {level} is Identity, cannot generate kernels. Check the generate_kernels_t_3x3x3 argument during initialization.')
        
        if not self.use_conv3_bank:
            return None, None

        # normalize like other heads
        z = self.decoder_norm(bank_query)      # [1,B,C]
        z = z.transpose(0, 1).squeeze(1)       # [B,C]

        logits = self.conv3_gate_heads[level](z)  # [B, K]
        logits = self.conv3_logits_dropout(logits)

        if self.conv3_use_noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * self.conv3_noisy_gating_std

        gates = F.softmax(logits / self.conv3_temperature, dim=1)  # [B,K]

        bank_idx = self.conv3_level_to_bank_idx[level]
        if bank_idx == -1:
            raise ValueError(f"No conv3 bank defined for level {level}, cannot generate 3x3x3 kernels.")
        Wbank = self.conv3_weight_bank[bank_idx]  # [K,Cout,Cin,3,3,3]
        Bbank = self.conv3_bias_bank[bank_idx]    # [K,Cout]

        conv3_kernel = torch.einsum("bk,kocxyz->bocxyz", gates, Wbank)
        conv3_bias = torch.einsum("bk,ko->bo", gates, Bbank)

        return conv3_kernel, conv3_bias
        
    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["hidden_dim"] = cfg["hidden_dim"]
        ret["nheads"] = cfg["nheads"]
        ret["dim_feedforward"] = cfg["dim_feedforward"]
        ret["dec_layers"] = cfg["dec_layers"] - 1
        ret["pre_norm"] = cfg["pre_norm"]
        ret["enforce_input_project"] = cfg["enforce_input_project"]
        ret["is_mhsa_float32"] = cfg.get("is_mhsa_float32", False)
        ret["no_max_hw_pe"] = cfg.get("no_max_hw_pe", False)
        ret["use_layer_scale"] = cfg.get("use_layer_scale", False)
        # ---- R-Super additions ----
        ret["use_report"] = cfg.get("use_report", False)
        ret["num_classes_segmentation"] = cfg.get("num_classes_segmentation", 0)
        ret["report_token_dims"] = cfg.get("report_token_dims", None)
        ret["feature_in_channels"] = cfg["feature_in_channels"]
        ret["kernel_channels"] = cfg["kernel_channels"]
        ret["max_spatial_size"] = cfg.get("max_spatial_size", 32)

        return ret
    
    def _zero_touch_modulelist_params(self, modules: nn.ModuleList, layer_slice: slice) -> torch.Tensor:
        """
        Returns a scalar tensor connected to the parameters of modules[layer_slice],
        but multiplied by 0 so it does not change activations.
        This forces DDP to see these params as 'used' (grad will be zero, not None).
        """
        z = None
        for i in range(layer_slice.start, layer_slice.stop):
            m = modules[i]
            # touch *one element* per parameter to keep overhead tiny
            for p in m.parameters(recurse=True):
                if p is None:
                    continue
                t = p.view(-1)[0]
                z = t if z is None else (z + t)
        if z is None:
            # must return a tensor on correct device/dtype; use a dummy from a known param
            return torch.zeros((), device=self.query_feat.weight.device, dtype=self.query_feat.weight.dtype)
        return z * 0.0

    def forward(self, x, report_tokens_list, output_mask_avg_pool,level_index, previous_queries=None,
                use_checkpoint=True,layer_to_run='both', 
                generate_1x1x1_kernel=True, generate_3x3x3_kernel=False,
                other_time_report_tokens_list=None, other_time_x=None,other_time_output_mask_avg_pool=None):
        #x: features, tensor
        
        D, H, W = x.shape[-3:]
        kD = max(1, (D + self.max_spatial_size - 1) // self.max_spatial_size)
        kH = max(1, (H + self.max_spatial_size - 1) // self.max_spatial_size)
        kW = max(1, (W + self.max_spatial_size - 1) // self.max_spatial_size)
        
        if kD > 1.2 or kH > 1.2 or kW > 1.2:
            x = F.avg_pool3d(x, kernel_size=(kD,kH,kW), stride=(kD,kH,kW))
            if self.longitudinal:
                other_time_x = F.avg_pool3d(other_time_x, kernel_size=(kD,kH,kW), stride=(kD,kH,kW))
                #assert shapes match
                assert x.shape == other_time_x.shape, f"After avg pooling, x has shape {x.shape} but other_time_x has shape {other_time_x.shape}, they should match"
        
        pos = self.pe_layer(x, None).flatten(2)
        src = self.input_proj[level_index](x).flatten(2) + self.level_embed.weight[level_index][None, :, None]
        if self.longitudinal:
            other_src = self.input_proj[level_index](other_time_x).flatten(2) + self.level_embed.weight[level_index][None, :, None] + self.other_time_level_embed.weight[level_index][None, :, None]
            other_src = other_src.permute(2,0,1)
        # flatten NxCxHxW to HWxNxC
        pos = pos.permute(2, 0, 1)
        src = src.permute(2, 0, 1)

        _, bs, _ = src.shape

        query_positional = self.query_positional.weight.unsqueeze(1).repeat(1, bs, 1)

        # ---- per-query level embedding (vectorized) ----
        # [Q-1, C]
        lvl_embed_wo_mem = self.query_level_embed(self.query_level_ids)  # embedding lookup
        # [Q-1, B, C]
        lvl_embed_wo_mem = lvl_embed_wo_mem.unsqueeze(1).expand(-1, bs, -1)

        # memory token embedding: [1, B, C]
        mem_lvl = self.memory_query_level.view(1, 1, -1).expand(1, bs, -1)

        # [Q, B, C]
        level_positional_embeds = torch.cat([lvl_embed_wo_mem, mem_lvl], dim=0)

        # add to positional and to values (your design)
        query_positional = query_positional + level_positional_embeds

        if level_index == 0:
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            output = output + level_positional_embeds
        else:
            assert previous_queries is not None
            assert previous_queries.shape == (self.num_queries, bs, self.query_feat.embedding_dim)      
            output = previous_queries
        
        #we will process only the queries at our current level (for the current layer) and the ones after it
        start_L = self.queries_per_level[level_index].start
        active = slice(start_L, self.num_queries)
        out_active = output[active]
        qpos_active = query_positional[active]
        
        if self.use_report:
            #get positional embeddings for report and mask tokens
            report_mask_pos_embed = self.report_mask_positional.weight.unsqueeze(1).repeat(1, bs, 1)
            #concatenate report embedding with avgpool of segmentation probabilities
            assert report_tokens_list is not None, "report_tokens_list is None but use_report=True"
            assert output_mask_avg_pool is not None, "output_mask_avg_pool is None but use_report=True"
            report_memory = self.build_report_and_mask_memory(report_tokens_list, output_mask_avg_pool)  # [S,B,C]
            if self.longitudinal:
                other_report_memory = self.build_report_and_mask_memory(other_time_report_tokens_list, other_time_output_mask_avg_pool)  # [S,B,C]   
                other_report_memory = other_report_memory + self.other_time_level_embed.weight[level_index].view(1, 1, -1)  

        layers = self.layers_per_level[level_index]
        assert layer_to_run in ['both','first_half','second_half'], "layer_to_run must be 'both', 'first_half' or 'second_half'"
        if layer_to_run == 'first_half':
            assert (layers.stop - layers.start) % 2 == 0, "Cannot run first_half when number of layers is odd"
            layers = slice(layers.start, layers.start + (layers.stop - layers.start)//2)
        elif layer_to_run == 'second_half':
            assert (layers.stop - layers.start) % 2 == 0, "Cannot run second_half when number of layers is odd"
            layers = slice(layers.start + (layers.stop - layers.start)//2, layers.stop)
        for layer_idx in range(layers.start, layers.stop):
            if use_checkpoint:
                if self.use_report and self.longitudinal:
                    #print(f'Running longitudinal cross attention, Report token list is {other_time_report_tokens_list}', flush=True)
                    out_active = checkpoint(
                        lambda oa, rm, rpos, qpos: self.transformer_cross_attention_report_layers_other_time[layer_idx](
                            oa, rm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=rpos,
                            query_pos=qpos,
                        ),
                        out_active, other_report_memory, report_mask_pos_embed, qpos_active,
                        use_reentrant=False
                    )
                
                if self.use_report:
                    out_active = checkpoint(
                        lambda oa, rm, rpos, qpos: self.transformer_cross_attention_report_layers[layer_idx](
                            oa, rm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=rpos,
                            query_pos=qpos,
                        ),
                        out_active, report_memory, report_mask_pos_embed, qpos_active,
                        use_reentrant=False
                    )
                    
                if self.longitudinal:
                    #print(f'Running longitudinal cross attention, features are {other_src.mean()}', flush=True)
                    out_active = checkpoint(
                        lambda oa, s, p, qpos: self.transformer_cross_attention_layers_other_time[layer_idx](
                            oa, s,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=p,
                            query_pos=qpos,
                        ),
                        out_active, other_src, (None if self.no_max_hw_pe else pos), qpos_active,
                        use_reentrant=False
                    )


                # BIGGEST MEMORY SAVER: cross-attention to src
                out_active = checkpoint(
                    lambda oa, s, p, qpos: self.transformer_cross_attention_layers[layer_idx](
                        oa, s,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=p,
                        query_pos=qpos,
                    ),
                    out_active, src, (None if self.no_max_hw_pe else pos), qpos_active,
                    use_reentrant=False
                )

                # self-attn
                out_active = checkpoint(
                    lambda oa, qpos: self.transformer_self_attention_layers[layer_idx](
                        oa,
                        tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=qpos,
                    ),
                    out_active, qpos_active,
                    use_reentrant=False
                )

                # ffn
                out_active = checkpoint(
                    lambda oa: self.transformer_ffn_layers[layer_idx](oa),
                    out_active,
                    use_reentrant=False
                )
            
            else:
                if self.use_report and self.longitudinal:
                    #print(f'Running longitudinal cross attention, Report token list is {other_time_report_tokens_list}', flush=True)
                    out_active = self.transformer_cross_attention_report_layers_other_time[layer_idx](
                        out_active, other_report_memory,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=report_mask_pos_embed,
                        query_pos=qpos_active
                    )
                
                if self.use_report:
                    out_active = self.transformer_cross_attention_report_layers[layer_idx](
                    out_active, report_memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,  # fixed S, no padding
                    pos=report_mask_pos_embed,
                    query_pos=qpos_active
                )
                    
                if self.longitudinal:
                    #print(f'Running longitudinal cross attention, features are {other_src.mean()}', flush=True)
                    out_active = self.transformer_cross_attention_layers_other_time[layer_idx](
                        out_active, other_src,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=None if self.no_max_hw_pe else pos,
                        query_pos=qpos_active
                    )

                out_active = self.transformer_cross_attention_layers[layer_idx](
                    out_active, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None if self.no_max_hw_pe else pos, query_pos=qpos_active
                )

                out_active = self.transformer_self_attention_layers[layer_idx](
                    out_active, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=qpos_active
                )
                
                # FFN
                out_active = self.transformer_ffn_layers[layer_idx](
                    out_active
                )
                  
            
        queries = torch.cat([output[:start_L], out_active], dim=0)#update only queries for current and future layers
        
        #get kernels and biases for this level
        level_queries = queries[self.queries_per_level[level_index], :, :]  # select only queries for this level  
        
        if not generate_3x3x3_kernel and not generate_1x1x1_kernel:
            return queries
        
        if generate_1x1x1_kernel:
            #1x1x1 conv kernels and biases
            kernel,bias = self.forward_prediction_heads(level_queries, level_index)
        
        # 3x3 kernels from bank 
        if generate_3x3x3_kernel:
            if not self.use_conv3_bank:
                raise ValueError("generate_3x3x3_kernel is True but use_conv3_bank is False")
            if generate_1x1x1_kernel:
                raise ValueError("Cannot generate both 1x1x1 and 3x3x3 kernels at the same time")   
            bank_qi = self.bank_query_index[level_index]
            bank_query = queries[bank_qi:bank_qi+1, :, :]  # [1,B,C]
            kernel,bias = self.forward_conv3_bank_heads(bank_query, level_index)
        
        return queries, kernel, bias

    def forward_prediction_heads(self, output, level):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        #check if the head is Identity (we may have disabled some heads)
        if isinstance(self.kernel_heads[level], nn.Identity) or isinstance(self.bias_heads[level], nn.Identity):
            raise ValueError(f"Kernel or bias head for level {level} is Identity, cannot generate kernels. Check the generate_kernels argument during initialization.")

        kernel = self.kernel_heads[level](decoder_output)
        bias = self.bias_heads[level](decoder_output)
        
        bias = bias.squeeze(-1)
        kernel = kernel.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return kernel, bias
        
    
    def build_report_and_mask_memory(
        self,
        report_tokens_list: list,          # length N_report, each is [B, D_i]
        output_mask_avg_pool: torch.Tensor # [B, Cm]
    ) -> torch.Tensor:
        """
        Returns:
            memory: [S, B, hidden_dim] where S = N_report + Cm
        """
        if len(report_tokens_list) != self.num_report_tokens:
            raise ValueError(
                f"Expected {self.num_report_tokens} report tokens, got {len(report_tokens_list)}"
            )
        if output_mask_avg_pool.ndim != 2:
            raise ValueError(f"output_mask_avg_pool must be [B,Cm], got {tuple(output_mask_avg_pool.shape)}")
        if output_mask_avg_pool.shape[1] != self.num_mask_channels:
            raise ValueError(
                f"output_mask_avg_pool channel mismatch: expected Cm={self.num_mask_channels}, got {output_mask_avg_pool.shape[1]}"
            )

        B = output_mask_avg_pool.shape[0]

        # ---- report tokens: [N_report, B, hidden_dim] ----
        report_tokens = []
        for i, (t, proj, norm) in enumerate(zip(report_tokens_list, self.report_token_proj, self.report_token_norm)):
            if t.ndim != 2:
                raise ValueError(f"report token {i} must be [B, D_i], got {tuple(t.shape)}")
            if t.shape[0] != B:
                raise ValueError(f"Batch mismatch in report token {i}: expected B={B}, got {t.shape[0]}")
            if t.shape[1] != self.report_token_dims[i]:
                raise ValueError(
                    f"report token {i} dim mismatch: expected {self.report_token_dims[i]}, got {t.shape[1]}"
                )

            z = proj(t)                         # [B, hidden_dim]
            z = norm(z)                         # [B, hidden_dim]
            z = z + self.report_token_type_embed[i].unsqueeze(0)  # [B, hidden_dim]
            report_tokens.append(z)

        report_mem = torch.stack(report_tokens, dim=0)  # [N_report, B, hidden_dim]

        # ---- mask tokens: one per channel -> [Cm, B, hidden_dim] ----
        # scalars per channel: [B, Cm] -> [Cm, B, 1]
        m = output_mask_avg_pool.transpose(0, 1).unsqueeze(-1)     # [Cm, B, 1]
        m = self.mask_scalar_proj(m)                               # [Cm, B, hidden_dim]
        m = self.mask_scalar_norm(m)                               # [Cm, B, hidden_dim]
        m = m + self.mask_channel_embed.unsqueeze(1)               # [Cm, 1, hidden_dim] broadcast over B

        # concat along token axis S
        memory = torch.cat([report_mem, m], dim=0)                 # [N_report+Cm, B, hidden_dim]
        return memory
        
        
        
        
        
                
            