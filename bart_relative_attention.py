
"""
Add relative attention to BART.

"""

# Stdlib
import math
import time
from typing import *

# Third party
import functorch as ft  # type: ignore[import]
try:
    import pretty_traceback  # type: ignore
    pretty_traceback.install()
except ImportError:
    pass
import numpy as np
import rich
import torch
import torch.amp
from torch import nn
from tqdm import tqdm  # type: ignore
import transformers.models.bart.modeling_bart as original

# First party
import general_shared_constants
import general_utils

def build_rel_attn_fn():
    def inter(q, k, r):
        assert k.shape == r.shape, (k.shape, r.shape)
        assert q.shape[0] == k.shape[1] == k.shape[1], (q.shape[0], k.shape[1], k.shape[1])
        assert q.ndim == 1, q.ndim

        s0 = k + r  # Both L2 x H
        output = s0 @ q  # Output is multiplied by q
        
        return output

    # (L1, H) -> H, (L2, H) -> (L2, H) and (L2, L1, H) -> L2, H
    a1 = ft.vmap(inter, (0, None, 1), 0) 
    # (B, L1, H) -> (L1, H), (B, L2, H) -> (L2, H) and (B, L1, L2, H) -> (L1, L2, H)
    rel_attn = ft.vmap(a1, (0, 0, 0), 0) 

    return rel_attn

class RelAttBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rel_attn_fn = build_rel_attn_fn()


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rel_att_keys: torch.Tensor,
        rel_att_values: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        """Input shape: Batch x Time x Channel"""


        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        assert not is_cross_attention, "relative positions only makes sense for self attention"

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        
        verbose = False
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # We're doing cross attention and have cached values
            # reuse k,v, cross_attentions
            if verbose:
                print(f"yes CROSS yes PAST: Num past: seq len {past_key_value.shape[2]}")
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # We're doing cross attention and don't have cached values
            # cross_attentions
            if verbose:
                print("yes CROSS not PAST")
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # We're not doing cross attention but we have cached values
            # reuse k, v, self_attention
            if verbose:
                print(f"not CROSS yes PAST. Num past: {past_key_value[0].shape[2]}")
            # If we're not doing cross attention we need to also
            # compute the key and value of the current state(s)
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # We're not doing cross attention and don't have cached values
            # self_attention
            if verbose:
                print("not CROSS not PAST")
            # We have all hidden states
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # bsz x num_heads x ? x head_dim
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) #  bsz x num_heads x ? x head_dim
        key_states = key_states.view(*proj_shape)  # bsz x num_heads x ? x head_dim
        value_states = value_states.view(*proj_shape)  # bsz x num_heads x ? x head_dim
        
        src_len = key_states.size(1)
        tgt_len = query_states.size(1)

        # Each head is seen as a different batch samples in a way (before they are remerged), a bit like beams often are in decoding
        # Both here should be grids with src_len x tgt_len. Each src_len attends to each of the tgt_len.
        
        
        ###################################################################################################
        # Rel Att
        ###################################################################################################
        REL_ATT_ON = True
        TORCH_VER = True
        tgt_len = query_states.shape[1]
        
        # print(rel_att_values.shape)

        if TORCH_VER:
            rel_att_keys   = rel_att_keys  .view(bsz * self.num_heads, src_len, -1, self.head_dim).type(query_states.dtype)  # type: ignore[operator]
            rel_att_values = rel_att_values.view(bsz * self.num_heads, src_len, -1, self.head_dim).type(query_states.dtype)  # type: ignore[operator]
            
            dummy_query_states = query_states.view(bsz * self.num_heads,       1, tgt_len, self.head_dim) 
            dummy_key_states   = key_states  .view(bsz * self.num_heads, src_len,       1, self.head_dim) 

            if REL_ATT_ON:
                dummy_key_states = dummy_key_states + rel_att_keys

            # We're just looking for the head_dim to disappear
            attn_weights = torch.einsum("ijkl, ijkl->ijk", (dummy_query_states, dummy_key_states)).transpose(1, 2)        
        else:
            rel_att_keys   = rel_att_keys  .view(bsz * self.num_heads, src_len, -1, self.head_dim).type(query_states.dtype)  # type: ignore[operator]
            rel_att_values = rel_att_values.view(bsz * self.num_heads, src_len, -1, self.head_dim).type(query_states.dtype)  # type: ignore[operator]

            if REL_ATT_ON:
                maybe_rel_att = rel_att_keys
            else:
                maybe_rel_att = torch.zeros_like(rel_att_keys)

            dummy_query_states = query_states.view(bsz * self.num_heads, tgt_len, self.head_dim) 
            dummy_key_states   = key_states  .view(bsz * self.num_heads, src_len, self.head_dim)

            assert dummy_query_states.shape[0] == dummy_key_states.shape[0] == maybe_rel_att.shape[0], (
                f"{dummy_query_states.shape = }", f"{dummy_key_states.shape = }", f"{maybe_rel_att.shape = }")
            assert dummy_query_states.shape[1] == maybe_rel_att.shape[2], (
                f"{dummy_query_states.shape = }", f"{maybe_rel_att.shape = }")
            assert dummy_key_states.shape[1] == maybe_rel_att.shape[1], (
                f"{dummy_key_states.shape = }", f"{maybe_rel_att.shape = }")
            assert dummy_query_states.shape[2] == dummy_key_states.shape[2] == maybe_rel_att.shape[3], (
                f"{dummy_query_states.shape = }", f"{dummy_key_states.shape = }", f"{maybe_rel_att.shape = }")

            attn_weights = self.rel_attn_fn(dummy_query_states, dummy_key_states, maybe_rel_att)

        if not REL_ATT_ON:
            orig_attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            rel_err = torch.allclose(orig_attn_weights, attn_weights, rtol=1/100)
            qty_err = torch.isclose(orig_attn_weights, attn_weights).float().mean() > 0.99
            assert rel_err or qty_err, "ref != b"

        ###################################################################################################

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)},"
                f" but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )


        ###################################################################################################
        # Rel Att
        ###################################################################################################
        dummy_value_states = value_states.view(bsz * self.num_heads, src_len, 1, self.head_dim)
        if REL_ATT_ON:
            dummy_value_states = dummy_value_states + rel_att_values
        attn_output = torch.einsum("byx, bxyh->byh", attn_probs, dummy_value_states)

        if not REL_ATT_ON:
            orig_attn_output = torch.bmm(attn_probs, value_states) # + rel_att_values.type(value_states.dtype)
            assert torch.allclose(orig_attn_output, attn_output), "ref != b"
        ###################################################################################################


        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def clamp(x: Union[float, int], min_: Union[float, int], max_: Union[float, int]):
    return min(max(x, min_), max_)

def _build_rel_att_mat_ref(
    attention_mask: torch.Tensor,
    num_embeddings: int,
):
    assert False
    batch_size = attention_mask.shape[0]
    src_len = attention_mask.shape[1]
    tgt_len = attention_mask.shape[2]

    relative_ids_test = torch.empty(
        batch_size, src_len, tgt_len, 
        dtype=torch.long
    )

    for batch_idx in range(batch_size):
        q_incr = 0
        for q_idx in range(src_len):
            if attention_mask[batch_idx, q_idx]:
                q_incr += 1

            k_incr = 0
            for k_idx in range(src_len):
                if attention_mask[batch_idx, k_idx]:
                    k_incr += 1

                relative_ids_test[batch_idx, q_idx, k_idx] = clamp(
                    k_incr - q_incr + num_embeddings // 2, 
                    0, num_embeddings - 1
                )
                    
    return relative_ids_test


def _build_rel_att_mat(
    attention_mask: torch.Tensor,
    num_embeddings: int,
    tgt_array_indices: Optional[torch.Tensor],
):
    """
        - In bidirectional setting, TGT == SRC
        - In masked self attention, with caching, SRC = Sequence decoded so far, TGT = current token
    """
    

    bsz = attention_mask.shape[0]
    src_len = attention_mask.shape[1]

    attention_mask_cs = attention_mask.cumsum(-1)

    key_idx = attention_mask_cs.reshape(bsz, src_len, 1)
    query_idx = attention_mask_cs

    if tgt_array_indices is None:
        # rich.print(f"[red]Internal step no past")
        tgt_len = src_len
        query_idx = query_idx.reshape(bsz, 1, src_len)  # Self attention, we do them all
        # general_utils.check_shape(query_idx.shape, (bsz, 1, src_len))
    else:
        # rich.print(f"[red]Internal step with past {tgt_array_indices}")
        tgt_len = tgt_array_indices.shape[1]
        assert tgt_array_indices.shape[0] == 1, tgt_array_indices.shape
        query_idx = query_idx[:, tgt_array_indices[0]].reshape(bsz, 1, tgt_len)
        # general_utils.check_shape(query_idx.shape, (bsz, 1, tgt_len))

    output = torch.clamp(key_idx - query_idx  + num_embeddings // 2, 0, num_embeddings - 1)

    general_utils.check_shape(output.shape, (bsz, src_len, tgt_len))
    # print(f"emb shape: {output.shape} {tgt_array_indices.shape if tgt_array_indices else None}")
    return output


class RelAttBartEncoderLayer(nn.Module):
    def __init__(self, config: original.BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = RelAttBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = original.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        rel_att_keys: torch.Tensor,
        rel_att_values: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            rel_att_keys=rel_att_keys,
            rel_att_values=rel_att_values,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class RelPosEmbs(nn.Module):
    def __init__(self, model_d: int, num_embeddings: int, mode: str):
        super().__init__()
        assert mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs, (
            "RelPosEmbs Should not be initialized in this case"
        )
        assert mode in general_shared_constants.RelPosEmbsChoices.__choices__, (
            f"mode {mode} not in {general_shared_constants.RelPosEmbsChoices.__choices__}"
        )

        self.mode = mode
        self.num_embeddings = num_embeddings

        if mode == general_shared_constants.RelPosEmbsChoices.two_embedders:
            self.positional_embeddings_k = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=model_d
            )
            self.positional_embeddings_v = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=model_d, 
            )
        elif mode == general_shared_constants.RelPosEmbsChoices.two_embedders:
            self.positional_embeddings = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=model_d
            )
            self.positional_embeddings_linear_k = nn.Linear(model_d, model_d)
            self.positional_embeddings_linear_v = nn.Linear(model_d, model_d)
        else:
            raise ValueError(f"mode must be 1 or 2, got {mode}")

    def forward(
        self, attention_mask: torch.LongTensor, tgt_array_indices: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(attention_mask.shape) == 2, len(attention_mask.shape)

        positions = _build_rel_att_mat(
            attention_mask=attention_mask, 
            num_embeddings=self.num_embeddings, 
            tgt_array_indices=tgt_array_indices,
        )
        
        if self.mode == general_shared_constants.RelPosEmbsChoices.two_embedders:
            assert self.positional_embeddings_k.weight.device == self.positional_embeddings_v.weight.device

            positions = positions.to(self.positional_embeddings_k.weight.device)
            k = self.positional_embeddings_k(positions)
            v = self.positional_embeddings_v(positions)
            return k, v
        if self.mode == general_shared_constants.RelPosEmbsChoices.one_embedder:
            positions = positions.to(self.positional_embeddings.weight.device)
            shared = self.positional_embeddings(positions)
            k = self.positional_embeddings_linear_k(shared)
            v = self.positional_embeddings_linear_v(shared)
            return k, v
        else:
            raise ValueError("mode must be 1 or 2")


class RelAttBartDecoderLayer(nn.Module):
    def __init__(self, config: original.BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = RelAttBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = original.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = original.BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_att_keys: torch.Tensor,
        rel_att_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rel_att_keys=rel_att_keys,
            rel_att_values=rel_att_values,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

if __name__ == "__main__":
    NUM_EMBEDDINGS = 64
    SEQ_LEN = 190
    BATCH_SIZE = 256
    PROB = 0.5
    N = 100

    dims = (BATCH_SIZE, SEQ_LEN)
    

    def build_args():
        attention_mask = (torch.rand(*dims) < PROB).cuda()
        return dict(
            attention_mask=attention_mask,
            num_embeddings=NUM_EMBEDDINGS,
        )

    speeds_not_jitted = []
    for i in range(N):
        args = build_args()
        start = time.perf_counter()
        res_test = _build_rel_att_mat(**args)
        speeds_not_jitted.append(time.perf_counter() - start)
    print(f"Not jitted: {np.mean(speeds_not_jitted)}")

    jitted = torch.jit.script(_build_rel_att_mat)
    speeds_jitted = []
    for i in range(N):
        args = build_args()
        start = time.perf_counter()
        res_test = jitted(**args)
        speeds_jitted.append(time.perf_counter() - start)
    print(f"Jitted: {np.mean(speeds_jitted)}")


    start = time.perf_counter()
    res_ref = _build_rel_att_mat_ref(**args)
    print("ref", time.perf_counter() - start)
    assert torch.allclose(res_ref, res_test)
