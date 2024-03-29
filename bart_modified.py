# coding=utf-8
########################################################################################################################
#
# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
# Modified so that the positional embeddings of the decoder take decoder_attention_mask into account.
#
########################################################################################################################
#
# Original comments:
# ======================================================================================================================
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################################################################

""" PyTorch BART model."""
from dataclasses import dataclass
import logging
import math
import random
from typing import *

import pretty_traceback  # type: ignore
pretty_traceback.install()
import rich
import torch
from torch import nn
import transformers.models.bart.modeling_bart as original

import bart_relative_attention
import general_shared_constants

logger = logging.getLogger(__name__)


@torch.jit.script
def _compute_abs_position_ids(
    position_ids: Optional[torch.Tensor], past_key_values_length: int, attention_mask: torch.Tensor
) -> torch.Tensor:
    # In order to work with cache (which is when past_key_values_length is not None), we need to
    # have position_ids given to us because we can't compute them.
    if past_key_values_length != 0:
        assert position_ids is not None

    # If we don't have position_ids, we need to compute them
    if position_ids is None:
        assert past_key_values_length == 0, past_key_values_length
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)  # Prevent negative positions
    
    return position_ids


class FixedPositionalEmbedding(nn.Module):
    """From TransformerXL
    https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15
    """
    def __init__(self, demb):
        super().__init__()
        rich.print("[blue]BUILT FIXED POS EMBS")
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0., demb, 2.) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, 
        *,
        past_key_values_length: int, 
        attention_mask: torch.Tensor, 
        position_ids: Optional[torch.LongTensor], 
    ):
        assert attention_mask is not None
        position_ids = _compute_abs_position_ids(position_ids, past_key_values_length, attention_mask)

        sinusoid_inp = torch.einsum("bi,bj->bij", position_ids, self.inv_freq.reshape(1, -1))  # type: ignore[operator]
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb


class ModifiedBartLearnedPositionalEmbedding(nn.Embedding):
    """
    Only modified to take decoder_attention_mask into account.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(  # type: ignore[override]
        self,
        *,
        past_key_values_length: int,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:  
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # All of this is changed
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We should always have an attention mask at this point.
        assert attention_mask is not None

        position_ids = _compute_abs_position_ids(position_ids, past_key_values_length, attention_mask)

        # If we already had position_ids, they would also be use here.
        assert isinstance(position_ids, torch.Tensor), type(position_ids)
        output = super().forward(position_ids + self.offset)
        return output

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), float("-inf"))


class ModifiedBartEncoder(original.BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self, 
        *,
        config: original.BartConfig, 
        abs_pos_embs_mode: str,
        rel_pos_embs_mode: str,
        num_rel_pos_embs: int,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)

        assert abs_pos_embs_mode in general_shared_constants.AbsPosEmbsModes.__choices__, abs_pos_embs_mode

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.abs_pos_embs_mode = abs_pos_embs_mode
        self.rel_pos_embs_mode = rel_pos_embs_mode
        self.num_rel_pos_embs = num_rel_pos_embs

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Name changed.
        #     Added fixed positional embeddings.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert abs_pos_embs_mode

        self.abs_embed_positions : Union[
            FixedPositionalEmbedding, ModifiedBartLearnedPositionalEmbedding, None
        ]
        if abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.fixed_pos_embs:
            self.abs_embed_positions = FixedPositionalEmbedding(
                config.d_model,
            )
        elif abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.learned_pos_embs:
            self.abs_embed_positions = ModifiedBartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
            )
        elif abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.no_abs_pos_embs:
            self.abs_embed_positions = None

        if rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
            self.layers = nn.ModuleList(
                [bart_relative_attention.RelAttBartEncoderLayer(config) for _ in range(config.encoder_layers)]
            )
            self.rel_pos_embs = bart_relative_attention.RelPosEmbs(
                model_d=config.d_model, 
                mode=rel_pos_embs_mode,
                num_embeddings=num_rel_pos_embs,
            )

        else:
            self.layers = nn.ModuleList(
                [original.BartEncoderLayer(config) for _ in range(config.encoder_layers)]
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        *,
        tgt_array_indices,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.abs_embed_positions:
            abs_embed_pos = self.abs_embed_positions(
                past_key_values_length=0, # No past
                attention_mask=attention_mask,
                position_ids=None, # No previous pos ids
            )
        else:
            abs_embed_pos = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Modified by us
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
            rel_att_keys, rel_att_values = self.rel_pos_embs(
                attention_mask=attention_mask,
                tgt_array_indices=tgt_array_indices,
            )
            assert rel_att_keys.dtype == inputs_embeds.dtype, rel_att_keys.dtype
            assert rel_att_values.dtype == inputs_embeds.dtype, rel_att_values.dtype
            assert rel_att_keys.device == inputs_embeds.device, rel_att_keys.device
            assert rel_att_values.device == inputs_embeds.device, rel_att_values.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        hidden_states = inputs_embeds
        if abs_embed_pos is not None:
            hidden_states += abs_embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    assert False, "Gradient checkpointing is not supported with rel pos embs"
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    encoder_layer_args = dict(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                    if self.rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
                        encoder_layer_args["rel_att_keys"] = rel_att_keys
                        encoder_layer_args["rel_att_values"] = rel_att_values
                
                    layer_outputs = encoder_layer(
                        **encoder_layer_args
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return original.BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class ModifiedBartDecoder(original.BartDecoder):
    """

    Only modified to:
    - use ModifiedBartLearnedPositionalEmbedding in `__init__`.
    - pass it decoder_attention_mask in `forward`.

    """

    def __init__(
        self, 
        *,
        config: original.BartConfig, 
        abs_pos_embs_mode: str,
        rel_pos_embs_mode: str,
        num_rel_pos_embs: int,
        embed_tokens: Optional[nn.Embedding],
    ):
        original.BartPretrainedModel.__init__(self, config)

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.rel_pos_embs_mode = rel_pos_embs_mode
        self.num_rel_pos_embs = num_rel_pos_embs

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.d_model, self.padding_idx
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Name changed.
        #     Added fixed positional embeddings.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        AbsEmbedPosType = Optional[Union[FixedPositionalEmbedding, ModifiedBartLearnedPositionalEmbedding]]
        if abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.fixed_pos_embs:
            self.abs_embed_positions: AbsEmbedPosType = FixedPositionalEmbedding(  # type: ignore[no-redef]
                config.d_model,
            ) 

        elif abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.learned_pos_embs:
            self.abs_embed_positions: AbsEmbedPosType = ModifiedBartLearnedPositionalEmbedding(  # type: ignore[no-redef]
                config.max_position_embeddings,
                config.d_model,
            )  
        elif abs_pos_embs_mode == general_shared_constants.AbsPosEmbsModes.no_abs_pos_embs:
            self.abs_embed_positions: AbsEmbedPosType = None  # type: ignore[no-redef]
        else:
            raise ValueError(f"Unknown abs_pos_embs_mode: {abs_pos_embs_mode}")

        if rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
            self.layers = nn.ModuleList(
                [bart_relative_attention.RelAttBartDecoderLayer(config) 
                for _ in range(config.decoder_layers)]
            )
            self.rel_pos_embs = bart_relative_attention.RelPosEmbs(
                model_d=config.d_model, 
                mode=rel_pos_embs_mode,
                num_embeddings=num_rel_pos_embs
            )
        else:
            self.layers = nn.ModuleList(
                [original.BartDecoderLayer(config) for _ in range(config.decoder_layers)]
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = original._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(  # type: ignore[override]
        self,
        *,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules
        # Modified by us
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        decoder_position_ids: Optional[torch.LongTensor],
        tgt_array_indices: Optional[torch.LongTensor],
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, original.BaseModelOutputWithPastAndCrossAttentions]:    
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Added by us
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert inputs_embeds is None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # type: ignore[assignment]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules
        # Added by us.
        # Either we get passed an attention mask by generation, or we generate one when
        # the user calls forward on the whole sentence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if past_key_values_length == 0:
            attention_mask = input_ids != self.config.pad_token_id

        assert attention_mask is not None

        un_prepared_attention_mask = attention_mask
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = original._expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Modified by us
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.abs_embed_positions is not None:
            abs_pos_emb = self.abs_embed_positions(
                past_key_values_length=past_key_values_length,
                attention_mask=un_prepared_attention_mask,
                position_ids=decoder_position_ids,
            )

        hidden_states = inputs_embeds
        if self.abs_embed_positions is not None:
            hidden_states += abs_pos_emb  # typing: ignore[misc]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # decoder layers
        all_hidden_states: Optional[tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        all_self_attns: Optional[tuple[torch.Tensor, ...]] = () if output_attentions else None
        all_cross_attentions: Optional[tuple[torch.Tensor, ...]] = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache: Optional[tuple[torch.Tensor, ...]] = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                assert isinstance(attn_mask, torch.Tensor), type(attn_mask)
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {attn_mask.size()[0]}."
                    )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Modified by us
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
            rel_att_keys, rel_att_values = self.rel_pos_embs(
                attention_mask=un_prepared_attention_mask,
                tgt_array_indices=tgt_array_indices,
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                new_all_hidden_states = (hidden_states,)
                all_hidden_states += new_all_hidden_states

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                assert False, "rel pos embs are not supported for gradient checkpointing"

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                )
            else:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Added relative pos embs
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                decoder_layer_args = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                if self.rel_pos_embs_mode != general_shared_constants.RelPosEmbsChoices.no_rel_pos_embs:
                    decoder_layer_args["rel_att_keys"] = rel_att_keys
                    decoder_layer_args["rel_att_values"] = rel_att_values
                
                layer_outputs = decoder_layer(
                    **decoder_layer_args    
                )
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            
            hidden_states = layer_outputs[0]

            if use_cache:
                new_decoder_cache = (layer_outputs[3 if output_attentions else 1],)
                next_decoder_cache += new_decoder_cache

            if output_attentions:
                new_all_self_attns = layer_outputs[1]
                all_self_attns += new_all_self_attns

                if encoder_hidden_states is not None:
                    new_all_cross_attentions = layer_outputs[2]
                    all_cross_attentions += new_all_cross_attentions

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return original.BaseModelOutputWithPastAndCrossAttentions(
            # The annotations of these are badly defined in the huggingface code
            last_hidden_state=hidden_states,  # type: ignore[arg-type]
            past_key_values=next_cache,  # type: ignore[arg-type]
            hidden_states=all_hidden_states,  # type: ignore[arg-type]
            attentions=all_self_attns,  # type: ignore[arg-type]
            cross_attentions=all_cross_attentions,  # type: ignore[arg-type]
        )



class ModifiedBartModel(original.BartModel):
    """
    Only modified to use `ModifiedBartDecoder` instead of `BartDecoder`.
    """

    def __init__(
        self, 
        config: original.BartConfig, 
        abs_pos_embs_mode: str,
        rel_pos_embs_mode: str,
        num_rel_pos_embs: int,
    ):
        original.BartPretrainedModel.__init__(self, config)
        assert abs_pos_embs_mode in general_shared_constants.AbsPosEmbsModes.__choices__

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = ModifiedBartEncoder(  # type: ignore[assignment]
            config=config, 
            abs_pos_embs_mode=abs_pos_embs_mode, 
            rel_pos_embs_mode=rel_pos_embs_mode, 
            num_rel_pos_embs=num_rel_pos_embs,
            embed_tokens=self.shared,
        )
        self.decoder = ModifiedBartDecoder(  # type: ignore[assignment]
            config=config, 
            abs_pos_embs_mode=abs_pos_embs_mode, 
            rel_pos_embs_mode=rel_pos_embs_mode,
            num_rel_pos_embs=num_rel_pos_embs,
            embed_tokens=self.shared,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        *,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules
        # Added by me
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        decoder_position_ids: Optional[torch.LongTensor],
        tgt_array_indices: Optional[torch.LongTensor] = None,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, original.Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = original.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                tgt_array_indices=tgt_array_indices,

                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, original.BaseModelOutput):
            encoder_outputs = original.BaseModelOutput(   # type: ignore[assignment]
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,   # type: ignore[arg-type]
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,  # type: ignore[arg-type]
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_position_ids=decoder_position_ids,
            tgt_array_indices=tgt_array_indices,

            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        assert isinstance(encoder_outputs, original.BaseModelOutput), type(encoder_outputs)
        assert isinstance(decoder_outputs, original.BaseModelOutputWithPastAndCrossAttentions), type(decoder_outputs)
        assert encoder_outputs is not None

        return original.Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ModifiedBartForConditionalGeneration(original.BartForConditionalGeneration):
    """
    Only modified to use `ModifiedBartModel` instead of `BartModel`.
    """

    def __init__(
        self, 
        config: original.BartConfig, 
        abs_pos_embs_mode: str, 
        rel_pos_embs_mode: str, 
        num_rel_pos_embs: int
    ):
        original.BartPretrainedModel.__init__(self, config)
        self.model = ModifiedBartModel(
            config, 
            abs_pos_embs_mode=abs_pos_embs_mode,
            rel_pos_embs_mode=rel_pos_embs_mode,
            num_rel_pos_embs=num_rel_pos_embs,
        )
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        *,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules
        # Added by me
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        decoder_position_ids: Optional[torch.LongTensor] = None,
        tgt_array_indices: Optional[torch.LongTensor] = None,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, original.Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # assert past_key_values is not None or torch.all(decoder_input_ids[:, 0] == self.config.bos_token_id), (
        # "The first token of the decoder_input_ids should be the <BOS> token."
        # )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = original.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Jules
            # Added by me
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            decoder_position_ids=decoder_position_ids,
            tgt_array_indices=tgt_array_indices,
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = original.CrossEntropyLoss()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Jules
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert self.config.bos_token_id not in labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return original.Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        tgt_array_indices=None,
        **kwargs,
    ):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules.
        # Some checks for things that should not happen.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert head_mask is None, head_mask
        assert decoder_head_mask is None
        assert cross_attn_head_mask is None
        decoder_attention_mask = decoder_input_ids != self.model.config.pad_token_id
        
        assert kwargs == {}, kwargs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules.
        # If we're using a cache, we can't compute the position_ids from
        # the decoder_input_ids inside of the forward call, because
        # we don't have the decoder_input_ids for that.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert tgt_array_indices is None, tgt_array_indices
        if past:
            assert decoder_attention_mask is not None 

            decoder_position_ids = decoder_attention_mask.cumsum(dim=-1) - 1
            decoder_position_ids.masked_fill_(decoder_attention_mask < 0, 0)
            decoder_position_ids = decoder_position_ids[:, -1:]
            
            tgt_array_indices = (torch.tensor(decoder_attention_mask.shape[1]) - 1).reshape(1, -1)
            # rich.print(f"[green]There is a past. {tgt_array_indices}")
        else:
            decoder_position_ids = None
            tgt_array_indices = None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Jules.
        # If we're using a cache, we only need the last value
        # of decoder_input_ids.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "tgt_array_indices": tgt_array_indices,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


def main(model=None):
    print("Starting main.")
    print("importing")
    import numpy as np
    import transformers
    import data_tokenizer
    import data_collator

    print("done w imports")

    maxlen = 1000
    seq_len = 128
    depth = 10
    bsz = 3

    modified_layer = ModifiedBartLearnedPositionalEmbedding(maxlen, depth)
    original_layer = original.BartLearnedPositionalEmbedding(maxlen, depth)
    modified_layer.weight = original_layer.weight
    fake_attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
    input_ids = torch.ones((bsz, seq_len), dtype=torch.long)

    new = modified_layer(
        past_key_values_length=0,
        attention_mask=fake_attention_mask,
        position_ids=None,
    )
    old = original_layer(
        input_ids_shape=input_ids.shape,
        past_key_values_length=0,
    )
    assert torch.allclose(new, old)

    tokenizer = data_tokenizer.Tokenizer()

    config = transformers.BartConfig.from_pretrained("facebook/bart-base")
    config.no_repeat_ngram_size = 0
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.forced_bos_token_id = tokenizer.forced_bos_token_id
    config.forced_eos_token_id = tokenizer.forced_eos_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id

    assert config.eos_token_id != config.pad_token_id, (
        f"eos_token_id and pad_token_id should not be the same. eos_token_id:"
        f" {config.eos_token_id}, pad_token_id: {config.pad_token_id}"
    )
    config.vocab_size = len(tokenizer.vocab)
    config.task_specific_params = {}

    ###############################################################
    # Can change
    ###############################################################
    config.num_hidden_layers = 2  # Default is 6
    config.hidden_size = 128  # Default is 768
    config.encoder_attention_heads = 4  # Default is 16
    config.decoder_attention_heads = 4  # Default is 16
    config.encoder_ffn_dim = config.hidden_size * 4  # Default is 4096
    config.decoder_ffn_dim = config.hidden_size * 4

    config.dropout = 0
    config.layerdrop = 0

    assert tokenizer.max_length == config.max_position_embeddings, (
        f"max_length={tokenizer.max_length} != "
        f"config.max_position_embeddings={config.max_position_embeddings}"
    )

    if model is None:
        model = ModifiedBartForConditionalGeneration(config).cuda().eval()

    collator = data_collator.DataCollatorWithDecoderInputIds(
        tokenizer, model, 512, -100
    )

    ###############################################################
    # Prep input a
    ###############################################################
    inputs_a = tokenizer(
        "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15"
    )
    labels_a = tokenizer(
        "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 = 4"
    )
    features_a = [
        dict(
            input_ids=inputs_a,
            labels=labels_a,
        )
    ]
    preped_a = collator(features_a)
    for k, v in preped_a.items():
        preped_a[k] = v.cuda()
    output_a = model(**preped_a, use_cache=False)

    ###############################################################
    # Prep input b
    ###############################################################
    inputs_b = inputs_a
    diff = 15
    pos = 3
    rtol = 0.01
    labels_b = np.concatenate(
        [
            [0] * diff,
            labels_a[:pos],
            [0] * diff,
            labels_a[pos:-pos],
            [0] * diff,
            labels_a[-pos:],
        ]
    )
    # assert tokenizer.eos_token_id not in labels_b, labels_b
    features_b = [
        dict(
            input_ids=inputs_b,
            labels=labels_b,
        )
    ]

    ###############################################################
    # Collate and stuff
    ###############################################################
    preped_b = collator(features_b)
    assert tokenizer.eos_token_id not in preped_b["decoder_input_ids"], preped_b[
        "decoder_input_ids"
    ]
    assert tokenizer.eos_token_id not in preped_a["decoder_input_ids"], preped_a[
        "decoder_input_ids"
    ]
    # preped_b["decoder_attention_mask"] = preped_b["decoder_input_ids"] != tokenizer.pad_token_id

    for k, v in preped_b.items():
        preped_b[k] = v.cuda()

    ###############################################################
    # Test forward
    ###############################################################
    output_b = model(**preped_b, use_cache=False)

    first_set = np.arange(15, 15 + 3)
    second_set = np.arange(15 + 3, 15 + 3 + 32)
    third_set = np.arange(15 + 3 + 32, 15 + 3 + 32 + 3)

    labels_first_set = [labels_b[i] for i in first_set]
    labels_second_set = [labels_b[i + diff] for i in second_set]
    labels_third_set = [labels_b[i + diff * 2] for i in third_set]

    print(f"labels_a: { labels_a        }")
    print(f"labels_b: { labels_b        }")
    print(f"first:    {labels_first_set }")
    print(f"second:   {labels_second_set}")
    print(f"third:    {labels_third_set }")

    assert np.all(np.array(labels_first_set) != 0)
    assert np.all(np.array(labels_second_set) != 0)
    assert np.all(np.array(labels_third_set) != 0)

    first_range = range(len(first_set))
    second_range = range(first_range.stop, first_range.stop + len(second_set))
    third_range = range(second_range.stop, second_range.stop + len(third_set))

    print(
        [
            torch.allclose(output_a.logits[0, i], output_b.logits[0, j], rtol=rtol)
            for i, j in zip(first_range, first_set)
        ]
    )
    print(
        [
            torch.allclose(
                output_a.logits[0, i], output_b.logits[0, j + diff], rtol=rtol
            )
            for i, j in zip(second_range, second_set)
        ]
    )
    print(
        [
            torch.allclose(
                output_a.logits[0, i], output_b.logits[0, j + 2 * diff], rtol=rtol
            )
            for i, j in zip(third_range, third_set)
        ]
    )

    ###############################################################
    # Prepare for gen
    ###############################################################
    no_labels_a = {k: v for k, v in preped_a.items() if k != "labels"}
    no_labels_b = {k: v for k, v in preped_b.items() if k != "labels"}

    gen_options = dict(
        num_beams=1,
        num_beam_groups=1,
        do_sample=False,
        use_cache=False,
        min_length=0,
        max_length=len(labels_b) + 10,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        tgt_array_indices=None,
    )
    no_labels_a["decoder_input_ids"] = no_labels_a["decoder_input_ids"]

    ###############################################################
    # Test gen
    ###############################################################
    assert tokenizer.eos_token_id not in no_labels_a["decoder_input_ids"], preped_a[
        "decoder_input_ids"
    ]
    assert tokenizer.eos_token_id not in no_labels_b["decoder_input_ids"], preped_b[
        "decoder_input_ids"
    ]

    if "decoder_attention_mask" in "no_labels_a":
        del no_labels_a["decoder_attention_mask"]

    if "decoder_attention_mask" in "no_labels_b":
        del no_labels_b["decoder_attention_mask"]

    print(f"{no_labels_a['decoder_input_ids'] = }")
    print(f"{no_labels_b['decoder_input_ids'] = }")
    gen_a_dict = model.generate(**no_labels_a, **gen_options)
    gen_b_dict = model.generate(**no_labels_b, **gen_options)
    gen_options["use_cache"] = True
    gen_c_dict = model.generate(**no_labels_b, **gen_options)

    print(f"{gen_a_dict['sequences'][0] = }")
    print(f"{gen_b_dict['sequences'][0] = }")
    print(f"{gen_c_dict['sequences'][0] = }")

    str_a = tokenizer.decode(
        gen_a_dict["sequences"][0].detach().cpu().numpy().tolist(), False
    )
    str_b = tokenizer.decode(
        gen_b_dict["sequences"][0].detach().cpu().numpy().tolist(), False
    )
    str_c = tokenizer.decode(
        gen_b_dict["sequences"][0].detach().cpu().numpy().tolist(), False
    )

    print(f"{str_a = }")
    print(len(str_a.strip().split()))
    print(f"{str_b = }")
    print(len(str_b.strip().split()))
    print(f"{str_c = }")
    print(len(str_c.strip().split()))

    min_length = min(
        len(gen_a_dict["scores"]),
        len(gen_b_dict["scores"]),
        len(gen_c_dict["scores"]),
    )
    for i in range(min_length):
        a = gen_a_dict["scores"][i]
        b = gen_b_dict["scores"][i]
        c = gen_c_dict["scores"][i]
        print("a & b", torch.allclose(a, b, rtol=rtol))
        print("b & c", torch.allclose(b, c, rtol=rtol))
        print("a & c", torch.allclose(a, c, rtol=rtol))
        print()


if __name__ == "__main__":
    main()
