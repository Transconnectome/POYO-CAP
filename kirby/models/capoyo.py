from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from einops import rearrange, repeat

from kirby.taxonomy import DecoderSpec, Decoder
from kirby.taxonomy.mice import Cre_line, Depth_classes
from kirby.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    PerceiverRotary,
    prepare_for_multitask_readout)

from kirby.data import pad, chain, track_mask, track_batch
from kirby.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
    get_sinusoidal_encoding,
)

from kirby.models.poyo_plus import BACKEND_CONFIGS


class CaPOYO(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_input=None,
        dim_head=64,
        num_latents=64,
        patch_size=1,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_cre_line_embedding=True,
        use_depth_embedding=False,
        use_spatial_embedding=True,
        use_roi_feat_embedding=True,
        use_session_embedding=True,
        use_unit_embedding=True,
        use_depth_class_embedding=True,
        use_cre_loss=True,
        use_cre_data=False,
        use_random_masking_loss=True,
        random_masking_loss_option='dim',
        masking_rate=0.3,
        noise_level=0.1,
        no_decoding=False,
        backend_config="gpu_fp32",
        decoder_specs: Dict[str, DecoderSpec],
        pretrain=True,
        finetune=False,
        task='movie_decoding',
        unet_type='unet',
        output_activation='sigmoid',
        range_loss_weight=0.0,
    ):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size

        self.use_session_embedding = use_session_embedding
        self.use_unit_embedding = use_unit_embedding

        dim_input = dim_input or dim
        self.dim_input = dim_input

        # input embs
        self.unit_emb = InfiniteVocabEmbedding(dim_input, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim_input, init_scale=emb_init_scale)
        self.value_embedding_layer = nn.Linear(patch_size, dim_input, bias=False)

        self.use_roi_feat_embedding = use_roi_feat_embedding
        self.use_cre_line_embedding = use_cre_line_embedding # False!!
        self.use_depth_embedding = use_depth_embedding
        self.use_spatial_embedding = use_spatial_embedding

        ## Stella added "use_depth_class_embedding"
        self.use_depth_class_embedding = use_depth_class_embedding

        ## Stella added all these pretraining losses
        ## pre-train loss functions
        ## Stella added use_cre_loss
        self.use_cre_data = use_cre_data
        self.use_cre_loss = use_cre_loss # default (28th OCT 2024) : True
        self.use_random_masking_loss = use_random_masking_loss # default (3rd DEC 2024) : True
        self.masking_rate = masking_rate # default (20th MAR 2025) : 0.9
        self.random_masking_loss_option = random_masking_loss_option
        self.no_decoding = no_decoding # pretraining option (12th MAR 2025)
        self.task = task
        self.noise_level = noise_level # masking option (13th MAR 2025)
        self.pretrain = pretrain
        self.finetune = finetune
        self.unet_type = unet_type

        if self.use_cre_line_embedding:
            self.cre_line_embedding_layer = Embedding(
                Cre_line.max_value() + 1, dim_input, init_scale=emb_init_scale
            )

        if self.use_roi_feat_embedding:
            self.unit_feat_embedding_layer = nn.Linear(3, dim_input, bias=True)

        if self.use_depth_embedding:
            self.depth_embedding_layer = Embedding(
                Depth_classes.max_value() + 1, dim_input, init_scale=emb_init_scale
            )
        if self.use_depth_class_embedding:
            self.depth_class_embedding_layer = Embedding(
                Depth_classes.max_value() + 1, dim, init_scale=emb_init_scale
            )

        # latent embs
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        ## this is trainable embedding!!

        # output embs
        # dim 64, emb_init_scale 0.02
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        
        self.task_emb = Embedding(
            Decoder.max_value() + 1, dim, init_scale=emb_init_scale
        )

        # determine backend
        if backend_config not in BACKEND_CONFIGS.keys():
            raise ValueError(
                f"Invalid backend config: {backend_config}, must be one of"
                f" {list(BACKEND_CONFIGS.keys())}"
            )

        self.batch_type = BACKEND_CONFIGS[backend_config][0]

        # the input will be a concatenation of the unit embedding, the value embedding,
        # and any additional embeddings
        context_dim_factor = 2 + sum(
            [
                self.use_cre_line_embedding,
                self.use_depth_embedding,
                self.use_roi_feat_embedding,
                self.use_spatial_embedding,
            ]
        )
        context_dim = context_dim_factor * dim_input

        self.perceiver_io = PerceiverRotary(
                dim=dim,
                context_dim=context_dim,
                dim_head=dim_head,
                depth=depth,
                cross_heads=cross_heads,
                self_heads=self_heads,
                ffn_dropout=ffn_dropout,
                lin_dropout=lin_dropout,
                atn_dropout=atn_dropout,
                backend=BACKEND_CONFIGS[backend_config][1],
                use_random_masking_loss=self.use_random_masking_loss,
                random_masking_loss_option=self.random_masking_loss_option,
                masking_rate=self.masking_rate,
                noise_level=self.noise_level,
                no_decoding=self.no_decoding
            )

        # Output projections + loss
        self.readout = MultitaskReadout(
            latent_dim=dim,
            decoder_specs=decoder_specs,
            batch_type=self.batch_type[2],
            use_cre_loss=self.use_cre_loss,
            use_random_masking_loss=self.use_random_masking_loss,
            no_decoding=self.no_decoding,
            masking_rate=self.masking_rate,
            pretrain=self.pretrain,
            finetune=self.finetune,
            task=self.task,
            unet_type=self.unet_type,
            output_activation=output_activation,
            range_loss_weight=range_loss_weight,
        )
       
        
        self.dim = dim
    
    def load_from_ckpt(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def freeze_encoder(self):
        encoder_modules = [
            self.unit_emb,
            self.session_emb,
            self.perceiver_io]
        for module in self.children():
            if module in encoder_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def freeze_encoder_former(self):
        encoder_modules = [
            self.unit_emb,
            self.session_emb,
            self.perceiver_io.enc_atn,
            self.perceiver_io.enc_ffn,]
        for module in self.children():
            if module in encoder_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
    
    def freeze_encoder_middle(self):
        encoder_modules = [
            self.unit_emb,
            self.session_emb,
            self.perceiver_io.proc_layers]
        for module in self.children():
            if module in encoder_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def freeze_encoder_latter(self):
        encoder_modules = [
            self.unit_emb,
            self.session_emb,
            self.perceiver_io.dec_atn,
            self.perceiver_io.dec_ffn,]
        for module in self.children():
            if module in encoder_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def freeze_middle(self) -> List[nn.Module]:
        # Freeze everything except the readout, unit embedding, and session embedding
        # layers.
        middle_modules = []
        banned_modules = [
            self.readout,
            self.unit_emb,
            self.session_emb,
            self.enc_atn,
            self.enc_ffn,
        ]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            middle_modules.append(module)

        return middle_modules

    def unfreeze_middle(self) -> None:
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        *,
        # input sequence
        unit_index,  # (B, N_in)
        timestamps,  # (B, N_in)
        patches,  # (B, N_in, N_feats)
        token_type,  # (B, N_in)
        unit_feats=None,  # (B, N_in, N_feats)
        unit_spatial_emb=None,  # (B, N_in, dim)
        unit_cre_line=None,  # (B, N_in)
        unit_depth=None,  # (B, N_in)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_decoder_index,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
        movie_frames : Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        input_feats = []
        if self.use_unit_embedding:
            ## these two are always identical.
            # print('unit index', unit_index.shape)
            # print('token type', token_type.shape)
            # print('self.unit_emb(unit_index)', self.unit_emb(unit_index).shape)
            # print('self.token_type_emb(token_type)', self.token_type_emb(token_type).shape)
            input_feats.append(
                self.unit_emb(unit_index) + self.token_type_emb(token_type)
            )
        else:
            input_feats.append(self.token_type_emb(token_type))

        # print('patches shape', patches.shape) # (batch, ????, 1) : problem occurs when ???? is not N_in
        # print('self.value_embedding_layer(patches)', self.value_embedding_layer(patches).shape) # (batch, ????, 64)
        input_feats.append(self.value_embedding_layer(patches))

        if self.use_roi_feat_embedding:
            input_feats.append(self.unit_feat_embedding_layer(unit_feats))

        if self.use_spatial_embedding:
            input_feats.append(unit_spatial_emb)

        if self.use_cre_line_embedding:
            input_feats.append(self.cre_line_embedding_layer(unit_cre_line))

        if self.use_depth_embedding:
            input_feats.append(self.depth_embedding_layer(unit_depth))

        if self.use_cre_loss:
            output_timestamps = output_timestamps[:, 0:1]
        # print('patches', patches.shape)
        # print('input feats first', input_feats[0].shape)
        # print('input feats second', input_feats[1].shape)

        inputs = torch.cat(
            input_feats,
            dim=-1,
        )

        # print('input shape: ', inputs.shape)
        ## (128, 8544, 128) - the middle number keeps changing. timeseries?
        ## temporal, spatial, spatiotemporal


        # latents
        latents = self.latent_emb(latent_index) # just an empty shell. content is filled in here.
        # outputs

        if self.use_cre_loss:
            # do not use decoder attached to CRE_LINE
            output_decoder_index = output_decoder_index[:, 0:1]
            session_index = session_index[:, 0:1]
        output_queries = self.task_emb(output_decoder_index)
        ## output decoder index (drifting gratings) : (128, 1)
        ## output queries (drifting gratings) : (128, 1, 64)

        if self.use_session_embedding:
            output_queries = output_queries + self.session_emb(session_index)

        # feed into perceiver
        if self.no_decoding:
            if self.use_random_masking_loss:
                masked_latents, original_latents = self.perceiver_io(
                    inputs=inputs,
                    latents=latents,
                    output_queries=output_queries,
                    input_timestamps=timestamps,
                    latent_timestamps=latent_timestamps,
                    output_query_timestamps=output_timestamps,
                    input_mask=input_mask,
                    input_seqlen=input_seqlen,
                    latent_seqlen=latent_seqlen,
                    output_query_seqlen=output_seqlen,
                )
            
        else:
            if self.use_random_masking_loss:
                masked_latents, original_latents, output_latents = self.perceiver_io(
                    inputs=inputs,
                    latents=latents,
                    output_queries=output_queries,
                    input_timestamps=timestamps,
                    latent_timestamps=latent_timestamps,
                    output_query_timestamps=output_timestamps,
                    input_mask=input_mask,
                    input_seqlen=input_seqlen,
                    latent_seqlen=latent_seqlen,
                    output_query_seqlen=output_seqlen,
                )
            else:
                output_latents = self.perceiver_io(
                    inputs=inputs,
                    latents=latents,
                    output_queries=output_queries,
                    input_timestamps=timestamps,
                    latent_timestamps=latent_timestamps,
                    output_query_timestamps=output_timestamps,
                    input_mask=input_mask,
                    input_seqlen=input_seqlen,
                    latent_seqlen=latent_seqlen,
                    output_query_seqlen=output_seqlen,
                )
        
        # print('shape of output_latents', output_latents.shape)
        # output_latents torch.Size([128, 1, 64])

        # Readout layer
        # print('output_values ["DRIFTING_GRATINGS"] in CaPOYO class', output_values['DRIFTING_GRATINGS'].shape) # 128
        # unit_cre_line.shape : (128 (batch size), 4-digits like 7904)
        ## in same batch, number of cells is same. different batch? different number of cells.
        ## unique value : 0 and another integer.. is it padded??? (but 0 means another cre-line)
        ## each sample is zero-padded to the length of the longest sample in the batch. yes, it is padded.
        # it means it have 7904 cells here.. haha

        ###### Save the latents !! ########
        ### just for test phase ###
        # torch.save(output_latents, '/scratch/x3312a02/POYO-SSL/geometry_analysis/output_latents_pretrain_exc_finetune_inh.pt')
        ###################################
        if self.no_decoding:
            if self.use_random_masking_loss: # only this case (12th MAR 2025)
                output, loss, losses_taskwise = self.readout(
                        output_latents=latents,
                        output_decoder_index=output_decoder_index,
                        output_batch_index=output_batch_index,
                        output_values=output_values,
                        output_weights=output_weights,
                        cre_lines=unit_cre_line,
                        masked_latents=masked_latents,
                        original_latents=original_latents,
                    )
                
                return None, None, output, loss, losses_taskwise
                # if you are using 'just' random masking loss, then output is empty!!
        
        else:   
            if self.use_random_masking_loss:
                output, loss, losses_taskwise = self.readout(
                        output_latents=output_latents,
                        output_decoder_index=output_decoder_index,
                        output_batch_index=output_batch_index,
                        output_values=output_values,
                        output_weights=output_weights,
                        cre_lines=unit_cre_line,
                        masked_latents=masked_latents,
                        original_latents=original_latents,
                    )
            else:
                if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
                    output, loss, losses_taskwise = self.readout(
                                            output_latents=output_latents,
                                            output_decoder_index=output_decoder_index,
                                            output_batch_index=output_batch_index,
                                            output_values=output_values,
                                            output_weights=output_weights,
                                            cre_lines=unit_cre_line,
                                            movie_frames=movie_frames
                                        )
                else:    
                    output, loss, losses_taskwise = self.readout(
                            output_latents=output_latents,
                            output_decoder_index=output_decoder_index,
                            output_batch_index=output_batch_index,
                            output_values=output_values,
                            output_weights=output_weights,
                            cre_lines=unit_cre_line
                        )
            return output_values, output_latents, output, loss, losses_taskwise
        

class CaPOYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        decoder_registry (Dict): Registry of the decoders.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        decoder_registry,
        latent_step,
        num_latents_per_step,
        dim,
        patch_size,
        batch_type,
        eval=False,
        use_cre_line_embedding=False,
        use_depth_embedding=False,
        use_spatial_embedding=False,
        use_roi_feat_embedding=False,
        use_cre_loss=False,
        use_cre_data=False,
        use_random_masking_loss=False,
        no_decoding=False,
        task='movie_decoding',
        unet_type='unet'
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer
        self.no_decoding = no_decoding
        self.task = task
        self.unet_type = unet_type

        self.decoder_registry = decoder_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.patch_size = patch_size

        self.batch_type = batch_type
        self.eval = eval

        self.use_cre_line_embedding = use_cre_line_embedding
        self.use_depth_embedding = use_depth_embedding
        self.use_spatial_embedding = use_spatial_embedding
        self.use_roi_feat_embedding = use_roi_feat_embedding
        
        self.use_cre_loss = use_cre_loss
        self.use_cre_data = use_cre_data
        self.use_random_masking_loss = use_random_masking_loss

    def __call__(self, data):
        # context window
        start, end = 0.0, 1.0

        ### prepare input
        unit_ids = data.units.id

        calcium_traces = data.calcium_traces.df_over_f.astype(
            np.float32
        )  # (time x num_rois)
        timestamps = data.calcium_traces.timestamps.astype(np.float32)
        num_rois = calcium_traces.shape[1]
        if self.task in ['static_gratings', 'scene_decoding']:
            calcium_traces = np.tile(calcium_traces , (3,1))
            timestamps = np.tile(timestamps, 3)
        # patch tokenization
        # clip the time dimension to accomodate the patch size
        # WARNING: it is important to still have a multiple of patch_size
        # this is a fix to deal with the arbitrary slicing that might happen
        num_frames = calcium_traces.shape[0] // self.patch_size * self.patch_size
        if num_frames == 0:
            #print(f"🚫 Empty trace")
            return None
            # raise ValueError(
            #     f"The patch size ({self.patch_size}) is larger than "
            #     f"sequence length ({calcium_traces.shape[0]})."
            # )
        calcium_traces = calcium_traces[:num_frames]
        timestamps = timestamps[:num_frames]

        calcium_traces = calcium_traces.reshape(
            -1, self.patch_size, calcium_traces.shape[1]
        )
        timestamps = timestamps.reshape(-1, self.patch_size).mean(axis=1)

        # now flatten
        patches = rearrange(calcium_traces, "t d c -> (t c) d")
        unit_index = repeat(np.arange(num_rois), "c -> (t c)", t=timestamps.shape[0])

        if self.use_spatial_embedding:
            if not "imaging_plane_xy" in data.units.keys:
                raise ValueError(
                    "ROI coordinates in the imaging plane are required for ROI spatial embeddings."
                )
            unit_lvl_spatial_emb = get_sinusoidal_encoding(
                data.units.imaging_plane_xy[:, 0],
                data.units.imaging_plane_xy[:, 1],
                self.dim // 2,
            ).astype(np.float32)
            unit_spatial_emb = repeat(
                unit_lvl_spatial_emb, "c d -> (t c) d", t=timestamps.shape[0]
            )
        else:
            unit_spatial_emb = None

        if self.use_roi_feat_embedding:
            if not all(
                [
                    "imaging_plane_area" in data.units.keys,
                    "imaging_plane_width" in data.units.keys,
                    "imaging_plane_height" in data.units.keys,
                ]
            ):
                raise ValueError(
                    "ROI area, width, and height are required for ROI feature embeddings."
                )
            unit_lvl_feats = np.stack(
                [
                    data.units.imaging_plane_area,
                    data.units.imaging_plane_width,
                    data.units.imaging_plane_height,
                ],
                axis=1,
            ).astype(np.float32)
            unit_feats = repeat(unit_lvl_feats, "c f -> (t c) f", t=timestamps.shape[0])
        else:
            unit_feats = None

        timestamps = repeat(timestamps, "t -> (t c)", c=num_rois)

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(unit_index)]
        )
        unit_index = np.concatenate([se_unit_index, unit_index])
        timestamps = np.concatenate([se_timestamps, timestamps])
        patches = np.concatenate(
            [
                np.zeros((se_unit_index.shape[0], patches.shape[1]), dtype=np.float32),
                patches,
            ]
        )
        if unit_feats is not None:
            unit_feats = np.concatenate(
                [
                    unit_lvl_feats[se_unit_index],
                    unit_feats,
                ]
            )
        if unit_spatial_emb is not None:
            unit_spatial_emb = np.concatenate(
                [
                    unit_lvl_spatial_emb[se_unit_index],
                    unit_spatial_emb,
                ]
            )

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        unit_index = local_to_global_map[unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        ### prepare outputs
        session_index = self.session_tokenizer(data.session)

        (
            output_timestamps,
            output_task_index,
            output_values,
            output_weights,
            output_subtask_index,
        ) = prepare_for_multitask_readout(
            data,
            self.decoder_registry,
        )

        if self.use_cre_line_embedding or self.use_cre_loss or self.use_cre_data:
            subject_cre_line = data.subject.cre_line
            subject_cre_line_index = Cre_line.from_string(subject_cre_line).value
            unit_cre_line = np.full_like(unit_index, subject_cre_line_index)

        if self.use_depth_embedding:
            subject_depth = data.subject.depth_class
            subject_depth_index = Depth_classes.from_string(subject_depth).value
            unit_depth = np.full_like(unit_index, subject_depth_index)

        batch = {}
        if self.batch_type[0] == "stacked":
            batch = {
                **batch,
                # input sequence
                "unit_index": pad(unit_index),
                "timestamps": pad(timestamps),
                "patches": pad(patches),
                "token_type": pad(token_type_index),
                "input_mask": track_mask(unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
            }
            if self.use_spatial_embedding:
                batch["unit_spatial_emb"] = pad(unit_spatial_emb)
            if self.use_roi_feat_embedding:
                batch["unit_feats"] = pad(unit_feats)
            if self.use_cre_line_embedding or self.use_cre_loss or self.use_cre_data:
                batch["unit_cre_line"] = pad(unit_cre_line)
            if self.use_depth_embedding:
                batch["unit_depth"] = pad(unit_depth)
        else:
            batch = {
                **batch,
                # input sequence
                "unit_index": chain(unit_index),
                "timestamps": chain(timestamps),
                "patches": chain(patches),
                "token_type": chain(token_type_index),
                "input_seqlen": len(unit_index),
                # latent sequence
                "latent_index": chain(latent_index),
                "latent_timestamps": chain(latent_timestamps),
                "latent_seqlen": len(latent_index),
            }
            if self.use_spatial_embedding:
                batch["unit_spatial_emb"] = chain(unit_spatial_emb)
            if self.use_roi_feat_embedding:
                batch["unit_roi_feats"] = chain(unit_feats)
            if self.use_cre_line_embedding or self.use_cre_loss or self.use_cre_data:
                batch["unit_cre_line"] = chain(unit_cre_line)
            if self.use_depth_embedding:
                batch["unit_depth"] = chain(unit_depth)
        if self.batch_type[1] == "chained":
            batch["latent_seqlen"] = len(latent_index)

        if self.batch_type[2] == "stacked":
            batch = {
                **batch,
                # output sequence
                "session_index": pad(np.repeat(session_index, len(output_timestamps))),
                "output_timestamps": pad(output_timestamps),
                "output_decoder_index": pad(output_task_index),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }
        else:
            batch = {
                **batch,
                # output sequence
                "session_index": chain(session_index),
                "output_timestamps": chain(output_timestamps),
                "output_decoder_index": chain(output_task_index),
                "output_seqlen": len(output_timestamps),
                "output_batch_index": track_batch(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(output_subtask_index)

        return batch
