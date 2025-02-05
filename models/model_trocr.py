import torch
import torch.nn as nn
from transformers import ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

class TrocrResizer:
    def __init__(self, trocr_model, optimal_height, optimal_width, mini_patch_size, device):
        self.trocr_model = trocr_model.to(device)
        self.optimal_height = optimal_height
        self.optimal_width = optimal_width
        self.mini_patch_size = mini_patch_size
        self.device = device
        
    def resize_and_create_model(self):
        encoder_config = self.trocr_model.config.encoder

        new_vit_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=encoder_config.num_hidden_layers,
            num_attention_heads=12,
            intermediate_size=encoder_config.intermediate_size,
            hidden_act=encoder_config.hidden_act,
            layer_norm_eps=encoder_config.layer_norm_eps,
            hidden_dropout_prob=encoder_config.hidden_dropout_prob,
            attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
            image_size=(self.optimal_height, self.optimal_width),
            patch_size=self.mini_patch_size,
            num_channels=encoder_config.num_channels,
            qkv_bias=encoder_config.qkv_bias,
        )

        num_patches_height = self.optimal_height // self.mini_patch_size
        num_patches_width = self.optimal_width // self.mini_patch_size
        num_patches = num_patches_height * num_patches_width
        
        print(f"Number of patches (height x width): {num_patches_height} x {num_patches_width} = {num_patches}")

        new_trocr_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=new_vit_config,
            decoder_config=self.trocr_model.config.decoder
        )
        
        new_trocr_model = VisionEncoderDecoderModel(config=new_trocr_config).to(self.device)
        new_num_patches = self.update_position_embeddings(new_trocr_model)
        
        old_patch_embeddings = self.trocr_model.encoder.embeddings.patch_embeddings.projection.weight.data.to(self.device)
        new_patch_embeddings = nn.Parameter(torch.zeros((old_patch_embeddings.size(0), new_num_patches)).to(self.device))
        new_patch_embeddings.data = old_patch_embeddings[:, :new_num_patches]
        new_trocr_model.encoder.embeddings.patch_embeddings.projection.weight = new_patch_embeddings
        
        encoder_state_dict = self.trocr_model.encoder.state_dict()
        encoder_state_dict.pop('embeddings.position_embeddings', None)
        encoder_state_dict.pop('embeddings.patch_embeddings.projection.weight', None)
        new_trocr_model.encoder.load_state_dict(encoder_state_dict, strict=False)
        new_trocr_model.decoder.load_state_dict(self.trocr_model.decoder.state_dict(), strict=False)
        
        return new_trocr_model

    def update_position_embeddings(self, new_model):
        old_position_embeddings = self.trocr_model.encoder.embeddings.position_embeddings.data.to(self.device)
        num_old_patches = old_position_embeddings.size(1) - 1
        old_grid_size = int(num_old_patches ** 0.5)

        num_new_patches_height = self.optimal_height // self.mini_patch_size
        num_new_patches_width = self.optimal_width // self.mini_patch_size
        new_num_patches = num_new_patches_height * num_new_patches_width

        class_token = old_position_embeddings[:, 0:1, :].to(self.device)
        new_position_embeddings = torch.zeros(1, new_num_patches + 1, old_position_embeddings.size(-1)).to(self.device)

        old_patch_tokens = old_position_embeddings[:, 1:, :].reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2).to(self.device)
        interpolated_patch_tokens = nn.functional.interpolate(old_patch_tokens, size=(num_new_patches_height, num_new_patches_width), mode='bilinear', align_corners=False)
        interpolated_patch_tokens = interpolated_patch_tokens.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)

        new_position_embeddings[:, 0:1, :] = class_token
        new_position_embeddings[:, 1:, :] = interpolated_patch_tokens
        
        new_model.encoder.embeddings.position_embeddings = nn.Parameter(new_position_embeddings)
        
        return new_num_patches
