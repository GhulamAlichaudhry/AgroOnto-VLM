"""
AgroOnto-VLM model: CLIP ViT-B/32 + LoRA adapters + classification head.
"""
import math
import torch
import torch.nn as nn
import clip

from . import config


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer."""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class AgroOntoVLM(nn.Module):
    """CLIP ViT-B/32 with LoRA adapters and a classification head."""

    def __init__(self, num_classes, class_names, clip_model_name="ViT-B/32",
                 lora_rank=8, lora_alpha=16, lora_blocks=6, device="cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device

        clip_model, _ = clip.load(clip_model_name, device="cpu", jit=False)
        self.clip = clip_model.float()

        for p in self.clip.parameters():
            p.requires_grad = False

        self.embed_dim = self.clip.visual.output_dim  # 512 for ViT-B/32

        self.lora_layers = nn.ModuleList()
        num_blocks = len(self.clip.visual.transformer.resblocks)
        start_block = max(0, num_blocks - lora_blocks)

        for i in range(start_block, num_blocks):
            block = self.clip.visual.transformer.resblocks[i]
            in_proj_dim = block.attn.in_proj_weight.shape[1]
            out_proj_dim = block.attn.out_proj.weight.shape[1]
            mlp_c_fc_dim = block.mlp[0].weight.shape[1]
            mlp_c_proj_dim = block.mlp[2].weight.shape[1]

            lora_dict = nn.ModuleDict({
                "q": LoRALinear(in_proj_dim, in_proj_dim, lora_rank, lora_alpha),
                "k": LoRALinear(in_proj_dim, in_proj_dim, lora_rank, lora_alpha),
                "v": LoRALinear(in_proj_dim, in_proj_dim, lora_rank, lora_alpha),
                "out": LoRALinear(out_proj_dim, out_proj_dim, lora_rank, lora_alpha),
                "mlp_fc": LoRALinear(mlp_c_fc_dim, mlp_c_fc_dim * 4, lora_rank, lora_alpha),
                "mlp_proj": LoRALinear(mlp_c_fc_dim * 4, mlp_c_proj_dim, lora_rank, lora_alpha),
            })
            self.lora_layers.append(lora_dict)

        self.head = nn.Linear(self.embed_dim, num_classes)

        self.delta = nn.Parameter(torch.zeros(num_classes, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self._register_hooks(start_block)

    def _register_hooks(self, start_block):
        self._hooks = []
        for idx, i in enumerate(range(start_block,
                                       len(self.clip.visual.transformer.resblocks))):
            block = self.clip.visual.transformer.resblocks[i]
            lora = self.lora_layers[idx]

            def make_attn_hook(lora_dict, blk):
                def hook(module, input, output):
                    x = input[0]
                    B = x.shape[1]
                    d = x.shape[2]
                    delta_q = lora_dict["q"](x.reshape(-1, d)).reshape(x.shape)
                    delta_k = lora_dict["k"](x.reshape(-1, d)).reshape(x.shape)
                    delta_v = lora_dict["v"](x.reshape(-1, d)).reshape(x.shape)
                    delta_out = lora_dict["out"](output.reshape(-1, d)).reshape(output.shape)
                    return output + delta_out
                return hook

            def make_mlp_hook(lora_dict, blk):
                def hook(module, input, output):
                    x = input[0]
                    d = x.shape[-1]
                    delta = lora_dict["mlp_proj"](
                        torch.relu(lora_dict["mlp_fc"](x.reshape(-1, d)))
                    ).reshape(output.shape)
                    return output + delta
                return hook

            h1 = block.attn.register_forward_hook(make_attn_hook(lora, block))
            h2 = block.mlp.register_forward_hook(make_mlp_hook(lora, block))
            self._hooks.extend([h1, h2])

    def encode_image(self, images):
        features = self.clip.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def forward(self, images):
        features = self.encode_image(images)
        logits_cls = self.head(features)

        text_weights = self.head.weight.data
        text_weights = text_weights + self.delta
        text_weights = text_weights / text_weights.norm(dim=-1, keepdim=True)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_clip = scale * (features @ text_weights.T)

        return logits_cls, logits_clip, features

    def get_param_groups(self, lr_lora, lr_head, lr_delta, lr_logit_scale):
        lora_params = list(self.lora_layers.parameters())
        return [
            {"params": lora_params, "lr": lr_lora},
            {"params": [self.head.weight, self.head.bias], "lr": lr_head},
            {"params": [self.delta], "lr": lr_delta},
            {"params": [self.logit_scale], "lr": lr_logit_scale},
        ]
