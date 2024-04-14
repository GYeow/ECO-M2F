from detectron2.modeling import MViT
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import torch.nn as nn
from functools import partial
# from icecream import ic

@BACKBONE_REGISTRY.register()
class D2MViT(MViT, Backbone):
    def __init__(self, cfg, input_shape):

        img_size_=cfg.MODEL.MViT.PRETRAIN_IMG_SIZE
        patch_kernel_=cfg.MODEL.MViT.PATCH_KERNEL
        patch_stride_=cfg.MODEL.MViT.PATCH_STRIDE
        patch_padding_=cfg.MODEL.MViT.PATCH_PADDING
        embed_dim_ = cfg.MODEL.MViT.EMBED_DIM
        depth_=cfg.MODEL.MViT.DEPTH
        num_heads_=cfg.MODEL.MViT.NUM_HEADS
        last_block_indexes_=cfg.MODEL.MViT.LAST_BLOCK_INDEXES
        qkv_pool_kernel_=cfg.MODEL.MViT.QKV_POOL_KERNEL
        adaptive_kv_stride_=cfg.MODEL.MViT.ADAPTIVE_KV_STRIDE
        adaptive_window_size_=cfg.MODEL.MViT.ADAPTIVE_WINDOW_SIZE
        residual_pooling_=cfg.MODEL.MViT.RESIDUAL_POOLING
        mlp_ratio_=cfg.MODEL.MViT.MLP_RATIO
        drop_path_rate_=cfg.MODEL.MViT.DROP_PATH_RATE
        use_abs_pos_=cfg.MODEL.MViT.USE_ABS_POS
        use_rel_pos_=cfg.MODEL.MViT.USE_REL_POS
        rel_pos_zero_init_=cfg.MODEL.MViT.REL_POS_ZERO_INIT

        in_chans_= 3
        qkv_bias_=True
        norm_layer_=partial(nn.LayerNorm, eps=1e-6)
        act_layer_=nn.GELU
        use_act_checkpoint_=False
        pretrain_img_size_=224
        pretrain_use_cls_token_=True
        _out_features_=("scale3", "scale4", "scale5")
        super().__init__(
            img_size=img_size_,
            patch_kernel=patch_kernel_,
            patch_stride=patch_stride_,
            patch_padding=patch_padding_,
            in_chans=in_chans_,
            embed_dim=embed_dim_,
            depth=depth_,
            num_heads=num_heads_,
            last_block_indexes=last_block_indexes_,
            qkv_pool_kernel=qkv_pool_kernel_,
            adaptive_kv_stride=adaptive_kv_stride_,
            adaptive_window_size=adaptive_window_size_,
            residual_pooling=residual_pooling_,
            mlp_ratio=mlp_ratio_,
            qkv_bias=qkv_bias_,
            drop_path_rate=drop_path_rate_,
            norm_layer=norm_layer_,
            act_layer=act_layer_,
            use_abs_pos=use_abs_pos_,
            use_rel_pos=use_rel_pos_,
            rel_pos_zero_init=rel_pos_zero_init_,
            use_act_checkpoint=use_act_checkpoint_,
            pretrain_img_size=pretrain_img_size_,
            pretrain_use_cls_token=pretrain_use_cls_token_,
            out_features=_out_features_
        )

        self._out_features = _out_features_

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"Backbone takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        # ic(len(y))
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        # ic(outputs.keys())
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
