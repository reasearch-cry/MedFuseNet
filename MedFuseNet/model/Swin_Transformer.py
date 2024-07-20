import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils_hiformer import BasicLayer

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        # self.out_channels=out_channels
        self.patch_embed = nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=4)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self,x):
        # B,C,H,W=x.shape
        x = self.patch_embed(x) #[B,embed_dim,h,w]
        B, C, H, W = x.shape
        x = x.flatten(2)    #[B,embed_dim,h*w]
        x = x.permute([0,2,1])
        x = self.norm(x)
        x=x.reshape([B,C,H,W])
        return x



class PatchMerging(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)
        return x

# 将layer分成若干个windows，然后在每个windows内attention计算
def windows_partition(x , window_size):
    B , H , W , C = x.shape
    x = x.reshape([B,H//window_size,window_size,W//window_size,window_size,C])
    # [B,H//window_size,W//window_size,window_size,window_size,C]
    x.permute([0,1,3,2,4,5])
    x.reshape([-1,window_size,window_size,C])
    # [B*H//window_size*w//window_size,window_size,window_size,c]
    return x

#将若干个windows合并为一个layer。
def window_reverse(window, window_size , H , W ):
    B = window.shape[0]//((H//window_size)*(W//window_size))
    x = window.reshape([B,H//window_size,W//window_size,window_size,window_size,-1])
    x = x.permute([0,1,3,2,4,5])
    x = x.reshape([B,H,W,-1])
    return x


class window_attention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim, int(dim * 3))
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + (self.num_heads, self.dim_head)
        x = x.reshape(new_shape)
        # [B,num_patches,num_heads,dim_head]
        x = x.permute([0, 2, 1, 3])
        # [B,num_heads,num_patches,dim_head]
        return x

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, -1)

        q, k, v = map(self.transpose_multi_head, qkv)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(3,2))

        # attn = self.softmax(attn)
        if mask is None:
            attn = self.softmax(attn)
        else:
            attn = attn.reshape([B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])
            attn = self.softmax(attn)
        attn = torch.matmul(attn, v)
        # [B,num_heads,num_patches,dim_head]
        attn = attn.permute([0, 2, 1, 3])
        # [B,num_patches,num_heas,dim_head]
        attn = attn.reshape([B, N, C])
        out = self.proj(attn)
        return out

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,window_size,shift_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.att_norm = nn.LayerNorm(dim)
        self.attn = window_attention(dim=dim,window_size=window_size, num_heads=num_heads)
        self.mlp = Mlp(dim)
        self.shift_size = shift_size
        self.mlp_norm = nn.LayerNorm(dim)
        if self.shift_size > 0:
            H, W = self.resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = torch.where(attn_mask != 0,
                                     torch.ones_like(attn_mask) * float(-100.0),
                                     attn_mask)
            attn_mask = torch.where(attn_mask == 0,
                                     torch.zeros_like(attn_mask),
                                     attn_mask)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self,x):

        H,W = self.resolution
        B,N,C = x.shape
        h = x
        x = self.att_norm(x)
        x = x.reshape([B,H,W,C])
        if self.shift_size >0 :
            shift_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shift_x = x
        x_windows = windows_partition(shift_x,self.window_size)
        x_windows = x_windows.reshape([-1,self.window_size*self.window_size,C])
        attn_windows = self.attn(x_windows,mask = self.attn_mask)
        attn_windows = attn_windows.reshape([-1,self.window_size,self.window_size,C])
        shifted_x = window_reverse(attn_windows,self.window_size,H,W)
        if self.shift_size>0:
            x = torch.roll(shifted_x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            x = shifted_x
        x = x.reshape([B,-1,C])
        x = h+x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h+x
        return x

class SwinTransformerStage(nn.Module):
    def __init__(self,dim,input_resolution,depth,num_heads,window_size,patch_merging= None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # print(i)
            self.blocks.append(SwinBlock(dim = dim,input_resolution=input_resolution,num_heads=num_heads,window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size//2))
        if patch_merging is None:
            self.patch_merging = Identity()
        else:
            self.patch_merging = patch_merging(input_resolution,dim)
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        x = self.patch_merging(x)
        return x

# class Swin(nn.Module):
#     def __init__(self,
#                  image_size=224,
#                  patch_size=4,
#                  in_channels=3,
#                  embed_dim=96,
#                  window_size=7,
#                  num_heads=[3,6,12],
#                  depths = [2,2,6],
#                  ):
#         super().__init__()
#         self.in_channels=in_channels
#         self.depths = depths
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.num_stages = len(depths)
#         self.num_features = int(self.embed_dim * 2 ** (self.num_stages - 1))
#         self.patch_resolution = [image_size//patch_size,image_size//patch_size]
#         self.patch_embedding = PatchEmbedding(patch_size=patch_size,in_channels=in_channels,out_channels=embed_dim)
#         self.stages = nn.ModuleList()
#         for idx,(depth,num_heads) in enumerate(zip(self.depths,num_heads)):
#
#             stage = SwinTransformerStage(dim=int(self.embed_dim*2**idx),
#                                         input_resolution=(self.patch_resolution[0]//(2**idx),
#                                                           self.patch_resolution[0]//(2**idx)),
#                                         depth=depth,
#                                         num_heads=num_heads,
#                                         window_size=window_size,
#                                         patch_merging=PatchMerging if (idx < self.num_stages-1 ) else None )
#             self.stages.append(stage)
#         self.norm = nn.LayerNorm(self.num_features)
#         # self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.number=4
#         self.fc1 = nn.Linear(self.num_features,4*self.num_features)
#         self.fc2 = nn.Linear(self.num_features, 8 * self.num_features)
#         self.fc3 = nn.Linear(self.num_features, 32 *(self.num_features))
#     def forward(self,x):
#         B, C, H, W = x.shape
#         x = self.patch_embedding(x)
#         for stage in self.stages:
#             x = stage(x)
#         x = self.norm(x)
#         # x = x.permute([0,2,1])
#         # x = self.avgpool(x)
#         # x = x.flatten(1)
#         # x=x.permute(0,2,1)
#
#         if self.in_channels==3:
#             x = self.fc1(x)
#             C=int(32*C)
#             H=int(H/4)
#             W=int(W/4)
#             x = x.reshape([B, C, H, W])
#             return x
#         if self.in_channels==96:
#             x = self.fc2(x)
#             C=int(64*C)
#             H=int(H/8)
#             W=int(W/8)
#             x=x.reshape([B,C,H,W])
#             return x
#         if self.in_channels==192:
#             x = self.fc3(x)
#             C=int(128*C)
#             H=int(H/16)
#             W=int(W/16)
#             x = x.reshape([B, C, H, W])
#
#             return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

# if __name__ == '__main__':
#     # jy=PatchEmbedding(in_channels=1,out_channels=3)
#     # t=torch.rand(24,1,224,224)
#     # print(jy(t).shape) #torch.Size([24, 3136, 3])
#
#     # xi=PatchMerging([56,56],3)
#     # t=torch.rand([24,3136,3])
#     # print(xi(t).shape) #torch.Size([24, 784, 6])
#
#     # wi=window_attention(24,7,8)
#     # t=torch.rand([1,784,24])
#     # print(wi(t).shape)
# if __name__ == '__main__':
    #
    # model = Swin()
    # print(model)
#     t = torch.randn(24,3, 224, 224)
#     B,C,H,W=t.shape
#     print(B,C,H,W)
#     out = model(t)
#     out=out.reshape([B,C,H,W])
# #     print(out.shape)
# if __name__ == '__main__':
#     jy=Swin(in_channels=3,embed_dim=96,image_size=224)
#     t=torch.randn([24,3, 224, 224])
#     out=jy(t)
#     print(out.shape)

