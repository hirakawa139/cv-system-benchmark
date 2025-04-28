import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class Patching(nn.Module):
    """画像をパッチに分割するモジュール
    """
    def __init__(self, patch_size):
        """初期化

        Args:
            patch_size (int): パッチの縦の長さ（= 横の長さ）
        """        
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b(h w) (ph pw c)", ph = patch_size, pw = patch_size)

    def forward(self, x):
        """画像データをパッチに分割する

        Args:
            x (torch.Tensor): 画像データ
                x.shape = torch.Size([batch_size, channeles, img_height, img_width])
        """
        x = self.net(x)
        return x        

class LinerProjection(nn.Module):
    """パッチを線形変換するモジュール
    """
    def __init__(self, patch_dim, dim):
        """初期化

        Args:
            patch_dim (int): 一枚あたりのパッチの次元（= channeles * (patch_size ** 2)）
            dim (int): パッチが変換されたベクトルの次元
        """        
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        """パッチを線形変換する

        Args:
            x (torch.Tensor): パッチデータ
                x.shape = torch.Size([batch_size, n_patches, patch_dim])
        """
        x = self.net(x)
        return x

class Embedding(nn.Module):
    """パッチを埋め込むモジュール
    """
    def __init__(self, dim, n_pathches):
        """初期化

        Args:
            dim (int): パッチが変換されたベクトルの次元
            n_pathches (int): パッチの枚数
        """        
        super().__init__()
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_pathches + 1, dim))

    def forward(self, x):
        """パッチを埋め込む

        Args:
            x (torch.Tensor): パッチデータ
                x.shape = torch.Size([batch_size, n_patches, dim])
        """
        # バッチサイズを抽出
        batch_size, _, __ = x.shape

        # [class] トークン付加
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)

        # 位置エンコーディング加算
        x += self.pos_embedding
        return x

class MLP(nn.Module):
    """MLPモジュール
    """
    def __init__(self, dim, hidden_dim):
        """初期化
        Args:
            dim (int): 各パッチが変換されたベクトルの長さ
            hidden_dim (int): 隠れ層のノード数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        """MLPの順伝播

        Args:
            x (torch.Tensor): 
                x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """        
        x = self.net(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Attentionモジュール
    """
    def __init__(self, dim, n_heads):
        """初期化

        Args:
            dim (int): 各パッチが変換されたベクトルの長さ
            n_heads (int): Multi-Head Attentionのヘッド数
        """        
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)

        self.softmax = nn.Softmax(dim = -1)

        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        """Multi-Head Attentionの順伝播

        Args:
            x (torch.Tensor): パッチごとに変換されたベクトル
                x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # Logit[i] = Q[i] * tK[i] / sqrt(D) (i = 1, 2, ..., n_heads)
        # AttentionWeight[i] = Softmax(Logit[i]) (i = 1, 2, ..., n_heads)
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)

        # Head[i] = AttentionWeight[i] * V[i] (i = 1, 2, ..., n_heads)
        # Output = concat[Head[i], ..., Head[n_heads]] 
        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output

class TransformerEncoder(nn.Module):
    """Transformer Encoderモジュール
    """
    def __init__(self, dim, n_heads, mlp_dim, depth):
        """初期化

        Args:
            dim (int): 各パッチが変換されたベクトルの長さ
            n_heads (int): Multi-Head Attentionのヘッド数
            mlp_dim (int): MLPの隠れ層のノード数
            depth (int): Transformer Encoderの層の深さ
        """        
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim = dim, n_heads = n_heads)
        self.mlp = MLP(dim = dim, hidden_dim = mlp_dim)
        self.depth = depth

    def forward(self, x):
        """Transformer Encoderの順伝播

        Args:
            x (torch.Tensor): パッチごとに変換されたベクトル
                x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """        
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x
        return x

class MLPHead(nn.Module):
    """MLP Headモジュール
    """
    def __init__(self, dim, out_dim):
        """初期化

        Args:
            dim (int): パッチ数？（ここまでの処理から、[class]トークンに対応する部分を受け取ってる？）
            out_dim (int): 出力の次元
        """        
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        """MLP Headの順伝播
        """
        x = self.net(x)
        return x

class ViT(nn.Module):
    """Vision Transformerモデル
    """
    def __init__(self, img_size, patch_size, n_classes, dim, depth, n_heads, channeles = 3, mlp_dim = 256):
        """初期設定

        Args:
            img_size (int): 画像の縦の長さ（= 横の長さ）
            patch_size (int): パッチの縦の長さ（= 横の長さ）
            n_classes (int): 分類するクラスの数
            dim (int): 各パッチのベクトルが変換されたベクトルの長さ
            depth (int): Transformer Encoderの層の深さ
            n_heads (int): Multi-Head Attentionのヘッド数
            channeles (int, optional): 入力のチャネル数. Defaults to 3.
            mlp_dim (int, optional): MLPの隠れ層のノード数. Defaults to 256.
        """        

        super().__init__()

        # Params
        n_pathches = (img_size // patch_size) ** 2
        patch_dim = channeles * patch_size ** 2
        self.depth = depth

        # Layers
        self.patching = Patching(patch_size = patch_size)
        self.liner_projection_of_flattened_patches = LinerProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_pathches = n_pathches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)
    
    def forward(self, img):
        """順伝播

        Args:
            img (torch.Tensor): 入力画像
                img.shape = torch.Size([batch_size, channeles, img_height, img_width])

        Returns:
            torch.Tensor: 分類結果
        """        

        x = img

        # 1. パッチに分割
        # x.shape = [batch_size, channeles, img_height, img_width] -> [batch_size, n_patches, channeles * (patch_size ** 2)]
        x = self.patching(x)

        # 2. 各パッチをベクトルに変換
        # x.shape = [batch_size, n_patches, channeles * (patch_size ** 2)] -> [batch_size, n_patches, dim]
        x = self.liner_projection_of_flattened_patches(x)

        # 3. トークン付加 + 位置エンコーディング
        # x.shape = [batch_size, n_patches, dim] -> [batch_size, n_patches + 1, dim]
        x = self.embedding(x)

        # 4. Transformer Encoder
        # x.shape = No Change
        x = self.transformer_encoder(x)

        # 5. 出力の0番目のベクトルを MLP Head で処理
        x = x[:, 0]
        x = self.mlp_head(x)

        return x