import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.d_model = d_model
        pos_h = torch.arange(height).unsqueeze(1).expand(height, width).reshape(1, height * width)
        pos_w = torch.arange(width).unsqueeze(0).expand(height, width).reshape(1, height * width)
        
        pe = torch.zeros(1, height * width, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, :, 0::2] = torch.sin(pos_h.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(pos_h.unsqueeze(-1) * div_term)
        
        if d_model > 2:  # 채널이 2개 이상일 때만 width 위치 정보 추가
            offset = min(d_model // 2, 2)  # 절반의 채널 사용하거나 최소 2개 채널 사용
            pe[:, :, offset::2] += torch.sin(pos_w.unsqueeze(-1) * div_term)[:, :, :(d_model-offset)//2]
            pe[:, :, offset+1::2] += torch.cos(pos_w.unsqueeze(-1) * div_term)[:, :, :(d_model-offset-1)//2]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where L is sequence length (H*W)
        Returns:
            x + positional encoding
        """
        return x + self.pe

class TrajectoryEncoder(nn.Module):
    def __init__(self, time_steps=8, feature_dim=3, hidden_dim=128, output_dim=256):
        """
        Args:
            time_steps (int): 궤적의 시간 스텝 수 (8)
            feature_dim (int): 각 시간 스텝의 특징 차원 (3: x, y, heading)
            hidden_dim (int): 히든 레이어 차원
            output_dim (int): 출력 임베딩 차원
        """
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        
        # 전체 궤적을 한번에 인코딩하는 Time-Aware 접근법
        
        # 1. 시공간 합성곱 네트워크 (시간과 특징을 함께 고려)
        # (batch, time_steps, feature_dim) -> (batch, hidden_dim)
        self.spatiotemporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 전역 풀링
        )
        
        # 2. 궤적 전체를 보는 Transformer 블록
        self.pos_embedding = nn.Parameter(torch.randn(1, time_steps, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. 궤적 특징 프로젝션 (시간 차원을 임베딩 전에 통합)
        self.projection = nn.Linear(time_steps * feature_dim, hidden_dim)
        
        # 최종 출력 레이어 (다양한 인코딩 방식의 결합)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D) - 배치 크기, 시간 스텝, 특징 차원 (x, y, heading)
        Returns:
            (B, output_dim) - 인코딩된 궤적
        """
        B, T, D = x.shape
        
        # 1. 시공간 합성곱 인코딩
        # (B, T, D) -> (B, D, T)
        x_conv = x.transpose(1, 2)
        conv_features = self.spatiotemporal_conv(x_conv).squeeze(-1)  # (B, hidden_dim)
        
        # 2. Transformer 인코딩
        # (B, T, D) -> (B, T, hidden_dim)
        x_trans = nn.Linear(D, self.pos_embedding.size(-1))(x)
        x_trans = x_trans + self.pos_embedding
        trans_features = self.transformer(x_trans)  # (B, T, hidden_dim)
        trans_features = trans_features.mean(dim=1)  # (B, hidden_dim)
        
        # 3. 전체 궤적 직접 프로젝션
        # (B, T, D) -> (B, T*D)
        x_flat = x.reshape(B, -1)
        proj_features = self.projection(x_flat)  # (B, hidden_dim)
        
        # 인코딩 방식 결합
        combined_features = torch.cat([
            conv_features,    # 지역적 시공간 패턴
            trans_features,   # 전역적 시간 관계
            proj_features     # 직접 프로젝션
        ], dim=1)
        
        # 최종 출력
        output = self.output_layer(combined_features)
        
        return output

class BEVFeatureProcessor(nn.Module):
    def __init__(self, input_channels, output_dim, h=81, w=81):
        """
        Args:
            input_channels (int): BEV 특징 맵 입력 채널 수
            output_dim (int): 출력 임베딩 차원
            h, w (int): BEV 특징 맵의 높이와 너비
        """
        super().__init__()
        
        # CNN 블록들로 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 출력 프로젝션
        self.projection = nn.Conv2d(64, output_dim, kernel_size=1)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding2D(output_dim, h, w)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - BEV 특징 맵
        Returns:
            features: (B, H*W, output_dim) - 위치 인코딩이 적용된 특징
            shape_info: (H, W) - 원래 공간 차원 (필요시 활용)
        """
        B, C, H, W = x.shape
        
        # CNN으로 특징 추출
        x = self.cnn(x)  # (B, 64, H, W)
        x = self.projection(x)  # (B, output_dim, H, W)
        
        # 위치 인코딩을 위한 형태 변환
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, output_dim)
        x = x.view(B, H*W, -1)  # (B, H*W, output_dim)
        
        # 위치 인코딩 적용
        x = self.pos_encoding(x)  # (B, H*W, output_dim)
        
        return x, (H, W)

class TrajectoryBEVCrossAttentionSelector(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1, num_attention_layers=2):
        """
        Args:
            d_model (int): 모델의 임베딩 차원
            nhead (int): 멀티헤드 어텐션의 헤드 수
            dropout (float): 드롭아웃 비율
            num_attention_layers (int): 쌓을 Cross-Attention 레이어 개수
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_attention_layers = num_attention_layers

        # Trajectory Encoder (8 time steps, 3 features per step)
        self.trajectory_encoder = TrajectoryEncoder(
            time_steps=8,
            feature_dim=3,
            hidden_dim=128,
            output_dim=d_model
        )
        
        # BEV Feature Processor (예상 입력: 256 채널, 81x81 크기)
        self.bev_processor = BEVFeatureProcessor(
            input_channels=256,
            output_dim=d_model,
            h=81,
            w=81
        )
        
        # 여러 개의 Cross-Attention 블록 생성
        self.attention_layers = nn.ModuleList()
        for _ in range(self.num_attention_layers):
            self.attention_layers.append(
                nn.ModuleDict({
                    'cross_attn': nn.MultiheadAttention(
                        embed_dim=d_model,
                        num_heads=nhead,
                        dropout=dropout,
                        batch_first=True
                    ),
                    'norm1': nn.LayerNorm(d_model),
                    'norm2': nn.LayerNorm(d_model),
                    'dropout1': nn.Dropout(dropout),
                    'dropout2': nn.Dropout(dropout),
                    'ffn': nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model * 4, d_model)
                    )
                })
            )
        
        # 최종 점수 예측
        self.scoring_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, trajectory_vocab, bev_features):
        """
        Args:
            trajectory_vocab (Tensor): (B, V, 8, 3) - B: Batch size, V: Vocabulary size
                                       8: Time steps, 3: x, y, heading
            bev_features (Tensor): (B, C, H, W) - C: Channels (예: 256), H,W: 높이/너비 (예: 81)

        Returns:
            trajectory_scores (Tensor): (B, V) - 각 궤적 후보에 대한 점수
        """
        B, V, T, D = trajectory_vocab.shape
        
        # 1. Trajectory Encoding
        traj_flat = trajectory_vocab.view(-1, T, D)
        traj_embeddings = self.trajectory_encoder(traj_flat)
        traj_query = traj_embeddings.view(B, V, self.d_model)
        
        # 2. BEV Feature Processing
        bev_embeddings, (bev_h, bev_w) = self.bev_processor(bev_features)
        
        # 3. 여러 Cross-Attention 레이어 통과
        attn_output = traj_query # 첫 입력은 traj_query
        for layer in self.attention_layers:
            # Cross-Attention
            attn_intermediate, attn_weights = layer['cross_attn'](
                query=attn_output, # 이전 레이어 출력을 query로 사용
                key=bev_embeddings,
                value=bev_embeddings
            )
            # 첫번째 잔차 연결 및 정규화
            attn_output = layer['norm1'](attn_output + layer['dropout1'](attn_intermediate))

            # Feed Forward Network
            ff_intermediate = layer['ffn'](attn_output)
            # 두번째 잔차 연결 및 정규화
            attn_output = layer['norm2'](attn_output + layer['dropout2'](ff_intermediate))

        # 4. 궤적 점수 예측
        output_flat = attn_output.reshape(-1, self.d_model)
        scores_flat = self.scoring_head(output_flat)
        trajectory_scores = scores_flat.view(B, V)
        
        return trajectory_scores
