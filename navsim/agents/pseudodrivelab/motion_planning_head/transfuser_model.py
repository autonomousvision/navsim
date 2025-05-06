from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index


class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(config.num_bounding_boxes, config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        # TrajectoryHead 대신 PDMScorer 사용
        self._pdm_scorer = PDMScorer(
            d_model=config.tf_d_model,
            d_ffn=config.tf_d_ffn,
            num_pdm_scores=config.num_pdm_scores,
            num_heads=config.tf_num_head,
            dropout=config.tf_dropout,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        # 궤적 vocabulary 입력 추가
        trajectory_vocab: torch.Tensor = features["trajectory_vocab"]  # [batch_size, voca_size, time_steps, features]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        # 이제 query_out을 직접 agents_query로 사용
        agents_query = query_out

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        
        # PDMScorer를 사용하여 궤적 vocabulary에 대한 점수 계산
        pdm_scores = self._pdm_scorer(trajectory_vocab, keyval)
        output.update(pdm_scores)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class PDMScorer(nn.Module):
    """PDM 점수 예측 헤드."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_pdm_scores: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        PDM 점수 예측 헤드 초기화
        :param d_model: 모델 차원 크기
        :param d_ffn: 피드포워드 네트워크 차원
        :param num_pdm_scores: PDM 점수 항목 개수
        :param num_heads: 멀티헤드 어텐션의 헤드 수
        :param dropout: 드롭아웃 비율
        """
        super(PDMScorer, self).__init__()
        
        # 궤적 인코더 - 단순 MLP 구조로 (batch_size, voca_size, time_steps, features) -> (batch_size, voca_size, d_model)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(8 * 3, d_ffn),  # 8 time steps, 3 features (x, y, heading)
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )
        
        # 크로스 어텐션 레이어
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 정규화 및 피드포워드 네트워크
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model)
        )
        
        # PDM 점수 출력 레이어
        self.score_head = nn.Linear(d_model, num_pdm_scores)
    
    def forward(self, trajectory_vocab, keyval):
        """
        PDM 점수 계산
        :param trajectory_vocab: 궤적 vocabulary, shape: [batch_size, voca_size, time_steps, features]
        :param keyval: BEV+상태 특징, shape: [batch_size, seq_len, d_model]
        :return: 각 궤적에 대한 PDM 점수를 포함하는 딕셔너리
        """
        batch_size, voca_size, time_steps, features = trajectory_vocab.shape
        
        # 궤적 인코딩
        traj_flat = trajectory_vocab.reshape(batch_size * voca_size, -1)  # [batch_size * voca_size, time_steps * features]
        traj_encoding = self.trajectory_encoder(traj_flat)  # [batch_size * voca_size, d_model]
        traj_encoding = traj_encoding.reshape(batch_size, voca_size, -1)  # [batch_size, voca_size, d_model]
        
        # BEV+상태 특징과 크로스 어텐션
        # keyval은 [batch_size, seq_len, d_model] 형태
        # 각 배치에 대해 크로스 어텐션 수행
        attn_outputs = []
        for i in range(batch_size):
            # [voca_size, d_model], [seq_len, d_model] -> [voca_size, d_model]
            attn_output, _ = self.cross_attention(
                query=traj_encoding[i],
                key=keyval[i],
                value=keyval[i]
            )
            attn_outputs.append(attn_output)
        
        attn_output = torch.stack(attn_outputs)  # [batch_size, voca_size, d_model]
        
        # 첫 번째 잔차 연결 및 정규화
        traj_features = self.norm1(traj_encoding + attn_output)
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(traj_features)
        
        # 두 번째 잔차 연결 및 정규화
        traj_features = self.norm2(traj_features + ffn_output)
        
        # PDM 점수 계산
        pdm_scores = self.score_head(traj_features)  # [batch_size, voca_size, num_pdm_scores]
        
        return {"pdm_scores": pdm_scores}


