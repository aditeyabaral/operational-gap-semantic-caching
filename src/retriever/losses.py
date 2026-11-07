import math
from collections.abc import Iterable
from typing import Any

import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import all_gather_with_grad
from torch import Tensor, nn


class ArcFaceInBatchLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        gather_across_devices: bool = False,
        margin: float = 0.0,
    ) -> None:
        """In-batch contrastive loss with an ArcFace additive angular margin.

        Given (anchor, positive) pairs or (anchor, positive, negative_1, ..., negative_n) tuples, each anchor is
        scored against every candidate in the batch (all positives and all hard negatives). The margin is added to
        the angle of the paired positive only, i.e. its logit becomes cos(theta + margin), so the anchor must beat
        every other candidate by an angular margin. With margin=0 this reduces to MultipleNegativesRankingLoss, up to
        the clamp of cosine scores to [-1 + 1e-4, 1 - 1e-4].

        Args:
            model: SentenceTransformer model.
            scale: Output of the similarity function is multiplied by this value (inverse temperature).
            similarity_fct: Similarity function between anchors and candidates. Should be cosine similarity.
            gather_across_devices: If True, gather the candidates across all devices before computing the loss.
            margin: Additive angular margin in radians applied to the positive pair.
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.margin = margin
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.eps = 1e-4

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(
        self, embeddings: list[Tensor], labels: Tensor
    ) -> Tensor:
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = embeddings[1:]  # (1 + num_negatives) x (batch_size, embedding_dim)
        batch_size = anchors.size(0)
        offset = 0

        if self.gather_across_devices:
            # gather only the candidates, so this device's anchors are scored against every device's candidates
            candidates = [all_gather_with_grad(column) for column in candidates]
            if torch.distributed.is_initialized():
                offset = torch.distributed.get_rank() * batch_size

        # (batch_size, batch_size * world_size * (1 + num_negatives))
        scores = self.similarity_fct(anchors, torch.cat(candidates, dim=0))
        # keep sin(theta) and its gradient finite when |cos(theta)| reaches 1, e.g. an anchor that is its own positive
        scores = scores.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # anchor[i] is paired with candidate[offset + i]
        rows = torch.arange(batch_size, device=scores.device)
        targets = rows + offset

        # cos(theta + m) = cos(theta) cos(m) - sin(theta) sin(m), on the positive column only
        cos_pos = scores[rows, targets]
        sin_pos = torch.sqrt((1.0 - cos_pos * cos_pos).clamp(min=self.eps))
        margin_pos = cos_pos * self.cos_m - sin_pos * self.sin_m
        logits = scores.scatter(1, targets.unsqueeze(1), margin_pos.unsqueeze(1))

        return nn.functional.cross_entropy(logits * self.scale, targets)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
            "margin": self.margin,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{deng2019arcface,
    title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
    author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={4690--4699},
    year={2019}
}
"""
