from sentence_transformers.util import pairwise_cos_sim

from src.reranker.cache_evaluator import CacheEvaluator


class BiEncoderCacheEvaluator(CacheEvaluator):
    """CacheEvaluator for bi-encoder (SentenceTransformer) models.

    Each pair is scored by the cosine similarity of its two embeddings, i.e. 1 minus the
    cosine distance that the Redis semantic cache returns at deployment. All metrics
    (PR AUC, PCHR/PCHR-VALID AUC, logit overlap) are inherited from CacheEvaluator.
    """

    def _compute_predictions(self, model):
        sentences_a = [pair[0] for pair in self.sentence_pairs]
        sentences_b = [pair[1] for pair in self.sentence_pairs]
        embeddings_a = model.encode(
            sentences_a,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self.device,
        )
        embeddings_b = model.encode(
            sentences_b,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self.device,
        )
        # cosine in fp32
        return (
            pairwise_cos_sim(embeddings_a.float(), embeddings_b.float()).cpu().tolist()
        )
