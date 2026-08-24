import numpy as np
import torch
from scipy.stats import gaussian_kde
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm


class CacheEvaluator(SentenceEvaluator):
    def __init__(
        self, sentence_pairs, labels, batch_size=32, name="cache", device="cuda"
    ):
        """Evaluator that computes the following metrics:
        - Precision-Recall AUC
        - Precision-Cache Hit Ratio AUC
        - Precision-Valid Cache Hit Ratio AUC
        - Logit overlap
        higher AUC -> better model performance.
        lower logit overlap -> better separation between positive and negative scores.

        Args:
            sentence_pairs: List of sentence pairs.
            labels: List of labels.
            batch_size: Batch size.
            name: Name of the evaluator.
            device: Device to run the evaluation on.

        Returns:
            Dictionary with PR AUC, PCHR/PCHR-VALID AUC and logit overlap.
        """
        super().__init__()
        self.sentence_pairs = sentence_pairs
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.name = name
        self.device = device
        self.primary_metric = f"{self.name}_pr_auc"

    def _compute_predictions(self, model):
        preds = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(self.sentence_pairs), self.batch_size), desc="Evaluating"
            ):
                batch = self.sentence_pairs[i : i + self.batch_size]
                if hasattr(model, "predict"):
                    # Standard models with predict method
                    batch_preds = model.predict(batch)
                    preds.extend(batch_preds)
                else:
                    # ColBERT model: compute MaxSim scores
                    sentences_a = [pair[0] for pair in batch]
                    sentences_b = [pair[1] for pair in batch]

                    # Encode queries and documents
                    # DON'T use padding=True - we need variable-length token embeddings
                    query_embeddings = model.encode(
                        sentences_a,
                        convert_to_tensor=True,
                        padding=False,
                        is_query=True,
                    )
                    doc_embeddings = model.encode(
                        sentences_b,
                        convert_to_tensor=True,
                        padding=False,
                        is_query=False,
                    )

                    # Compute ColBERT MaxSim scores for each pair
                    batch_preds = []
                    for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
                        # q_emb shape: [num_query_tokens, embedding_dim]
                        # d_emb shape: [num_doc_tokens, embedding_dim]
                        # Step 1: Compute similarity matrix between all query and doc tokens
                        # Result shape: [num_query_tokens, num_doc_tokens]
                        similarity_matrix = torch.matmul(q_emb, d_emb.T)
                        # Step 2: For each query token, find max similarity with any doc token
                        # Result shape: [num_query_tokens]
                        max_sim_scores = similarity_matrix.max(dim=1).values
                        # Step 3: Mean over query tokens to get final score
                        # We use mean so the score is usable in a classification setting
                        score = max_sim_scores.mean().item()
                        batch_preds.append(score)

                    preds.extend(batch_preds)

        return preds

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, **kwargs):
        model.eval()
        preds = np.array(self._compute_predictions(model))

        # Compute PR AUC
        precision, recall, thresholds = precision_recall_curve(self.labels, preds)
        pr_auc = auc(recall, precision)

        # Compute PCHR AUC
        cache_hit_ratios = []
        valid_cache_hit_ratios = []
        num_labels = len(self.labels)
        for threshold in thresholds:
            y_pred = (preds >= threshold).astype(int)
            tp = np.sum((self.labels == 1) & (y_pred == 1))
            fp = np.sum((self.labels == 0) & (y_pred == 1))
            cache_hit_ratio = (tp + fp) / num_labels if num_labels > 0 else 0.0
            valid_cache_hit_ratio = tp / num_labels if num_labels > 0 else 0.0
            cache_hit_ratios.append(cache_hit_ratio)
            valid_cache_hit_ratios.append(valid_cache_hit_ratio)

        precisions_for_auc = precision[:-1]  # Preserve a copy for AUC computation

        # Compute AUC for CHR
        if len(cache_hit_ratios) > 1:
            sorted_indices = np.argsort(cache_hit_ratios)
            cache_hit_ratios = np.array(cache_hit_ratios)[sorted_indices]
            precisions_chr = np.array(precisions_for_auc)[sorted_indices]
            precision_chr_auc = auc(cache_hit_ratios, precisions_chr)
        else:
            precision_chr_auc = 0.0

        # Compute AUC for Valid CHR
        if len(valid_cache_hit_ratios) > 1:
            sorted_indices = np.argsort(valid_cache_hit_ratios)
            valid_cache_hit_ratios = np.array(valid_cache_hit_ratios)[sorted_indices]
            precisions_chr = np.array(precisions_for_auc)[sorted_indices]
            precision_valid_chr_auc = auc(valid_cache_hit_ratios, precisions_chr)
        else:
            precision_valid_chr_auc = 0.0

        # Compute area of overlap in score distributions
        scores_pos = preds[self.labels == 1]
        scores_neg = preds[self.labels == 0]
        if len(scores_pos) > 1 and len(scores_neg) > 1:
            kde_pos = gaussian_kde(scores_pos)
            kde_neg = gaussian_kde(scores_neg)

            x_min = min(scores_pos.min(), scores_neg.min())
            x_max = max(scores_pos.max(), scores_neg.max())
            x_grid = np.linspace(x_min, x_max, 1000)

            pdf_pos = kde_pos(x_grid)
            pdf_neg = kde_neg(x_grid)

            bin_width = x_grid[1] - x_grid[0]
            overlap_area = np.sum(np.minimum(pdf_pos, pdf_neg) * bin_width)
        else:
            # Not enough samples to compute KDE
            overlap_area = np.nan

        return {
            f"{self.name}_pr_auc": pr_auc,
            f"{self.name}_logit_overlap": overlap_area,
            f"{self.name}_chr_auc": precision_chr_auc,
            f"{self.name}_valid_chr_auc": precision_valid_chr_auc,
        }
