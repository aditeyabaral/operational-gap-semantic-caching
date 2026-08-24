import time
from typing import ClassVar

import torch
from pylate import rank
from pylate.models import ColBERT
from redis import Redis
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.rerank import HFCrossEncoderReranker
from redisvl.utils.vectorize import HFTextVectorizer
from tqdm.auto import tqdm


class RetrieveAndRerankEvaluator:
    """Evaluator that retrieves candidates from a semantic cache and re-ranks them."""

    _DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    # _FA2_MODEL_KWARGS = {
    #     "attn_implementation": "flash_attention_2",
    #     "dtype": _DTYPE,
    # }
    _DEFAULT_MODEL_KWARGS: ClassVar[dict] = {
        "dtype": _DTYPE
    }  # HF picks sdpa automatically

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        biencoder_model_path: str,
        reranker_model_path: str,
        reranker_type: str,
        top_k: int,
        device: str = "cuda",
        flush_cache: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            redis_host: Hostname of the Redis server.
            redis_port: Port of the Redis server.
            biencoder_model_path: Path or name of the bi-encoder model used for semantic caching.
            reranker_model_path: Path or name of the re-ranker model.
            reranker_type: Type of re-ranker to use. One of "crossencoder" or "colbert".
            top_k: Number of top candidates to retrieve and re-rank.
            device: Device to run models on (e.g. "cpu", "cuda"). Default is "cuda".
            flush_cache: If True, flush the Redis cache on initialization. Default is False.
        """
        # set up Redis client
        self.redis_client = Redis(host=redis_host, port=redis_port)

        # flush the cache if requested, before any index setup
        if flush_cache:
            self.redis_client.flushall()

        # set up the embeddings cache
        self.embeddings_cache = EmbeddingsCache(
            redis_client=self.redis_client, ttl=86400
        )

        # set up the vectorizer
        self.vectorizer = HFTextVectorizer(
            model=biencoder_model_path,
            cache=self.embeddings_cache,
            device=device,
            model_kwargs=self._DEFAULT_MODEL_KWARGS,
            trust_remote_code=True,
        )

        # set up the semantic cache
        self.semantic_cache = SemanticCache(
            redis_client=self.redis_client,
            vectorizer=self.vectorizer,
            distance_threshold=2.0,  # set to max so we fetch all possible results
        )
        self.semantic_cache.set_ttl()

        # set up the re-ranker
        if reranker_type == "crossencoder":
            self.reranker = HFCrossEncoderReranker(
                model=reranker_model_path,
                limit=top_k,
                return_score=True,
                device=device,
                activation_fn=torch.nn.Identity(),
                model_kwargs=self._DEFAULT_MODEL_KWARGS,
            )

        elif reranker_type == "colbert":
            self.reranker = ColBERT(
                model_name_or_path=reranker_model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs=self._DEFAULT_MODEL_KWARGS,
            )

        else:
            raise ValueError(f"Invalid reranker_type: {reranker_type}")

        # store some metadata
        self.top_k = top_k
        self.device = device
        self.flush_cache = flush_cache
        self.biencoder_model_path = biencoder_model_path
        self.reranker_model_path = reranker_model_path
        self.reranker_type = reranker_type
        self.redis_host = redis_host
        self.redis_port = redis_port

    def populate_cache(self, sentences: list[str]):
        """Populate the semantic cache with a list of sentences.

        Args:
            sentences: Sentences to store in the cache.
        """
        for sentence in tqdm(sentences, desc="Encoding cache queries"):
            self.semantic_cache.store(
                prompt=sentence,
                response=sentence,
            )

    def get_cache_size(self) -> int:
        """Return the number of items currently stored in the semantic cache."""
        return self.semantic_cache.index.info()["num_docs"]

    def retrieve(self, query: str, num_results: int) -> list[dict]:
        """Retrieve candidate matches from the semantic cache.

        Args:
            query: Query string to search for.
            num_results: Number of candidates to retrieve.

        Returns:
            List of candidate dicts from the semantic cache.
        """
        return self.semantic_cache.check(
            prompt=query,
            num_results=num_results,
        )

    def rerank_crossencoder(
        self, query: str, candidates: list[str]
    ) -> tuple[list[str], list[float]]:
        """Re-rank candidates using a cross-encoder model.

        Args:
            query: Query string.
            candidates: List of candidate strings to re-rank.

        Returns:
            Tuple of (ranked candidate strings, reranking scores).
        """
        ranked_candidates, reranking_scores = self.reranker.rank(
            query=query,
            docs=candidates,
        )
        return [c["content"] for c in ranked_candidates], reranking_scores

    def rerank_colbert(
        self, query: str, candidates: list[str]
    ) -> tuple[list[str], list[float]]:
        """Re-rank candidates using a ColBERT model via pylate.

        Args:
            query: Query string.
            candidates: List of candidate strings to re-rank.

        Returns:
            Tuple of (ranked candidate strings, reranking scores).
        """
        query_embeddings = self.reranker.encode([query], is_query=True)
        candidate_embeddings = self.reranker.encode(candidates, is_query=False)
        candidate_ids = list(range(len(candidates)))

        reranked_results = rank.rerank(
            documents_ids=[candidate_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[candidate_embeddings],
        )

        reranked_for_query = reranked_results[0]
        ranked_candidates = []
        reranking_scores = []
        for item in reranked_for_query:
            ranked_candidates.append(candidates[item["id"]])
            reranking_scores.append(item["score"])

        return ranked_candidates, reranking_scores

    def rerank(
        self, query: str, candidates: list[str]
    ) -> tuple[list[str], list[float]]:
        """Re-rank candidates using the configured re-ranker.

        Dispatches to rerank_crossencoder or rerank_colbert based on reranker_type.

        Args:
            query: Query string.
            candidates: List of candidate strings to re-rank.

        Returns:
            Tuple of (ranked candidate strings, reranking scores).
        """
        if self.reranker_type == "crossencoder":
            ranked_candidates, reranking_scores = self.rerank_crossencoder(
                query, candidates
            )
        elif self.reranker_type == "colbert":
            ranked_candidates, reranking_scores = self.rerank_colbert(query, candidates)
        else:
            raise ValueError(f"Invalid reranker_type: {self.reranker_type}")

        sorted_pairs = sorted(
            zip(reranking_scores, ranked_candidates), key=lambda x: x[0], reverse=True
        )
        reranking_scores = [pair[0] for pair in sorted_pairs]
        ranked_candidates = [pair[1] for pair in sorted_pairs]
        return ranked_candidates, reranking_scores

    def evaluate(
        self, queries: list[str], ground_truths: list[str], labels: list[int]
    ) -> dict:
        """Run the full retrieve-and-rerank pipeline over a set of queries and return evaluation results.

        Args:
            queries: List of query strings to evaluate.
            ground_truths: List of expected answer strings, one per query.
            labels: List of integer labels indicating match/no-match, one per query.

        Returns:
            Dict containing per-query results and aggregated timing metrics.
        """
        results = []
        zipped = list(zip(queries, ground_truths, labels))
        total_items = len(zipped)
        for query, ground_truth, label in tqdm(
            zipped,
            desc=f"Retrieving and re-ranking top-{self.top_k} results",
            total=total_items,
        ):
            # retrieve top-k results for each test query
            torch.cuda.synchronize()
            retrieval_start_time = time.perf_counter()
            retrieved_candidates = self.retrieve(
                query=query,
                num_results=self.top_k,
            )
            torch.cuda.synchronize()
            retrieval_end_time = time.perf_counter()

            # process the candidates and get the text and score
            candidates, retrieval_scores = [], []
            for candidate in retrieved_candidates:
                candidates.append(candidate["prompt"])
                retrieval_scores.append(
                    1 - candidate["vector_distance"]
                )  # convert distance to similarity

            # re-rank the top-k candidates to fetch the match
            torch.cuda.synchronize()
            reranking_start_time = time.perf_counter()
            ranked_candidates, reranking_scores = self.rerank(
                query=query,
                candidates=candidates,
            )
            torch.cuda.synchronize()
            reranking_end_time = time.perf_counter()

            # store the results
            results.append(
                {
                    "query": query,
                    "ground_truth": ground_truth,
                    "label": label,
                    "retrieved_candidates": candidates,
                    "retrieved_scores": retrieval_scores,
                    "ranked_candidates": ranked_candidates,
                    "ranked_scores": reranking_scores,
                    "retrieval_duration": retrieval_end_time - retrieval_start_time,
                    "reranking_duration": reranking_end_time - reranking_start_time,
                    "total_duration": reranking_end_time - retrieval_start_time,
                }
            )

        # Compute aggregate timing metrics
        retrieval_duration = sum(x["retrieval_duration"] for x in results)
        reranking_duration = sum(x["reranking_duration"] for x in results)
        total_duration = sum(x["total_duration"] for x in results)

        return {
            "retrieval_duration": retrieval_duration,
            "reranking_duration": reranking_duration,
            "total_duration": total_duration,
            "results": results,
        }
