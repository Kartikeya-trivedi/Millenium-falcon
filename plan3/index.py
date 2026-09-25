"""Exact sparse retrieval with full-partition IDF and bounded target shards.

No labels enter this module. Query and target IDs are variable-length strings.
The word channel uses the same TF-IDF definition as Plan 1 v2.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .artifacts import contract, parquet, read_json, sha256, write_json

CHANNELS = {
    "word": {"field": "retrieval_text", "analyzer": "word", "missing_only": False},
    "char": {"field": "retrieval_text", "analyzer": "char", "missing_only": False},
    "name": {"field": "name_red", "analyzer": "char", "missing_only": False},
    "missing": {"field": "name_red", "analyzer": "char", "missing_only": True},
}


def vectorizer(channel: str, vocabulary=None):
    if CHANNELS[channel]["analyzer"] == "word":
        return CountVectorizer(analyzer=str.split, lowercase=False, dtype=np.float32, vocabulary=vocabulary)
    return CountVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False,
                           dtype=np.float32, vocabulary=vocabulary)


def save_sparse(path: Path, x) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        sparse.save_npz(f, x, compressed=True)
    os.replace(tmp, path)


def apply_idf(x, idf):
    x = x.tocsr().astype(np.float32, copy=False)
    x.sum_duplicates()
    if x.shape[1] == 0:
        return x
    np.log(x.data, out=x.data)
    x.data += 1
    x.data *= idf[x.indices]
    return normalize(x, norm="l2", copy=False)


def remap_counts(x, terms, vocabulary, idf):
    mapping = np.fromiter((vocabulary.get(t, -1) for t in terms), np.int32, count=len(terms))
    keep = np.flatnonzero(mapping >= 0)
    x = x[:, keep].tocsr()
    x.indices = mapping[keep][x.indices]
    x._shape = (x.shape[0], len(vocabulary))
    x.sort_indices()
    return apply_idf(x, idf)


def choose_vocabulary(stats: pl.DataFrame, n: int, min_df: int, max_features: int) -> pl.DataFrame:
    """Match CountVectorizer's frequency selection, then sklearn's smoothed IDF."""
    stats = stats.sort("term")
    df, tf = stats["df"].to_numpy(), stats["tf"].to_numpy()
    mask = df >= min_df
    if int(mask.sum()) > max_features:
        chosen = np.flatnonzero(mask)[(-tf[mask]).argsort()[:max_features]]
        mask[:] = False
        mask[chosen] = True
    selected = stats.filter(pl.Series(mask))
    idf = np.log(np.float32(n + 1) / (selected["df"].to_numpy().astype(np.float32) + 1)) + 1
    return selected.with_columns(idf=pl.Series(idf.astype(np.float32))).with_row_index("feature")


class SparseIndex:
    def __init__(self, path: Path):
        self.path = path
        self.meta = read_json(path / "index.json")
        self.channel = self.meta["channel"]
        v = pl.read_parquet(path / "vocabulary.parquet")
        self.vocabulary = dict(zip(v["term"].to_list(), v["feature"].to_list()))
        self.idf = v["idf"].to_numpy()
        self.ids = pl.read_parquet(path / "ids.parquet")["target_id"]
        self.vec = vectorizer(self.channel, self.vocabulary) if self.vocabulary else None

    @classmethod
    def build(cls, path: Path, targets: pl.DataFrame, channel: str, partition_sha: str,
              shard_size: int = 50_000, max_features: int = 300_000, min_df: int = 2):
        from .artifacts import versions

        spec = CHANNELS[channel]
        targets = targets.filter(pl.col("addr_missing")) if spec["missing_only"] else targets
        targets = targets.select("entity_id", spec["field"]).sort("entity_id")
        expected = {"format": 1, "channel": channel, "spec": spec, "rows": targets.height,
                    "partition_sha256": partition_sha, "shard_size": shard_size,
                    "max_features": max_features, "min_df": min_df,
                    "index_code": sha256(Path(__file__)), "dependencies": versions()}
        contract(path / "contract.json", expected)
        if (path / "index.json").exists():
            return cls(path)
        parquet(path / "ids.parquet", targets.select(target_id="entity_id"))
        # Counts and document frequencies must use the entire target partition.
        parts = []
        for start in range(0, targets.height, shard_size):
            key = f"{start:09d}"
            marker = path / f"{key}.count.json"
            if not marker.exists():
                texts = targets.slice(start, shard_size)[spec["field"]].to_list()
                vec = vectorizer(channel)
                try:
                    x = vec.fit_transform(texts).tocsr()
                    terms = vec.get_feature_names_out().tolist()
                except ValueError as error:
                    if "empty vocabulary" not in str(error):
                        raise
                    x, terms = sparse.csr_matrix((len(texts), 0), dtype=np.float32), []
                x.sum_duplicates()
                stat = pl.DataFrame({
                    "term": pl.Series(terms, dtype=pl.String),
                    "df": pl.Series(np.bincount(x.indices, minlength=x.shape[1]).astype(np.int64)),
                    "tf": pl.Series(np.asarray(x.sum(axis=0)).ravel().astype(np.int64)),
                })
                save_sparse(path / f"{key}.counts.npz", x)
                parquet(path / f"{key}.terms.parquet", stat)
                write_json(marker, {"start": start, "rows": len(texts)})
                del x, stat, vec, texts, terms
                gc.collect()
            parts.append(read_json(marker))
            if (start // shard_size) % 10 == 0:
                print(f"    {channel} counts {min(start+shard_size,targets.height):,}/{targets.height:,}", flush=True)
        stats = pl.DataFrame(schema={"term": pl.String, "df": pl.Int64, "tf": pl.Int64})
        for part in parts:
            key = f"{part['start']:09d}"
            stats = (pl.concat([stats, pl.read_parquet(path / f"{key}.terms.parquet")])
                     .group_by("term").agg(pl.col("df").sum(), pl.col("tf").sum()))
        vocab = choose_vocabulary(stats, targets.height, min_df, max_features)
        print(f"    {channel}: {vocab.height:,} vocabulary terms from {targets.height:,} full-pool targets", flush=True)
        parquet(path / "vocabulary.parquet", vocab)
        mapping = dict(zip(vocab["term"].to_list(), vocab["feature"].to_list()))
        idf = vocab["idf"].to_numpy()
        del stats, vocab
        for part in parts:
            key = f"{part['start']:09d}"
            target = path / f"{key}.tfidf.npz"
            if target.exists():
                continue
            counts = sparse.load_npz(path / f"{key}.counts.npz")
            terms = pl.read_parquet(path / f"{key}.terms.parquet")["term"].to_list()
            x = remap_counts(counts, terms, mapping, idf)
            save_sparse(target, x)
            del counts, terms, x
            gc.collect()
        write_json(path / "index.json", {**expected, "parts": parts, "features": len(mapping)})
        # Remove only redundant intermediate count matrices after the committed index exists.
        for part in parts:
            (path / f"{part['start']:09d}.counts.npz").unlink(missing_ok=True)
        return cls(path)

    def transform(self, texts):
        if self.vec is None:
            return sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        return apply_idf(self.vec.transform(texts), self.idf)

    def query(self, queries: pl.DataFrame, k: int, threads: int = 4) -> pl.DataFrame:
        if k <= 0:
            raise ValueError("k must be positive")
        schema = {"s1_id": pl.String, "target_id": pl.String, "score": pl.Float32, "rank": pl.Int32}
        if not self.vocabulary or not self.ids.len() or not queries.height:
            return pl.DataFrame(schema=schema)
        qm = self.transform(queries[CHANNELS[self.channel]["field"]].to_list())
        # Query batches bound these arrays. Each shard is searched against all batch queries.
        best_ids = np.full((queries.height, k), -1, np.int64)
        best_scores = np.zeros((queries.height, k), np.float32)
        for part in self.meta["parts"]:
            x = sparse.load_npz(self.path / f"{part['start']:09d}.tfidf.npz")
            result = sp_matmul_topn(qm, x.T.tocsr(), top_n=min(k, x.shape[0]), sort=True,
                                   n_threads=threads).tocsr()
            incoming_ids = np.full_like(best_ids, -1)
            incoming_scores = np.zeros_like(best_scores)
            rows = np.repeat(np.arange(queries.height), np.diff(result.indptr))
            columns = np.arange(len(result.data)) - np.repeat(result.indptr[:-1], np.diff(result.indptr))
            incoming_ids[rows, columns] = result.indices.astype(np.int64) + part["start"]
            incoming_scores[rows, columns] = result.data
            all_ids = np.concatenate((best_ids, incoming_ids), axis=1)
            all_scores = np.concatenate((best_scores, incoming_scores), axis=1)
            order = np.lexsort((all_ids, -all_scores), axis=1)[:, :k]
            best_ids = np.take_along_axis(all_ids, order, axis=1)
            best_scores = np.take_along_axis(all_scores, order, axis=1)
            del x, result, incoming_ids, incoming_scores, all_ids, all_scores, order
        rows, ranks = np.nonzero((best_ids >= 0) & (best_scores > 0))
        return pl.DataFrame({
            "s1_id": queries["entity_id"].gather(pl.Series(rows.astype(np.uint32))),
            "target_id": self.ids.gather(pl.Series(best_ids[rows, ranks].astype(np.uint32))),
            "score": best_scores[rows, ranks],
            "rank": (ranks + 1).astype(np.int32),
        })

    def pair_scores(self, queries: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
        """Actual cosine for arbitrary proposals; a missing word-channel rank is not zero cosine."""
        if not pairs.height:
            return pairs.select("s1_id", "target_id").with_columns(score=pl.lit(None, pl.Float32))
        qi = queries.select(s1_id="entity_id").with_row_index("_q")
        ti = pl.DataFrame({"target_id": self.ids}).with_row_index("_t")
        work = pairs.select("s1_id", "target_id").with_row_index("_row").join(qi, on="s1_id").join(ti, on="target_id")
        if work.height != pairs.height:
            raise ValueError("Pair contains IDs outside this query/target partition")
        scores = np.zeros(pairs.height, np.float32)
        if self.vocabulary:
            qm = self.transform(queries[CHANNELS[self.channel]["field"]].to_list())
            for part in self.meta["parts"]:
                block = work.filter((pl.col("_t") >= part["start"]) & (pl.col("_t") < part["start"] + part["rows"]))
                if not block.height:
                    continue
                x = sparse.load_npz(self.path / f"{part['start']:09d}.tfidf.npz")
                for b in block.iter_slices(50_000):
                    values = qm[b["_q"].to_numpy()].multiply(x[b["_t"].to_numpy() - part["start"]])
                    scores[b["_row"].to_numpy()] = np.asarray(values.sum(axis=1)).ravel()
                del x
        return pairs.select("s1_id", "target_id").with_columns(score=pl.Series(scores))
