"""Richer direct evidence, preserving all 28 imported pair features."""
from __future__ import annotations

import math
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from plan1.features import FEATURES as BASE_FEATURES, build_features as baseline_features, text_lookup

WEIGHTED = [f"{field}_{suffix}" for field in ("name","address") for suffix in
            ("shared_idf","coverage_query","coverage_target","unshared_query","unshared_target","weighted_jaccard")]
NUMERIC = [f"{kind}_{suffix}" for kind in ("house","unit","floor") for suffix in ("agreement","conflict")] + [
    "query_house_parsed", "target_house_parsed"]
RELIABILITY = ["query_indic_unknown_share","target_indic_unknown_share","target_indic_token_count",
               "alias_best_ratio","domain_best_ratio","query_name_log_frequency","target_name_log_frequency"]
CHANNEL = ["word_top50", "word_retrieved", "missing_score", "missing_rank", "missing_retrieved",
           "char_score", "char_rank", "char_retrieved", "name_score", "name_rank", "name_retrieved", "channel_count"]
EXTRA_FEATURES = WEIGHTED + NUMERIC + RELIABILITY + CHANNEL
FEATURES = BASE_FEATURES + EXTRA_FEATURES


def weighted_overlap(a: set, b: set, weights: dict, unseen: float) -> tuple:
    if not a or not b:
        return (np.nan,)*6
    shared = sum(weights.get(t,unseen) for t in a & b)
    wa = sum(weights.get(t,unseen) for t in a)
    wb = sum(weights.get(t,unseen) for t in b)
    return shared, shared/wa, shared/wb, wa-shared, wb-shared, shared/(wa+wb-shared)


def match_number(a: str, b: str) -> tuple:
    if not a or not b:
        return np.nan, np.nan
    return float(a==b), float(a!=b)


def rich_features(cands: pl.DataFrame, q_records: pl.DataFrame, t_records: pl.DataFrame,
                  idfs: dict[str,tuple[dict,int]]) -> pl.DataFrame:
    """Every candidate for each S1 must be in this call. idfs is keyed by S2/S3."""
    normal = pl.concat([q_records.select(c for c in q_records.columns if c in t_records.columns),
                        t_records.select(c for c in q_records.columns if c in t_records.columns)])
    q, t = text_lookup(normal,cands["s1_id"],cands["target_id"])
    base = baseline_features(cands,q,t)
    queries = {r["entity_id"]:r for r in q_records.iter_rows(named=True)}
    targets = {r["entity_id"]:r for r in t_records.iter_rows(named=True)}
    for record in list(queries.values()) + list(targets.values()):
        record["_name_set"] = set(record["name_red"].split())
        record["_addr_set"] = set(record["addr_norm"].split())
    rows = []
    for cand in cands.iter_rows(named=True):
        a, b = queries[cand["s1_id"]], targets[cand["target_id"]]
        weights, n = idfs[cand["source"]]
        unseen = math.log(n+1)+1
        values = list(weighted_overlap(a["_name_set"],b["_name_set"],weights,unseen))
        values += weighted_overlap(a["_addr_set"],b["_addr_set"],weights,unseen)
        for kind in ("house","unit","floor"):
            values += match_number(a[kind],b[kind])
        values += [a["house_parsed"],b["house_parsed"]]
        aliases_a, aliases_b = a["aliases"], b["aliases"]
        alias = (max(fuzz.ratio(x,y)/100 for x in (aliases_a or [a["name_red"]])
                     for y in (aliases_b or [b["name_red"]])) if (aliases_a or aliases_b) and a["name_red"] and b["name_red"] else np.nan)
        domain = (fuzz.ratio(a["domain"] or a["name_red"],b["domain"] or b["name_red"])/100
                  if (a["domain"] or b["domain"]) and a["name_red"] and b["name_red"] else np.nan)
        values += [a["indic_unknown_share"],b["indic_unknown_share"],b["indic_tokens"],alias,domain,
                   math.log1p(a.get("reference_name_count") or 0),math.log1p(b.get("reference_name_count") or 0)]
        values += [float(cand["word_rank"] is not None and cand["word_rank"]<=50),
                   float(cand["word_rank"] is not None)]
        for ch in ("missing","char","name"):
            values += [cand[ch+"_score"],cand[ch+"_rank"],float(cand[ch+"_rank"] is not None)]
        values += [sum(cand[ch+"_rank"] is not None for ch in ("word","missing","char","name"))]
        rows.append(values)
    extras = pl.DataFrame(rows,schema={c:pl.Float32 for c in EXTRA_FEATURES},orient="row").with_columns(pl.all().fill_nan(None))
    if base.height != extras.height or not base.select("s1_id","target_id").equals(cands.select("s1_id","target_id")):
        raise ValueError("Feature row alignment changed")
    return base.hstack(extras)
