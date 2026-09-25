"""Additional raw-text evidence. These views never replace the v2 normalized text."""
from __future__ import annotations

import gc
import re
import unicodedata
from pathlib import Path

import polars as pl

from plan1.normalize import LEGAL_FORMS, FUNCTION_WORDS, basic_clean, word_map, DOTTED_FORMS
from plan1.translit import make_converter
from .artifacts import contract, load_prepared, parquet, read_json, records, sha256, write_json

INDIC = re.compile("[ऀ-෿]")
ALIAS = re.compile(r"\b(?:d\s*b\s*a|dba|aka|fka|t\s+a|doing business as|formerly)\b", re.I)
DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9.-]*\.[a-z]{2,12})(?:[/\s]|$)", re.I)
HOUSE = re.compile(r"^\s*(?:(?:plot|house|door|h)\s*[.\s]*(?:no\.?\s*)?|no\.?\s*|n[°º]\s*|#\s*)?"
                   r"([a-z]?-?\d+[a-z]?(?:[/-]\d+[a-z]?)*)(?:\s+(bis|ter)\b)?(?=\s|[,.;:]|$)", re.I)
UNIT = re.compile(r"\b(?:suite|ste|unit|apt|apartment|flat|shop)\s*(?:no[.\s]*)?([a-z0-9][a-z0-9/-]*)", re.I)
FLOOR = re.compile(r"\b(\d+)(?:st|nd|rd|th)?\s*(?:floor|fl)\b|\b(?:floor|fl)\s*(\d+)", re.I)
DROP = re.compile(r"\b(?:" + "|".join(LEGAL_FORMS + FUNCTION_WORDS) + r")\b")


def number_view(address: str) -> tuple[str, str, str, float]:
    text = unicodedata.normalize("NFKC", address or "").lower()
    house, unit, floor = HOUSE.search(text), UNIT.search(text), FLOOR.search(text)
    hv = ((house[1] + (house[2] or "")).lower() if house else "")
    uv = unit[1].lower() if unit and any(c.isdigit() for c in unit[1]) else ""
    fv = next((g for g in floor.groups() if g), "") if floor else ""
    return hv, uv, fv, 1.0 if house else 0.0


def build_views(raw: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    clean = raw.select(entity_id="entity_id", clean=basic_clean(pl.col("business_name")))
    converter = make_converter(mapping)
    rows = []
    for record, cleaned in zip(raw.iter_rows(named=True), clean["clean"]):
        indic = [t for t in cleaned.split() if INDIC.search(t)]
        known = sum(t in mapping for t in indic)
        aliases = ALIAS.split(cleaned)
        aliases = [" ".join(DROP.sub(" ", converter(v)).split()) for v in aliases] if len(aliases)>1 else []
        aliases = [v for v in aliases if v]
        domain = DOMAIN.search((record["business_name"] or "").lower())
        stem = ""
        if domain:
            stem = re.sub(r"^www\.", "", domain[1])
            stem = re.sub(r"\.(?:co\.in|co\.uk|com\.au|com|net|org|in|fr|us|[a-z]{2,12})$", "", stem)
            stem = " ".join(re.sub(r"[^a-z0-9]", " ", stem).split())
        house, unit, floor, confidence = number_view(record["business_address"])
        rows.append({
            "entity_id":record["entity_id"], "aliases":aliases, "domain":stem,
            "house":house, "unit":unit, "floor":floor, "house_parsed":confidence,
            "indic_tokens":len(indic), "indic_unknown_share":1-known/len(indic) if indic else None,
        })
    return pl.DataFrame(rows, schema={"entity_id":pl.String,"aliases":pl.List(pl.String),"domain":pl.String,
                                      "house":pl.String,"unit":pl.String,"floor":pl.String,"house_parsed":pl.Float32,
                                      "indic_tokens":pl.Float32,"indic_unknown_share":pl.Float32})


def prepare_views(prepared: Path, out: Path, split="train"):
    meta = load_prepared(prepared)
    mapping = read_json(prepared/"translit.json")["mapping"]
    contract(out / f"{split}_contract.json", {
        "prepared_identity":meta["identity"], "dictionary_sha":meta["dictionary_sha256"],
        "views_code":sha256(Path(__file__)), "split":split})
    dataset = Path(meta["dataset"])
    for source in ("S1","S2","S3"):
        parts = [p for p in meta["partitions"] if p["split"]==split and p["source"]==source]
        if all((out / p["file"]).exists() for p in parts):
            continue
        print(f"Additional text views: {split}/{source}", flush=True)
        raw = records(dataset/split/f"{split}_source{source[1]}.tsv").collect(engine="streaming")
        for part in parts:
            destination = out / part["file"]
            if destination.exists():
                continue
            # Small record batches bound Python objects; output is one reusable partition.
            blocks = [build_views(block, mapping) for block in raw.filter(pl.col("country")==part["country"]).iter_slices(50_000)]
            parquet(destination, pl.concat(blocks))
            print(f"  {part['country']}/{source}: {part['rows']:,}", flush=True)
            del blocks
        del raw
        gc.collect()
    # S1 name ambiguity is corpus-only evidence, available for unseen countries too.
    freq = []
    for part in meta["partitions"]:
        if part["split"]==split and part["source"]=="S1":
            freq.append(pl.read_parquet(prepared/part["file"],columns=["name_red","country"])
                        .group_by("name_red","country").len(name="reference_name_count"))
    parquet(out/f"{split}_name_frequency.parquet",pl.concat(freq))
    write_json(out/f"{split}_complete.json",{"status":"prepared","split":split})
