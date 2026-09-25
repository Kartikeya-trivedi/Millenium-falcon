import math
import numpy as np
import polars as pl
import pytest

from plan1.normalize import normalize_records
from plan1.translit import make_converter
from plan3.features import FEATURES, rich_features, weighted_overlap
from plan3.views import build_views, number_view
from plan3.retrieve import union_channels


@pytest.mark.parametrize("address,house,unit", [
    ("12B Main Road Apt 5","12b","5"),
    ("12/3 Main Rd Unit A-5","12/3","a-5"),
    ("N°49 Rue des Lilas","49",""),
    ("47 bis Rue Alpha","47bis",""),
    ("Sector 12 Main Road","",""),
    ("Floor 2, 12 Main Road","",""),
])
def test_number_roles_preserve_conflicts(address,house,unit):
    actual = number_view(address)
    assert actual[:2] == (house,unit)


def test_contained_address_missing_distinctive_words():
    values = weighted_overlap({"12","main","rd","springfield"},{"main","rd"},
                              {"12":8,"main":2,"rd":1,"springfield":6},10)
    assert values[1] == pytest.approx(3/17)
    assert values[2] == 1
    assert values[3] == 14
    assert all(math.isnan(v) for v in weighted_overlap(set(),{"a"},{},10))


def test_rich_features_keep_original_features_and_actual_coverage():
    raw = pl.DataFrame([
        ("S1-1","Lakshmi Pvt Ltd DBA Alpha Tools","12B Main Rd Apt 5","India","S1"),
        ("S2-2","लक्ष्मी अपरिचित","12/3 Main Rd Apt 6","India","S2"),
        ("S3-3","www.alphatools.co.in","","India","S3"),
    ],schema=["entity_id","business_name","business_address","country","source"],orient="row")
    mapping={"लक्ष्मी":"lakshmi"}
    norm=normalize_records(raw,make_converter(mapping)).join(build_views(raw,mapping),on="entity_id")
    norm=norm.with_columns(reference_name_count=pl.lit(1))
    p=pl.DataFrame({"s1_id":["S1-1","S1-1"],"target_id":["S2-2","S3-3"],
                    "score":pl.Series([.8,.7],dtype=pl.Float32),"rank":pl.Series([1,1],dtype=pl.Int32)})
    c=union_channels({"word":p}).join(p.select("s1_id","target_id","score","rank"),on=["s1_id","target_id"])
    c=c.with_columns(source=pl.col("target_id").str.slice(0,2))
    f=rich_features(c,norm.filter(pl.col("source")=="S1"),norm.filter(pl.col("source")!="S1"),
                    {"S2":({},100),"S3":({},100)})
    assert set(FEATURES) <= set(f.columns)
    row=f.filter(pl.col("target_id")=="S2-2").row(0,named=True)
    assert row["target_indic_unknown_share"] == .5
    assert row["house_conflict"] == 1 and row["unit_conflict"] == 1
    missing=f.filter(pl.col("target_id")=="S3-3").row(0,named=True)
    assert missing["address_shared_idf"] is None
    assert missing["house_agreement"] is None
    assert missing["domain_best_ratio"] is not None
