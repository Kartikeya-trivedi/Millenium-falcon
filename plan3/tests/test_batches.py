import polars as pl
import pytest
from plan3.featurize import query_batches


def test_indexed_batches_keep_every_owner_candidate_and_source_together():
    c=pl.DataFrame({"s1_id":["b","a","a","b","c"],"target_id":["v","u","w","u","x"],
                    "source":["S2","S2","S3","S2","S3"]})
    q=pl.DataFrame({"entity_id":["c","a","b"],"name":["C","A","B"]})
    t=pl.DataFrame({"entity_id":["x","u","v","w"],"name":["X","U","V","W"]})
    result=list(query_batches(c,q,t,size=2))
    assert len(result)==2
    assert result[0][2]["entity_id"].to_list()==["a","b"]
    assert result[0][1].height==4
    assert set(result[0][3]["entity_id"])=={"u","v","w"}
    assert pl.concat([r[1] for r in result]).equals(c.sort("s1_id","target_id"))
    with pytest.raises(ValueError,match="Query records"):
        list(query_batches(c,q.head(2),t))
