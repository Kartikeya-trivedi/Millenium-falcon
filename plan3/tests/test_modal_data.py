from zipfile import ZipFile
from plan3.modal_client import pack_dataset


def test_cloud_archive_has_exact_inputs_and_content_identity(tmp_path):
    data = tmp_path/"data"
    names = [f"train/train_source{s}.tsv" for s in (1, 2, 3)] + ["train/train_ground_truth.tsv"]
    names += [f"test/test_source{s}.tsv" for s in (1, 2, 3)]
    for name in names:
        path = data/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    identity, archive, spec = pack_dataset(data, tmp_path/"out")
    with ZipFile(archive) as z:
        assert set(z.namelist()) == set(names)
    assert len(spec["files"]) == 7
    assert pack_dataset(data, tmp_path/"out")[0] == identity
    (data/names[0]).write_text("changed\n")
    assert pack_dataset(data, tmp_path/"out")[0] != identity
