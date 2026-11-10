import csv
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from mcp_tools.models.factory.training.data.dataloader import _load_dataset_dict
from backend.web_v2 import custom_model_api as api


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['aa_seq', 'label']); writer.writerows(rows)
    return str(path)


def test_uploaded_csv_uses_identical_preview_and_training_partitions(tmp_path, monkeypatch):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    rows = [('M' + alphabet[i // 20] + alphabet[i % 20] + 'ACDEFG', i % 2) for i in range(40)]
    path = table(tmp_path / 'train.csv', rows)
    monkeypatch.setattr(api, '_resolve_client_file_path', lambda value: __import__('pathlib').Path(value))
    args = dict(train_file=path, valid_file='', test_file='', sequence_column_name='aa_seq', label_column_name='label', problem_type='single_label_classification')
    stats = api._uploaded_training_stats(args)
    data = _load_dataset_dict(SimpleNamespace(**args))
    assert stats == {key: len(value) for key, value in data.items()} == {'train': 32, 'validation': 4, 'test': 4}
    assert set(data['train']['aa_seq']).isdisjoint(data['test']['aa_seq'])
    assert set(data['train']['aa_seq']).isdisjoint(data['validation']['aa_seq'])
    assert set(data['test']['aa_seq']).isdisjoint(data['validation']['aa_seq'])
    # Explicit validation remains unchanged; only the absent test split is derived.
    args['valid_file'] = table(tmp_path / 'validation.csv', [('WWWWA', 0), ('WWWWC', 1)])
    data = _load_dataset_dict(SimpleNamespace(**args))
    assert data['validation']['aa_seq'] == ['WWWWA', 'WWWWC']
    assert api._uploaded_training_stats(args) == {'train': 36, 'validation': 2, 'test': 4}


@pytest.mark.parametrize('issue', ['small', 'empty_label', 'overlap', 'column'])
def test_invalid_uploaded_tables_fail_before_enqueue(tmp_path, monkeypatch, issue):
    rows = [('ACDE' + str(i), i % 2) for i in range(40)]
    if issue == 'small': rows = rows[:2]
    if issue == 'empty_label': rows[0] = (rows[0][0], '')
    path = table(tmp_path / 'train.csv', rows)
    monkeypatch.setattr(api, '_resolve_client_file_path', lambda value: __import__('pathlib').Path(value))
    args = dict(train_file=path, sequence_column_name='aa_seq', label_column_name='label', problem_type='single_label_classification')
    if issue == 'overlap':
        args['valid_file'] = table(tmp_path / 'val.csv', rows)
        args['test_file'] = table(tmp_path / 'test.csv', rows)
    if issue == 'column': args['sequence_column_name'] = 'missing'
    with pytest.raises(HTTPException) as error: api._uploaded_training_stats(args)
    assert error.value.status_code == 400


@pytest.mark.parametrize("extension", ["csv", "tsv", "xlsx"])
def test_uploaded_table_headers_and_preview_use_the_same_columns(tmp_path, monkeypatch, extension):
    import pandas as pd
    source = pd.DataFrame({"sequence": ["ACDE", "MKWV", "GAVL", "SSSS"],
                           "activity": [1.5, 2.0, 3.0, 4.5]})
    path = tmp_path / ("measurements." + extension)
    if extension == "xlsx":
        source.to_excel(path, index=False)
    else:
        source.to_csv(path, index=False, sep="\t" if extension == "tsv" else ",")
    checked = []

    def resolve(value):
        checked.append(value)
        return path

    monkeypatch.setattr(api, "_resolve_client_file_path", resolve)
    assert api._read_tabular_columns(str(path)) == source.columns.tolist()
    pd.testing.assert_frame_equal(api._read_tabular_df(str(path)), source.head(3))
    pd.testing.assert_frame_equal(api._read_tabular_df(str(path), sample_n=None), source)
    assert checked == [str(path)] * 3


@pytest.mark.parametrize("reader", ["_read_tabular_columns", "_read_tabular_df"])
def test_uploaded_table_readers_enforce_access_before_reading(tmp_path, monkeypatch, reader):
    path = tmp_path / "private.csv"
    path.write_text("sequence,label\nACDE,1\n")

    def denied(value):
        raise HTTPException(404, "File not found")

    monkeypatch.setattr(api, "_resolve_client_file_path", denied)
    with pytest.raises(HTTPException) as error:
        getattr(api, reader)(str(path))
    assert error.value.status_code == 404
