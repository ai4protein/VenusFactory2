import math
import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.audit_turbopetase_publication import extract_triplicates, parse_rankings


def test_triplicates_preserve_conditions_and_do_not_silently_impute_missing_data():
    sheet = openpyxl.Workbook().active
    sheet.cell(5, 1, "condition")
    for column, value in enumerate([1.0, 2.0, 3.0, 2.0, math.sqrt(2 / 3)], 2):
        sheet.cell(5, column, value)
    kwargs = {"metadata": {"condition": 1}, "replicates": (2, 3, 4), "mean_column": 5, "sd_column": 6}
    row = extract_triplicates(sheet, **kwargs)[0]
    assert row["mean_abs_error"] == row["population_sd_abs_error"] == 0
    assert row["sample_sd"] == 1
    assert row["replicates"] == [1, 2, 3]
    sheet.cell(5, 3).value = None
    with pytest.raises(ValueError, match="imputation"):
        extract_triplicates(sheet, **kwargs)


def test_publication_ranking_requires_ten_rows_in_order():
    table = "\n".join(f"{i} A{i} 1.0 C{i} 2.0 D{i} 3.0 E{i} 4.0" for i in range(1, 11))
    parsed = parse_rankings(table)
    assert len(parsed) == 4
    assert parsed["BhrPETase_Uniclust30_BFD"][9]["residue"] == "A10"
    with pytest.raises(ValueError, match="ten published ranks"):
        parse_rankings(table.replace("10 A10", "9 A10"))
