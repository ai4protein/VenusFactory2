import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from fetch_pmc_assets import resolve_asset, verified_bytes


def test_pmc_url_and_checksum_binding():
    content = b"public data"
    checksum = hashlib.md5(content).hexdigest()
    url = f"s3://pmc-oa-opendata/PMC123.1/data.zip?md5={checksum}"
    assert resolve_asset(url, "PMC123.1") == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/data.zip",
        "data.zip",
        checksum,
    )
    assert verified_bytes(content, checksum)["sha256"] == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        verified_bytes(b"corrupt", checksum)


@pytest.mark.parametrize(
    "path",
    [
        "s3://other/PMC123.1/x",
        "s3://pmc-oa-opendata/PMC456.1/x",
        "s3://pmc-oa-opendata/PMC123.1/../x",
        "s3://pmc-oa-opendata/PMC123.1/%2e%2e%2Fx",
        "s3://pmc-oa-opendata/PMC123.1/x?md5=bad",
    ],
)
def test_unpinned_or_cross_article_assets_are_rejected(path):
    with pytest.raises(ValueError):
        resolve_asset(path, "PMC123.1")
