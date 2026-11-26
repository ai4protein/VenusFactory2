from html.parser import HTMLParser
import pytest
from backend.web_v2.report_rendering import render_report_markdown


class Nodes(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.nodes = []
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


@pytest.mark.parametrize("url", [
    "javascript:alert%281%29", "JaVaScRiPt:alert%281%29",
    "jav&#x61;script:alert%281%29", "data:text/html,bad",
    "file:///private/data", "//example.org/tracker",
])
def test_download_blocks_active_and_local_links(url):
    rendered = render_report_markdown("[unsafe](" + url + ")")
    assert all("href" not in attrs for tag, attrs in Nodes(rendered).nodes if tag == "a")


def test_download_preserves_prose_tables_citations_and_code():
    text = "# Report\n\n[Source](https://example.org/study) [S1](#ref-S1)\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('<example>')\n```"
    rendered = render_report_markdown(text)
    nodes = Nodes(rendered).nodes
    assert {"h1","a","table","pre","code"} <= {tag for tag, _ in nodes}
    assert {attrs["href"] for tag,attrs in nodes if tag == "a"} == {"https://example.org/study","#ref-S1"}
    assert "print" in rendered and "&lt;example&gt;" in rendered


def test_download_does_not_embed_images_or_raw_html():
    text = '![private-image](https://example.org/track.png)\n<script>alert(1)</script><img src=x onerror=alert(1)>'
    rendered = render_report_markdown(text)
    nodes = Nodes(rendered).nodes
    assert not {"img","script","iframe","style","object"} & {tag for tag,_ in nodes}
    assert "private-image" in rendered
    assert all(not any(key.startswith("on") for key in attrs) for _,attrs in nodes)
