"""Render downloaded research prose without active links or embedded resources."""
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

_TAGS = frozenset("p a em strong code pre blockquote ul ol li br hr table thead tbody tr th td del s sup sub h1 h2 h3 h4 h5 h6 dl dt dd".split())
_VOID = frozenset({"br", "hr"})


def _link(value):
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    if value.startswith("#"):
        return True
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"https", "http"} and bool(parsed.hostname)
    except ValueError:
        return False


class _ReportHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag not in _TAGS:
            if tag == "img":
                self.parts.append(escape(dict(attrs).get("alt") or ""))
            return
        attributes = []
        if tag == "a":
            for key, value in attrs:
                if key == "href" and _link(value):
                    attributes.append('href="' + escape(value, quote=True) + '"')
                elif key == "title" and value is not None:
                    attributes.append('title="' + escape(value, quote=True) + '"')
            attributes.append('rel="noopener noreferrer"')
        self.parts.append("<" + tag + (" " + " ".join(attributes) if attributes else "") + ">")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in _TAGS and tag not in _VOID:
            self.parts.append("</" + tag + ">")

    def handle_data(self, data):
        self.parts.append(escape(data))


def render_report_markdown(text):
    import markdown
    rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
    parser = _ReportHTML()
    parser.feed(rendered)
    parser.close()
    return "".join(parser.parts)
