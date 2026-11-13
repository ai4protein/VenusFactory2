"""Check explicit reported quantities against their cited measurements.

This deterministic check supplements independent scientific review. Matching a
quantity does not establish its protein, substrate, assay conditions or causality.
"""
from __future__ import annotations

from decimal import Decimal, DecimalException, localcontext
import json
import re

# Canonical units: mol/L, Da, angstrom, Celsius, percent, residues and g/L.
_UNITS = {
    "M": ("molar", "1"), "mM": ("molar", "1e-3"),
    "uM": ("molar", "1e-6"), "µM": ("molar", "1e-6"),
    "μM": ("molar", "1e-6"), "nM": ("molar", "1e-9"),
    "pM": ("molar", "1e-12"),
    "Da": ("mass", "1"), "kDa": ("mass", "1e3"), "MDa": ("mass", "1e6"),
    "Å": ("length", "1"), "nm": ("length", "10"),
    "°C": ("temperature", "1"), "C": ("temperature", "1"),
    "%": ("percent", "1"), "aa": ("residues", "1"),
    "pH": ("ph", "1"),
    "mg/mL": ("mass_concentration", "1"), "g/L": ("mass_concentration", "1"),
    "ug/mL": ("mass_concentration", "1e-3"),
    "µg/mL": ("mass_concentration", "1e-3"),
    "μg/mL": ("mass_concentration", "1e-3"),
}
_WORD_UNITS = {
    "angstrom": "Å", "angstroms": "Å", "degreecelsius": "°C",
    "degreescelsius": "°C", "aminoacid": "aa", "aminoacids": "aa",
    "residue": "aa", "residues": "aa",
}
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNIT = "(?:" + "|".join(re.escape(k) for k in sorted(_UNITS, key=len, reverse=True)) + (
    r"|°\s+C|(?i:angstroms?|degrees?\s+Celsius|amino[- ]acids?|residues?))"
)
_BOUNDARY = r"(?<![\w.,^×−-])"
_END = r"(?![\w/])"
_SEPARATOR = r"(?:[-–—]|\bto\b)"
_RANGE = re.compile(
    _BOUNDARY + r"(?P<low>" + _NUMBER + r")\s*" + _SEPARATOR
    + r"\s*(?P<high>" + _NUMBER + r")\s*(?P<unit>" + _UNIT + ")" + _END
)
_SCALAR = re.compile(
    _BOUNDARY + r"(?P<low>" + _NUMBER + r")\s*(?P<unit>" + _UNIT + ")" + _END
)
_PH = re.compile(
    r"\bpH(?:\s+(?:value|optimum))?"
    r"(?:\s+(?:of|for)\s+[^.;\n]{1,160}?\s+(?:is|was|=)\s*"
    r"|\s*(?:(?:is|was|of|=|:)\s*)?)(?P<low>" + _NUMBER + r")"
    + r"(?:\s*" + _SEPARATOR + r"\s*(?P<high>" + _NUMBER + r"))?", re.I
)
# ASCII A means angstrom only in an explicit resolution expression. Elsewhere
# it can identify a chain, alanine or electric current and must not become length.
_RESOLUTION_VALUE = (_BOUNDARY + r"(?P<low>" + _NUMBER + r")"
                     + r"(?:\s*" + _SEPARATOR + r"\s*(?P<high>" + _NUMBER + r"))?\s*A" + _END)
_ASCII_RESOLUTION = (
    re.compile(r"\b(?i:resolution)(?:\s+(?i:is|was|of|at|to))?\s*[:=]?\s*" + _RESOLUTION_VALUE),
    re.compile(_RESOLUTION_VALUE + r"\s+(?i:(?:spatial\s+)?resolution)\b"),
)
_JSON = json.JSONDecoder(parse_float=Decimal)


def _quantity(low, high, unit):
    if not isinstance(unit, str):
        return None
    compact = re.sub(r"\s+", "", unit)
    canonical = _WORD_UNITS.get(compact.casefold().replace("-", ""), compact)
    if canonical not in _UNITS or type(low) is bool or type(high) is bool:
        return None
    try:
        raw_first, raw_last = str(low), str(high)
        if any("," in value and not re.fullmatch(_NUMBER, value)
               for value in (raw_first, raw_last)):
            return None
        first = Decimal(raw_first.replace(",", ""))
        last = Decimal(raw_last.replace(",", ""))
        if (not first.is_finite() or not last.is_finite() or first > last
                or abs(first.adjusted()) > 1000 or abs(last.adjusted()) > 1000):
            return None
        dimension, factor = _UNITS[canonical]
        with localcontext() as context:
            context.prec = max(28, len(first.as_tuple().digits) + 12,
                               len(last.as_tuple().digits) + 12)
            return dimension, first * Decimal(factor), last * Decimal(factor)
    except (DecimalException, ValueError, TypeError):
        return None


def measurements(text, *, strict=False):
    """Find explicit scalar and interval quantities without changing the text."""
    found, intervals = set(), []

    def add(low, high, unit):
        item = _quantity(low, high, unit)
        if item is None:
            if strict:
                raise ValueError("Invalid explicit measured quantity")
        else:
            found.add(item)

    for match in _RANGE.finditer(text):
        # Attached M is also a model-size suffix; an unprefixed molar quantity
        # requires a separating space or an explicit structured unit field.
        if match["unit"] == "M" and match.start("unit") == match.end("high"):
            continue
        add(match["low"], match["high"], match["unit"])
        intervals.append(match.span())
    for match in _SCALAR.finditer(text):
        if any(start <= match.start() < end for start, end in intervals):
            continue
        if match["unit"] == "M" and match.start("unit") == match.end("low"):
            continue
        add(match["low"], match["low"], match["unit"])
    for match in _PH.finditer(text):
        add(match["low"], match["high"] or match["low"], "pH")
    for pattern in _ASCII_RESOLUTION:
        for match in pattern.finditer(text):
            add(match["low"], match["high"] or match["low"], "\u00c5")
    return found


# These exact keys are emitted by the platform ProtParam calculation. A key's
# suffix is not a general unit inference rule for arbitrary external JSON.
_PLATFORM_QUANTITIES = {"molecular_weight_Da": "Da", "length_aa": "aa"}


def _platform_quantity(key, value):
    if type(value) not in (int, float, Decimal):
        return None
    item = _quantity(value, value, _PLATFORM_QUANTITIES[key])
    if item is None or item[1] <= 0:
        return None
    if key == "length_aa" and item[1] != item[1].to_integral_value():
        return None
    return item


def _structured_measurements(value):
    found = set()
    if isinstance(value, str):
        return measurements(value)
    if isinstance(value, list):
        for child in value:
            found.update(_structured_measurements(child))
    elif isinstance(value, dict):
        unit = value.get("unit")
        if isinstance(unit, str):
            for key in ("value", "constant"):
                if key in value:
                    item = _quantity(value[key], value[key], unit)
                    if item is not None:
                        found.add(item)
        for key, child in value.items():
            if key in _PLATFORM_QUANTITIES:
                item = _platform_quantity(key, child)
                if item is not None:
                    found.add(item)
                continue
            if key in {"sequence_length", "residue_count", "amino_acid_count"}:
                item = _quantity(child, child, "aa")
                if item is not None:
                    found.add(item)
            found.update(_structured_measurements(child))
    return found


def _quoted_measurements(quote):
    found = set()
    # A short exact quote may omit its outer braces but contain complete records.
    # Decode each complete object once; numbers and units must share that object.
    consumed = 0
    for start in re.finditer(r"[\[{]", quote):
        if start.start() < consumed:
            continue
        try:
            value, end = _JSON.raw_decode(quote, start.start())
        except (ValueError, RecursionError):
            continue
        found.update(measurements(quote[consumed:start.start()]))
        consumed = end
        try:
            found.update(_structured_measurements(value))
        except RecursionError:
            continue
    found.update(measurements(quote[consumed:]))
    return found


def measured_values_anchored(text, evidence):
    """Require the same dimension and value, allowing exact unit conversions."""
    try:
        claimed = measurements(text, strict=True)
    except ValueError:
        return False
    quoted = set()
    for item in evidence:
        quoted.update(_quoted_measurements(item["quote"]))
    return claimed <= quoted
