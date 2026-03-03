"""
BRENDA SOAP API client for enzyme database access.

Provides get_km_values, get_reactions, and call_brenda for low-level SOAP calls.
Authentication via BRENDA_EMAIL and BRENDA_PASSWORD in .env (or env); legacy BRENDA_EMIAL supported.
Requires: zeep, requests. See references/api_reference.md for full API details.

If you get empty results: (1) Confirm BRENDA account is registered at brenda-enzymes.org and
credentials in .env are correct. (2) Set BRENDA_DEBUG=1 and run again; inspect
example/database/brenda/debug_getKmValue_raw.txt (and getReaction) for the raw SOAP response.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # noqa: D103
        pass

# Load .env so BRENDA_EMAIL / BRENDA_PASSWORD are available to all brenda modules
load_dotenv()

try:
    from zeep import Client, Settings
    from zeep.exceptions import Fault, TransportError
    ZEEP_AVAILABLE = True
except ImportError:
    ZEEP_AVAILABLE = False

WSDL_URL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
_client: Optional["Client"] = None


def load_env_from_file(path: str = ".env") -> None:
    """Load environment variables from .env file (uses python-dotenv). Kept for API compatibility."""
    load_dotenv(path)


def _get_credentials() -> tuple:
    """Return (email, password) from environment. Raises RuntimeError if missing."""
    load_dotenv()
    email = os.environ.get("BRENDA_EMAIL") or os.environ.get("BRENDA_EMIAL")
    password = os.environ.get("BRENDA_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "BRENDA credentials required. Set BRENDA_EMAIL and BRENDA_PASSWORD "
            "in .env or environment (see .env.example)."
        )
    return email, password


def _hash_password(password: str) -> str:
    """SHA-256 hex digest of password for BRENDA API."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _brenda_param(key: str, value: str) -> str:
    """Format a BRENDA SOAP parameter as key*value. Use key* for wildcard (*)."""
    if value and value.strip() and value != "*":
        return f"{key}*{value.strip()}"
    return f"{key}*"


def _debug_write(action: str, raw) -> None:
    """Write raw SOAP response to file when BRENDA_DEBUG=1 for debugging empty results."""
    try:
        out = os.path.join("example", "database", "brenda", f"debug_{action}_raw.txt")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"type: {type(raw).__name__}\n")
            f.write(f"repr: {repr(raw)[:2000]}\n")
            if isinstance(raw, list):
                f.write(f"len: {len(raw)}\n")
                for i, item in enumerate(raw[:5]):
                    f.write(f"  [{i}] type={type(item).__name__} repr={repr(item)[:200]}\n")
    except Exception:
        pass


def _get_client() -> "Client":
    """Return singleton Zeep SOAP client for BRENDA."""
    global _client
    if not ZEEP_AVAILABLE:
        raise ImportError("zeep is required. Install with: uv pip install zeep")
    if _client is None:
        settings = Settings(strict=False, xml_huge_tree=True)
        _client = Client(wsdl=WSDL_URL, settings=settings)
    return _client


def _item_to_str(item) -> str:
    """Convert a single response item (string or SOAP object) to string."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    # Zeep SOAP often returns objects with 'value' or '_value_1' for string content
    s = getattr(item, "value", None) or getattr(item, "_value_1", None)
    if s is not None:
        return (s if isinstance(s, str) else str(s)).strip()
    return str(item).strip()


def split_entries(return_text) -> List[str]:
    """Normalize BRENDA response to list of entry strings. Handles zeep ArrayOfstring, single string, and #-separated entries."""
    if return_text is None:
        return []
    # Zeep returns ArrayOfstring as list (or iterable of string/object)
    if isinstance(return_text, list):
        out = []
        for item in return_text:
            s = _item_to_str(item)
            if s:
                out.append(s)
        if out:
            return out
        # If list items are nested, recurse
        for item in return_text:
            out.extend(split_entries(item))
        return out
    if hasattr(return_text, "_value_1"):
        return split_entries(getattr(return_text, "_value_1", ""))
    try:
        if not isinstance(return_text, str) and hasattr(return_text, "__iter__") and not isinstance(return_text, (str, bytes)):
            out = []
            for item in return_text:
                s = _item_to_str(item)
                if s:
                    out.append(s)
            if out:
                return out
            for item in return_text:
                out.extend(split_entries(item))
            return out
    except Exception:
        pass
    if not isinstance(return_text, str):
        s = str(return_text).strip()
        return [s] if s else []
    text = return_text.strip()
    if not text:
        return []
    lines = [s.strip() for s in text.split("\n") if s.strip()]
    if len(lines) > 1:
        return lines
    return [text]


def call_brenda(action: str, parameters: List[str], use_kwargs: bool = False, param_names: Optional[List[str]] = None):
    """
    Execute a BRENDA SOAP action. Supports multi-arg, kwargs, or single concatenated string (BRENDA official format).
    """
    client = _get_client()
    if not hasattr(client.service, action):
        raise ValueError(f"Unknown BRENDA action: {action}")
    method = getattr(client.service, action)
    if use_kwargs and param_names and len(param_names) == len(parameters):
        kwargs = dict(zip(param_names, parameters))
        return method(**kwargs)
    return method(*parameters)


def get_km_values(
    ec_number: str,
    organism: str = "*",
    substrate: str = "*",
    km_value: str = "*",
    km_value_maximum: str = "*",
    commentary: str = "*",
    ligand_structure_id: str = "*",
    literature: str = "*",
) -> List[str]:
    """
    Retrieve Km values for the given EC number.
    Returns list of BRENDA entry strings (organism*...#substrate*...).
    BRENDA Zeep API expects each filter as "key*value" (e.g. ecNumber*1.1.1.1, organism*).
    """
    email, password = _get_credentials()
    pwd_hash = _hash_password(password)
    # Official format: (email, password, "ecNumber*...", "organism*...", "kmValue*", ...)
    params = [
        email,
        pwd_hash,
        _brenda_param("ecNumber", ec_number),
        _brenda_param("organism", organism),
        _brenda_param("kmValue", km_value),
        _brenda_param("kmValueMaximum", km_value_maximum),
        _brenda_param("substrate", substrate),
        _brenda_param("commentary", commentary),
        _brenda_param("ligandStructureId", ligand_structure_id),
        _brenda_param("literature", literature),
    ]
    raw = call_brenda("getKmValue", params)
    if os.environ.get("BRENDA_DEBUG"):
        _debug_write("getKmValue", raw)
    return split_entries(raw)


def get_reactions(
    ec_number: str,
    organism: str = "*",
    reaction: str = "*",
    commentary: str = "*",
    literature: str = "*",
) -> List[str]:
    """
    Retrieve reaction equations for the given EC number.
    Returns list of BRENDA entry strings (ecNumber*...#organism*...).
    BRENDA Zeep API expects each filter as "key*value" (e.g. ecNumber*1.1.1.1, organism*).
    """
    email, password = _get_credentials()
    pwd_hash = _hash_password(password)
    # Official format: (email, password, "ecNumber*", "reaction*", "commentary*", "literature*", "organism*")
    params = [
        email,
        pwd_hash,
        _brenda_param("ecNumber", ec_number),
        _brenda_param("reaction", reaction),
        _brenda_param("commentary", commentary),
        _brenda_param("literature", literature),
        _brenda_param("organism", organism),
    ]
    raw = call_brenda("getReaction", params)
    if os.environ.get("BRENDA_DEBUG"):
        _debug_write("getReaction", raw)
    return split_entries(raw)


__all__ = [
    "load_env_from_file",
    "get_km_values",
    "get_reactions",
    "call_brenda",
    "split_entries",
    "WSDL_URL",
]


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="BRENDA SOAP client: get_km_values, get_reactions.")
    parser.add_argument("--test", action="store_true", help="Run tests for non-helper functions; output under example/database/brenda")
    parser.add_argument("--ec", type=str, default="1.1.1.1", help="EC number (e.g. 1.1.1.1)")
    parser.add_argument("--organism", type=str, default="*", help="Organism filter")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "brenda")
        os.makedirs(out_base, exist_ok=True)
        print("Testing get_km_values, get_reactions (require BRENDA_EMAIL, BRENDA_PASSWORD)")
        if not os.environ.get("BRENDA_EMAIL") or not os.environ.get("BRENDA_PASSWORD"):
            print("  Skip: set BRENDA_EMAIL and BRENDA_PASSWORD to run")
            exit(0)
        ec = "1.1.1.1"
        print("  get_km_values(...)")
        km = get_km_values(ec, organism="*")
        print(f"    -> {len(km)} entries")
        with open(os.path.join(out_base, "km_entries_sample.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(km[:20]) if km else "no data")
        print("  get_reactions(...)")
        rx = get_reactions(ec, organism="*")
        print(f"    -> {len(rx)} entries")
        with open(os.path.join(out_base, "reactions_entries_sample.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rx[:20]) if rx else "no data")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not os.environ.get("BRENDA_EMAIL") or not os.environ.get("BRENDA_PASSWORD"):
        print("Set BRENDA_EMAIL and BRENDA_PASSWORD to use BRENDA API")
        exit(1)
    km = get_km_values(args.ec, organism=args.organism)
    print(f"get_km_values: {len(km)} entries")
    rx = get_reactions(args.ec, organism=args.organism)
    print(f"get_reactions: {len(rx)} entries")
