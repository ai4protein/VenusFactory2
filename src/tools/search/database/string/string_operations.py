"""
STRING DB: query (return text/bytes) and download (save to file) for all endpoints.
Re-exports query_* / download_* and keeps backward-compat aliases (string_* from atomic modules).

Endpoints: version, map_ids, network, network_image, interaction_partners,
  enrichment, ppi_enrichment, homology.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Union

try:
    from .string_map_ids import string_map_ids
    from .string_network import string_network
    from .string_network_image import string_network_image
    from .string_interaction_partners import string_interaction_partners
    from .string_enrichment import string_enrichment
    from .string_ppi_enrichment import string_ppi_enrichment
    from .string_homology import string_homology
    from .string_version import string_version
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "string" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.string.string_map_ids import string_map_ids
    from src.tools.search.database.string.string_network import string_network
    from src.tools.search.database.string.string_network_image import string_network_image
    from src.tools.search.database.string.string_interaction_partners import string_interaction_partners
    from src.tools.search.database.string.string_enrichment import string_enrichment
    from src.tools.search.database.string.string_ppi_enrichment import string_ppi_enrichment
    from src.tools.search.database.string.string_homology import string_homology
    from src.tools.search.database.string.string_version import string_version


# ---------- version (meta/info) ----------
def query_string_version() -> str:
    """Query STRING DB version. Returns TSV text. No file save."""
    return string_version()


def download_string_version(out_dir: str, filename: str = "version.txt") -> str:
    """Download STRING version to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_version()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- map_ids ----------
def query_string_map_ids(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Map protein names/IDs to STRING IDs. Returns TSV text. No file save."""
    return string_map_ids(identifiers, species=species, limit=limit, echo_query=echo_query, caller_identity=caller_identity)


def download_string_map_ids(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "map_ids.tsv",
) -> str:
    """Download map_ids result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_map_ids(identifiers, species=species, limit=limit, echo_query=echo_query, caller_identity=caller_identity)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- network ----------
def query_string_network(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Get PPI network (TSV). Returns text. No file save."""
    return string_network(
        identifiers,
        species=species,
        required_score=required_score,
        network_type=network_type,
        add_nodes=add_nodes,
        caller_identity=caller_identity,
    )


def download_string_network(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "network.tsv",
) -> str:
    """Download network TSV to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_network(
        identifiers,
        species=species,
        required_score=required_score,
        network_type=network_type,
        add_nodes=add_nodes,
        caller_identity=caller_identity,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- network_image ----------
def query_string_network_image(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
) -> bytes:
    """Get network as PNG. Returns bytes. No file save."""
    return string_network_image(
        identifiers,
        species=species,
        required_score=required_score,
        network_flavor=network_flavor,
        add_nodes=add_nodes,
        caller_identity=caller_identity,
    )


def download_string_network_image(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "network.png",
) -> str:
    """Download network PNG to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    data = query_string_network_image(
        identifiers,
        species=species,
        required_score=required_score,
        network_flavor=network_flavor,
        add_nodes=add_nodes,
        caller_identity=caller_identity,
    )
    with open(out_path, "wb") as f:
        f.write(data)
    return f"Saved to {out_path}"


# ---------- interaction_partners ----------
def query_string_interaction_partners(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Get interaction partners (TSV). Returns text. No file save."""
    return string_interaction_partners(
        identifiers,
        species=species,
        required_score=required_score,
        limit=limit,
        caller_identity=caller_identity,
    )


def download_string_interaction_partners(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "interaction_partners.tsv",
) -> str:
    """Download interaction partners TSV to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_interaction_partners(
        identifiers,
        species=species,
        required_score=required_score,
        limit=limit,
        caller_identity=caller_identity,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- enrichment ----------
def query_string_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Functional enrichment (TSV). Returns text. No file save."""
    return string_enrichment(identifiers, species=species, caller_identity=caller_identity)


def download_string_enrichment(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "enrichment.tsv",
) -> str:
    """Download enrichment TSV to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_enrichment(identifiers, species=species, caller_identity=caller_identity)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- ppi_enrichment ----------
def query_string_ppi_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """PPI enrichment (JSON). Returns text. No file save."""
    return string_ppi_enrichment(
        identifiers,
        species=species,
        required_score=required_score,
        caller_identity=caller_identity,
    )


def download_string_ppi_enrichment(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "ppi_enrichment.json",
) -> str:
    """Download ppi_enrichment JSON to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_ppi_enrichment(
        identifiers,
        species=species,
        required_score=required_score,
        caller_identity=caller_identity,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- homology ----------
def query_string_homology(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Homology scores (TSV). Returns text. No file save."""
    return string_homology(identifiers, species=species, caller_identity=caller_identity)


def download_string_homology(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    caller_identity: str = "claude_scientific_skills",
    filename: str = "homology.tsv",
) -> str:
    """Download homology TSV to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    text = query_string_homology(identifiers, species=species, caller_identity=caller_identity)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STRING DB: query/download all endpoints; use --test to run sample and save under example/database/string")
    parser.add_argument("--test", action="store_true", help="Run tests; output under example/database/string")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "string")
        os.makedirs(out_base, exist_ok=True)
        meta_dir = os.path.join(out_base, "metadata")
        test_id = "P43403"
        species = 9606

        print("Testing query_string_version / download_string_version(...)")
        v = query_string_version()
        with open(os.path.join(out_base, "query_version_sample.txt"), "w", encoding="utf-8") as f:
            f.write(v[:2000] if len(v) > 2000 else v)
        msg = download_string_version(meta_dir, "version.txt")
        print(f"  {msg}")

        print("Testing query_string_map_ids / download_string_map_ids(...)")
        t = query_string_map_ids(test_id, species=species)
        with open(os.path.join(out_base, "query_map_ids_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        msg = download_string_map_ids(test_id, meta_dir, species=species, filename="map_ids.tsv")
        print(f"  {msg}")

        print("Testing query_string_network / download_string_network(...)")
        t = query_string_network(test_id, species=species)
        with open(os.path.join(out_base, "query_network_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(t[:5000] if len(t) > 5000 else t)
        msg = download_string_network(test_id, out_base, species=species, filename="network.tsv")
        print(f"  {msg}")

        print("Testing query_string_network_image / download_string_network_image(...)")
        data = query_string_network_image(test_id, species=species)
        msg = download_string_network_image(test_id, out_base, species=species, filename="network.png")
        print(f"  {msg}")

        print("Testing query_string_interaction_partners / download_string_interaction_partners(...)")
        t = query_string_interaction_partners(test_id, species=species, limit=5)
        with open(os.path.join(out_base, "query_interaction_partners_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        msg = download_string_interaction_partners(test_id, out_base, species=species, limit=5, filename="interaction_partners.tsv")
        print(f"  {msg}")

        print("Testing query_string_enrichment / download_string_enrichment(...)")
        t = query_string_enrichment(test_id, species=species)
        with open(os.path.join(out_base, "query_enrichment_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(t[:5000] if len(t) > 5000 else t)
        msg = download_string_enrichment(test_id, out_base, species=species, filename="enrichment.tsv")
        print(f"  {msg}")

        print("Testing query_string_ppi_enrichment / download_string_ppi_enrichment(...)")
        t = query_string_ppi_enrichment(test_id, species=species)
        with open(os.path.join(out_base, "query_ppi_enrichment_sample.json"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        msg = download_string_ppi_enrichment(test_id, out_base, species=species, filename="ppi_enrichment.json")
        print(f"  {msg}")

        print("Testing query_string_homology / download_string_homology(...)")
        t = query_string_homology(test_id, species=species)
        with open(os.path.join(out_base, "query_homology_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        msg = download_string_homology(test_id, out_base, species=species, filename="homology.tsv")
        print(f"  {msg}")

        print(f"Done. Output under {out_base}")
        sys.exit(0)

    parser.print_help()
