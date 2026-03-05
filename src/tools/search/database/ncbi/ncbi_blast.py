"""
NCBI BLAST: query (submit and return XML/text) and download (save result to file).
Uses Biopython Bio.Blast.NCBIWWW.qblast for NCBI web BLAST; optional NCBIXML parsing.
"""
import json
import os
import time
import argparse
from typing import Literal

try:
    from Bio.Blast import NCBIWWW, NCBIXML
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# Programs: blastn, blastp, blastx, tblastn, tblastx
BLAST_PROGRAMS = ("blastn", "blastp", "blastx", "tblastn", "tblastx")
# Common DBs: nt, refseq_rna (nucleotide); nr, refseq_protein, pdb, swissprot (protein)
DEFAULT_PROGRAM = "blastp"
BLAST_DATABASES = ("nt", "nr_cluster_seq", "refseq_select", "refseq_protein", "landmark", "swissprot", "nr", "pataa", "env_nr", "tsa_nr", "pdb")
DEFAULT_DATABASE = "swissprot"


def query_ncbi_blast(
    sequence: str,
    program: Literal[BLAST_PROGRAMS] = DEFAULT_PROGRAM,
    database: Literal[BLAST_DATABASES] = DEFAULT_DATABASE,
    expect: float = 0.001,
    hitlist_size: int = 50,
    alignments: int = 25,
    format_type: str = "XML",
    entrez_query: str = None,
    **kwargs,
) -> str:
    """
    Submit BLAST search to NCBI via Biopython. Returns XML string or error JSON.
    sequence: FASTA string, raw sequence, or accession (e.g. GenBank ID).
    program: blastn, blastp, blastx, tblastn, tblastx.
    database: e.g. nt, nr, refseq_protein, pdb, swissprot.
    """
    if not HAS_BIOPYTHON:
        return json.dumps({"success": False, "error": "Biopython is required: pip install biopython"})

    program = program.lower()
    if program not in BLAST_PROGRAMS:
        return json.dumps({"success": False, "error": f"program must be one of {BLAST_PROGRAMS}"})

    try:
        result_handle = NCBIWWW.qblast(
            program=program,
            database=database,
            sequence=sequence.strip(),
            expect=expect,
            hitlist_size=hitlist_size,
            alignments=alignments,
            format_type=format_type,
            entrez_query=entrez_query,
            **kwargs,
        )
        xml_text = result_handle.read()
        result_handle.close()
        time.sleep(0.5)  # NCBI rate limit
        return xml_text
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def download_ncbi_blast(
    sequence: str,
    out_path: str,
    program: Literal[BLAST_PROGRAMS] = DEFAULT_PROGRAM,
    database: Literal[BLAST_DATABASES] = DEFAULT_DATABASE,
    expect: float = 0.001,
    hitlist_size: int = 50,
    alignments: int = 25,
    format_type: str = "XML",
    entrez_query: str = None,
    **kwargs,
) -> str:
    """
    Run BLAST and save result to file. Returns message string.
    out_path: path to output file (e.g. .xml or .txt).
    """
    if not HAS_BIOPYTHON:
        return "Failed: Biopython is required (pip install biopython)"

    result = query_ncbi_blast(
        sequence,
        program=program,
        database=database,
        expect=expect,
        hitlist_size=hitlist_size,
        alignments=alignments,
        format_type=format_type,
        entrez_query=entrez_query,
        **kwargs,
    )
    if result.strip().startswith("{"):
        try:
            err = json.loads(result)
            return f"BLAST failed: {err.get('error', result)}"
        except json.JSONDecodeError:
            pass

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    return f"BLAST result saved to {out_path}"


def parse_blast_xml(xml_content_or_path: str) -> list:
    """
    Parse BLAST XML (string or file path) into a list of hit summaries.
    Returns list of dicts: query, query_length, num_alignments, hits (list of alignment summaries).
    """
    if not HAS_BIOPYTHON:
        return []

    if os.path.isfile(xml_content_or_path):
        with open(xml_content_or_path, encoding="utf-8") as f:
            xml_content = f.read()
    else:
        xml_content = xml_content_or_path

    from io import StringIO
    records = []
    try:
        with StringIO(xml_content) as h:
            for blast_record in NCBIXML.parse(h):
                hits = []
                for aln in blast_record.alignments:
                    hsp = aln.hsps[0] if aln.hsps else None
                    hits.append({
                        "title": aln.title,
                        "accession": getattr(aln, "accession", ""),
                        "length": aln.length,
                        "e_value": hsp.expect if hsp else None,
                        "score": hsp.score if hsp else None,
                        "identities": f"{hsp.identities}/{hsp.align_length}" if hsp else None,
                        "percent_identity": (hsp.identities / hsp.align_length * 100) if (hsp and hsp.align_length) else None,
                    })
                records.append({
                    "query": blast_record.query,
                    "query_length": blast_record.query_length,
                    "database": blast_record.database,
                    "num_alignments": len(blast_record.alignments),
                    "hits": hits,
                })
    except Exception:
        pass
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCBI BLAST: submit query and save result (requires Biopython)")
    parser.add_argument("--test", action="store_true", help="Run tests for query_ncbi_blast, download_ncbi_blast, parse_blast_xml; output under example/database/ncbi")
    parser.add_argument("-i", "--sequence", help="Query: sequence string, FASTA, or accession")
    parser.add_argument("-f", "--fasta_file", help="FASTA file (single sequence used)")
    parser.add_argument("-o", "--out", help="Output file path (e.g. result.xml)")
    parser.add_argument("-p", "--program", default=DEFAULT_PROGRAM, choices=list(BLAST_PROGRAMS), help="BLAST program")
    parser.add_argument("-d", "--database", default=DEFAULT_DATABASE, choices=list(BLAST_DATABASES), help="BLAST database (e.g. nr, nt)")
    parser.add_argument("-e", "--expect", type=float, default=0.001, help="E-value threshold")
    parser.add_argument("--hitlist_size", type=int, default=50, help="Max hits to return")
    parser.add_argument("--alignments", type=int, default=25, help="Max alignments to show")
    parser.add_argument("--entrez_query", default=None, help="Entrez query to filter (e.g. 'Mus musculus[Organism]')")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "ncbi", "blast")
        os.makedirs(out_base, exist_ok=True)
        # Short peptide to keep BLAST test fast
        test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        print("Testing query_ncbi_blast(...) [short sequence]")
        xml_result = query_ncbi_blast(test_seq, program="blastp", database="swissprot", hitlist_size=5, alignments=3)
        if xml_result.strip().startswith("{"):
            print("  BLAST returned error (e.g. rate limit):", xml_result[:200])
        else:
            out_xml = os.path.join(out_base, "blast_result_sample.xml")
            with open(out_xml, "w", encoding="utf-8") as f:
                f.write(xml_result[:50000] if len(xml_result) > 50000 else xml_result)
            print(f"  saved to {out_xml}")
            print("Testing parse_blast_xml(...)")
            parsed = parse_blast_xml(xml_result)
            print(f"  parsed {len(parsed)} record(s)")
            with open(os.path.join(out_base, "blast_parsed_sample.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, default=str)
        print("Testing download_ncbi_blast(...)")
        out_path = os.path.join(out_base, "blast_download_sample.xml")
        msg = download_ncbi_blast(test_seq, out_path, program="blastp", database="swissprot", hitlist_size=5)
        print(f"  {msg}")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not args.out:
        print("Error: -o/--out required (or use --test)")
        exit(1)
    if not args.sequence and not args.fasta_file:
        print("Error: provide -i/--sequence or -f/--fasta_file")
        exit(1)

    if args.fasta_file:
        if not os.path.isfile(args.fasta_file):
            print(f"Error: file not found {args.fasta_file}")
            exit(1)
        with open(args.fasta_file) as f:
            sequence = f.read()
    else:
        sequence = args.sequence

    msg = download_ncbi_blast(
        sequence,
        args.out,
        program=args.program,
        database=args.database,
        expect=args.expect,
        hitlist_size=args.hitlist_size,
        alignments=args.alignments,
        entrez_query=args.entrez_query,
    )
    print(msg)
    if "successfully" in msg or os.path.isfile(args.out):
        print(f"Done. Output: {args.out}")
