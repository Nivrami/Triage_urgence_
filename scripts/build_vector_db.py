"""
Reconstruit la base vectorielle ChromaDB à partir des documents sources.

La base (data/vector_db/) n'est pas versionnée : elle est générée par ce
script à partir des documents de data/rag_document/.

Usage:
    python scripts/build_vector_db.py            # construit si absente
    python scripts/build_vector_db.py --rebuild   # reconstruit depuis zéro
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.rag.vector_store import build_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Supprime et reconstruit la base même si elle existe déjà",
    )
    args = parser.parse_args()

    build_vector_store(
        documents_dir=str(ROOT_DIR / "data" / "rag_document"),
        persist_dir=str(ROOT_DIR / "data" / "vector_db"),
        force_rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
