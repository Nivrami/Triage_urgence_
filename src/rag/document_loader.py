"""
Chargement et prétraitement des documents médicaux.
"""

import re
from pathlib import Path


class DocumentLoader:
    """
    Charge et prétraite les documents pour le RAG.

    Sources supportées:
    - Fichiers locaux (txt, json, csv, pdf)
    - HuggingFace datasets
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Initialise le loader.

        Args:
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks

        JUSTIFIER: Pourquoi ces valeurs?
        - chunk_size=500: Assez pour le contexte, pas trop pour l'embedding
        - chunk_overlap=50: Évite de couper des phrases importantes
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_file(self, file_path: str) -> list[dict]:
        """
        Charge depuis un fichier local.

        Supporte: .txt, .json, .csv, .pdf

        Args:
            file_path: Chemin vers le fichier

        Returns:
            Liste de {"text": ..., "metadata": {...}}
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".txt":
            return self._load_txt(file_path)
        elif suffix == ".json":
            return self._load_json(file_path)
        elif suffix == ".csv":
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Format non supporté: {suffix}")

    def _load_pdf(self, file_path: Path) -> list[dict]:
        """Charge un fichier PDF avec PyMuPDF."""
        import fitz  # PyMuPDF

        documents = []
        doc = fitz.open(file_path)

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            text = self.preprocess_text(text)

            if text.strip():  # Ignorer les pages vides
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": str(file_path),
                            "page": page_num,
                            "total_pages": len(doc),
                            "type": "pdf",
                        },
                    }
                )

        doc.close()
        return documents

    def _load_txt(self, file_path: Path) -> list[dict]:
        """Charge un fichier texte."""
        text = file_path.read_text(encoding="utf-8")
        text = self.preprocess_text(text)

        return [{"text": text, "metadata": {"source": str(file_path), "type": "txt"}}]

    def _load_json(self, file_path: Path) -> list[dict]:
        """Charge un fichier JSON."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        # Si c'est une liste de documents
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "text" in item:
                    documents.append(
                        {
                            "text": self.preprocess_text(item["text"]),
                            "metadata": {
                                "source": str(file_path),
                                "index": i,
                                "type": "json",
                                **item.get("metadata", {}),
                            },
                        }
                    )
                elif isinstance(item, str):
                    documents.append(
                        {
                            "text": self.preprocess_text(item),
                            "metadata": {"source": str(file_path), "index": i, "type": "json"},
                        }
                    )
        # Si c'est un objet unique
        elif isinstance(data, dict) and "text" in data:
            documents.append(
                {
                    "text": self.preprocess_text(data["text"]),
                    "metadata": {
                        "source": str(file_path),
                        "type": "json",
                        **data.get("metadata", {}),
                    },
                }
            )

        return documents

    def _load_csv(self, file_path: Path) -> list[dict]:
        """Charge un fichier CSV."""
        import csv

        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Cherche une colonne "text" ou utilise toutes les colonnes
                if "text" in row:
                    text = row["text"]
                else:
                    text = " ".join(str(v) for v in row.values())

                documents.append(
                    {
                        "text": self.preprocess_text(text),
                        "metadata": {
                            "source": str(file_path),
                            "row": i,
                            "type": "csv",
                            **{k: v for k, v in row.items() if k != "text"},
                        },
                    }
                )

        return documents

    def load_from_directory(self, dir_path: str, extensions: list[str] = None) -> list[dict]:
        """
        Charge tous les fichiers d'un répertoire.

        Args:
            dir_path: Chemin du répertoire
            extensions: Extensions à charger (ex: [".pdf", ".txt"])
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {dir_path}")

        if extensions is None:
            extensions = [".pdf", ".txt", ".json", ".csv"]

        documents = []

        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    docs = self.load_from_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_path}: {e}")

        return documents

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Découpe un document en chunks.

        Args:
            document: {"text": ..., "metadata": {...}}

        Returns:
            Liste de chunks avec métadonnées préservées
        """
        text = document["text"]
        metadata = document.get("metadata", {})

        # Si le texte est plus petit que chunk_size, pas besoin de découper
        if len(text) <= self.chunk_size:
            return [{"text": text, "metadata": {**metadata, "chunk_index": 0, "is_chunked": False}}]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculer la fin du chunk
            end = start + self.chunk_size

            # Si on n'est pas à la fin, essayer de couper proprement
            if end < len(text):
                # Chercher le dernier point ou retour à la ligne
                last_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                )

                # Si on trouve une coupure naturelle, l'utiliser
                if last_break > start + self.chunk_size // 2:
                    end = last_break + 1

            # Extraire le chunk
            chunk_text = text[start:end].strip()

            if chunk_text:  # Ignorer les chunks vides
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                            "is_chunked": True,
                        },
                    }
                )
                chunk_index += 1

            # Avancer avec overlap
            start = end - self.chunk_overlap

            # Éviter une boucle infinie
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Découpe plusieurs documents."""
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def preprocess_text(self, text: str) -> str:
        """
        Nettoie le texte.

        - Supprime les espaces multiples
        - Normalise les caractères
        - Supprime les caractères spéciaux inutiles
        """
        if not text:
            return ""

        # Remplacer les retours à la ligne multiples par un seul
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Remplacer les espaces multiples par un seul
        text = re.sub(r"[ \t]+", " ", text)

        # Supprimer les espaces en début/fin de ligne
        text = "\n".join(line.strip() for line in text.split("\n"))

        # Supprimer les caractères de contrôle
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        return text.strip()

    def load_from_huggingface(
        self, dataset_name: str, split: str = "train", text_column: str = "text"
    ) -> list[dict]:
        """
        Charge depuis HuggingFace.

        Args:
            dataset_name: Nom du dataset (ex: "mlabonne/medical-cases-fr")
            split: Split à charger
            text_column: Colonne contenant le texte

        Returns:
            Liste de {"text": ..., "metadata": {...}}
        """
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        documents = []

        for i, item in enumerate(dataset):
            text = item.get(text_column, "")
            if text:
                documents.append(
                    {
                        "text": self.preprocess_text(text),
                        "metadata": {
                            "source": dataset_name,
                            "index": i,
                            "type": "huggingface",
                            **{k: v for k, v in item.items() if k != text_column},
                        },
                    }
                )

        return documents

    def load_gravity_categories(self) -> list[dict]:
        """
        Charge les catégories de gravité depuis les ressources du projet.
        """
        # Catégories de gravité CIMU
        gravity_categories = [
            {
                "text": "ROUGE (Niveau 1): Urgence vitale immédiate. Patient en détresse vitale nécessitant une prise en charge immédiate. Exemples: arrêt cardiaque, détresse respiratoire sévère, polytraumatisme grave, AVC en phase aiguë.",
                "metadata": {"category": "gravite", "level": "ROUGE", "priority": 1},
            },
            {
                "text": "ORANGE (Niveau 2): Urgence vraie. État clinique instable ou potentiellement grave. Exemples: douleur thoracique, dyspnée modérée, traumatisme crânien avec perte de connaissance brève.",
                "metadata": {"category": "gravite", "level": "ORANGE", "priority": 2},
            },
            {
                "text": "JAUNE (Niveau 3): Urgence relative. État stable mais nécessitant une évaluation rapide. Exemples: fièvre élevée, douleur abdominale, fracture simple.",
                "metadata": {"category": "gravite", "level": "JAUNE", "priority": 3},
            },
            {
                "text": "VERT (Niveau 4): Consultation non urgente. État stable sans critère de gravité. Exemples: plaie superficielle, entorse légère, symptômes mineurs.",
                "metadata": {"category": "gravite", "level": "VERT", "priority": 4},
            },
            {
                "text": "BLEU (Niveau 5): Consultation différable. Problème chronique ou mineur ne relevant pas des urgences. Orientation vers médecine de ville recommandée.",
                "metadata": {"category": "gravite", "level": "BLEU", "priority": 5},
            },
        ]

        return gravity_categories

    def load_medical_cases_fr(self) -> list[dict]:
        """
        Charge le dataset medical-cases-fr de HuggingFace.
        """
        return self.load_from_huggingface(
            dataset_name="mlabonne/medical-cases-fr", text_column="text"
        )
