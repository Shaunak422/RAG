import os
import json
import hashlib
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
HASH_MANIFEST = os.path.join(BASE_DIR, "..", "db", ".ingested_hashes.json")  # tracks ingested files


def _file_hash(filepath):
    # Compute SHA-256 hash of a file for deduplication
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest():
    # Load already-ingested file hashes from disk
    if os.path.exists(HASH_MANIFEST):
        with open(HASH_MANIFEST, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest):
    # Save ingested file hashes to disk
    os.makedirs(os.path.dirname(HASH_MANIFEST), exist_ok=True)
    with open(HASH_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def load():
    # Load PDFs from /data, skip files already ingested (hash-based dedup)
    documents = []
    data_path = os.path.abspath(DATA_DIR)
    manifest = _load_manifest()
    new_files = 0

    print(f"Looking in: {data_path}")

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                fhash = _file_hash(file_path)

                # Skip if already ingested
                if fhash in manifest:
                    print(f"⏭️  Skipping (already ingested): {file}")
                    continue

                print(f"Loading {file_path}...")
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = file
                        doc.metadata["file_path"] = file_path
                        doc.metadata["file_hash"] = fhash

                    documents.extend(docs)
                    manifest[fhash] = file
                    new_files += 1

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    _save_manifest(manifest)
    print(f"\n📄 New files loaded: {new_files} | New pages: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load()
    for i, doc in enumerate(docs):
        print("File:", doc.metadata["source"])
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content)
        print("\n" + "=" * 60 + "\n")