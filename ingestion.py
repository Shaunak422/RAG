import os
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def load():

    documents = []

    data_path = os.path.abspath(DATA_DIR)

    print(f"Looking in: {data_path}")

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Loading {file_path}...")

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = file
                        doc.metadata["file_path"] = file_path

                    documents.extend(docs)
                    print(f"\nTotal text pages loaded: {len(documents)}")

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    print(f"\nTotal text pages loaded: {len(documents)}")
    return documents




if __name__ == "__main__":
    docs = load()

    for i, doc in enumerate(docs):
        print("File:", doc.metadata["source"])
        print("Page:", doc.metadata.get("page"))
        print(doc.page_content)
        print("\n" + "="*60 + "\n")