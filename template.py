import os
import pathlib

def creare_structure():
    structure = {
        "": [
            "app.py",
            "requirements.txt",
            ".env"
        ],
        "data/": [
            "documents/"
        ],
        "src": [
            "ingestion/",
            "embeddings/",
            "retrieval/",
            "reranking/",
            "memory/",
            "generation/",
            "pipeline/"
        ],
        "vectorstore/": [
            "index/"
        ],
        "tests/": []
    }

    for folder, files in structure.items():
        if folder:
            os.makedirs(folder, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder, file) if folder else file

            if file.endswith("/"):
                os.makedirs(file_path, exist_ok=True)
                print(f"Created folder: {file_path}")
            else:
                pathlib.Path(file_path).touch()
                print(f"Created file: {file_path}")

    print("Project structure created successfully.")

if __name__ == "__main__":
    creare_structure()
