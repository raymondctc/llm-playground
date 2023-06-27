import csv
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class PlayStoreReviewLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                # print(row.items())
                # content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                content = row['Review Text']
                last_update_ts = int(row['Review Last Update Millis Since Epoch'])
                submission_ts = int(row['Review Submit Millis Since Epoch'])
                rating = int(row['Star Rating'])
                # title = row['Review Title']
                lang = row['Reviewer Language']
                version_code = row['App Version Code']
                device = row['Device']
                
                int_version_code = 0
                if (version_code.isdigit()):
                    int_version_code = int(version_code) 
                else:
                    int_version_code = 0

                metadata = {
                    "last_update_ts": int(last_update_ts), 
                    "sub_ts": submission_ts,
                    "rating": rating,
                    # "title": title,
                    "lang": lang,
                    "version_code": int_version_code,
                    "device": device,
                    "source": self.file_path,
                    "platform": "android"
                }
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
                print(f"@@ {doc}")
        return docs