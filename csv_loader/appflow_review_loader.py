import csv
from datetime import datetime
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class AppFlowReviewLoader(BaseLoader):
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
            # Skip first line
            next(csvfile)

            # Parse the second line to get the platform
            review_meta_line = next(csvfile)
            # Find the index of the start and end positions of "Store"
            start_index = review_meta_line.find("Store: ") + len("Store: ")
            end_index = review_meta_line.find(",", start_index)

            # Extract the value of "Store" from the line
            store_value = review_meta_line[start_index:end_index]

            platform = ''
            if store_value == 'itunes':
                platform = 'ios'
            elif store_value == 'googleplay':
                platform = 'android'
                

            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = row['Review']
                last_update_ts_obj = datetime.strptime(row['Updated'], "%Y-%m-%dT%H:%M:%S")
                last_update_ts = last_update_ts_obj.timestamp()

                submission_ts_obj = datetime.strptime(row['Publication date'], "%Y-%m-%dT%H:%M:%S")
                submission_ts = submission_ts_obj.timestamp()
                
                rating = row['Rating']
                lang = row['Review Language']
                version = row['Version']

                metadata = {
                    "last_update_ts": int(last_update_ts),
                    "sub_ts": int(submission_ts),
                    "rating": rating,
                    "lang": lang,
                    "version_code": "",
                    "version": version,
                    "device": "",
                    "source": self.file_path,
                    "platform": platform
                }
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
                print(f"@@ {doc}")
        return docs