from datetime import datetime
import pandas as pd
from timescale_vector.client import uuid_from_time

import sys
import os

# Get the absolute path of the current script (myscript.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move two levels up to the 'project' directory
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "app"))

# Add the project root to sys.path
sys.path.append(project_root)

from database.vector_store import VectorStore

# Initialize VectorStore
vec = VectorStore()

csv_file = os.path.join(project_root, "..", "data", "faq_dataset.csv")

# Read the CSV file
df = pd.read_csv(csv_file, sep=";")

# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)
