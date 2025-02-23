from datetime import datetime
from timescale_vector import client

import sys
import os

# Get the absolute path of the current script (myscript.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move two levels up to the 'project' directory
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "app"))

# Add the project root to sys.path
sys.path.append(project_root)

from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "What are your shipping options?"
results = vec.search(relevant_question, limit=3)

response = Synthesizer.generate_response(question=relevant_question, context=results, prompt="system", subject_domain="an e-commerce FAQ system")

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vec.search(irrelevant_question, limit=3)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

metadata_filter = {"category": "Shipping"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Advanced filtering using Predicates
# --------------------------------------------------------------

predicates = client.Predicates("category", "==", "Shipping")
results = vec.search(relevant_question, limit=3, predicates=predicates)


predicates = client.Predicates("category", "==", "Shipping") | client.Predicates(
    "category", "==", "Services"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)


predicates = client.Predicates("category", "==", "Shipping") & client.Predicates(
    "created_at", ">", "2024-09-01"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)

# --------------------------------------------------------------
# Time-based filtering
# --------------------------------------------------------------

# September — Returning results
time_range = (datetime(2024, 9, 1), datetime(2024, 9, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)

# August — Not returning any results
time_range = (datetime(2024, 8, 1), datetime(2024, 8, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)
