from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

DATASET_NAME = "Greek‑Economy‑RAG‑eval"

example = {
    "inputs": {
        "question": (
            "What was the average headline inflation rate for Greece in 2024, and how did it compare to the euro area average?"
        )
    },
    "outputs": {
        "answer": (
            "The average headline inflation rate for Greece in 2024 was 3.0%, which was above the euro area average of 2.4%"
        )
    },
}

client = Client()

# Check if the dataset exists
existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing_datasets:
    ds = existing_datasets[0]
else:
    ds = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Ground‑truth Q‑A pairs for the GPT‑4‑o Greek‑economy RAG backend",
    )

# Add the example to the dataset
client.create_examples(
    dataset_id=ds.id,
    examples=[example],
)

# Retrieve the dataset to get the updated example count
updated_ds = client.read_dataset(dataset_id=ds.id)
examples = list(client.list_examples(dataset_id=updated_ds.id))
print(f"✅ Dataset “{DATASET_NAME}” now has {len(examples)} examples.")
