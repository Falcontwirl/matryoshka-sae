"""Streams raw text for SAE activation collection."""

from datasets import load_dataset


def stream_text(dataset_name: str, text_field: str = "text", seed: int = 0):
    """Infinite generator of raw text strings from a streamed HF dataset.

    Restarts from the beginning if the underlying stream is ever exhausted.
    Shouldn't happen mid-run given dataset size, but keeps the pipeline from
    crashing outright if it does.
    """
    while True:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for example in ds:
            text = example[text_field]
            if text:
                yield text
