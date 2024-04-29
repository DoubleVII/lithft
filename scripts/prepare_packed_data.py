from pathlib import Path
import glob
import os
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from litgpt.data.packed_dataset import PackedDatasetBuilder
import numpy as np
from tqdm import tqdm

effective_block_size = 2048
batch_size = 32


def main(
    source_path: str = "/", destination_path: str = "/", tokenizer_path: str = "/", prefix: str = "data_part", chunk_size_n: int = 8
):

    token_count = 0

    destination_path = Path(destination_path)
    source_path = Path(source_path)
    tokenizer_path = Path(tokenizer_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    parquet_files = glob.glob(os.path.join(source_path, "*.parquet"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    builder = PackedDatasetBuilder(
        outdir=destination_path,
        prefix=prefix,
        chunk_size=(effective_block_size + 1) * 1024 * chunk_size_n,
        sep_token=tokenizer.eos_token_id,
        dtype="auto",
        vocab_size=len(tokenizer),
    )

    for file_path in tqdm(parquet_files):

        df = pd.read_parquet(file_path)

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            text_batch = [row.text for row in batch.itertuples()]
            text_ids_batch = tokenizer(text_batch)
            text_ids_batch = text_ids_batch.input_ids
            for text_ids in text_ids_batch:
                text_ids.append(tokenizer.eos_token_id)
                token_count += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

    builder.write_reminder()
    print("Total token count:", token_count)


if __name__ == "__main__":
    from scripts import CLI

    CLI(main)
