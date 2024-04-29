from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from dataclasses import dataclass, field
from litgpt.prompts import PromptStyle

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch
from torch import Tensor

from transformers import PreTrainedTokenizer

class Identity(PromptStyle):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt + " "


@dataclass
class ParquetData(DataModule):
    """Alpaca data module for supervised finetuning."""

    data_path: Union[str, Path] = Path("data/")
    mask_prompt: bool = True
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 114514
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""

    tokenizer: Optional[PreTrainedTokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.prompt_style = Identity()

    def connect(
        self, tokenizer: Optional[PreTrainedTokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:

        df = pd.read_parquet(self.data_path)
        train_data = []
        for _, row in df.iterrows():
            if len(row["response"]) == 0:
                continue
            train_data.append({"instruction": row["prompt"], "output": row["response"]})

        self.train_dataset = HF_SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = HF_SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )



class HF_SFTDataset(SFTDataset):

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        super().__init__(data, tokenizer, prompt_style, max_seq_length, mask_prompt, ignore_index, transform)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        example = self.data[idx]
        if self.transform is not None:
            example = self.transform(example)
        prompt = self.prompt_style.apply(prompt=example["instruction"], **example)
        prompt_and_response = prompt + example["output"]
        encoded_prompt = self.tokenizer.encode(prompt)
        encoded_prompt_and_response = self.tokenizer.encode(
            prompt_and_response, max_length=self.max_seq_length if self.max_seq_length != -1 else None
        )
        encoded_prompt_and_response.append(self.tokenizer.eos_token_id)

        encoded_prompt = torch.tensor(encoded_prompt)
        encoded_prompt_and_response = torch.tensor(encoded_prompt_and_response)

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64)}
