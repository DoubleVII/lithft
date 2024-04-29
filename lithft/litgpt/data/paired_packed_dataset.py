# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from litgpt.data import DataModule
from pathlib import Path
from torch.utils.data import DataLoader
import glob

from litgpt.data.packed_dataset import code, dtypes


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes




@dataclass
class PairedPackedData(DataModule):

    data_path: Union[str, Path] = Path("data/")
    """The path to the data directory containing the preprocessed chunks for the streaming dataset
    The path can also be a remote path (e.g., s3://). See also ``split_names`` if this path contains subfolders
    for training- and validation splits."""
    split_names: Optional[Tuple[str, str]] = None
    """Optional tuple for names of subfolders for training and validation under ``data_path``. If not provided,
    all data under data_path will be used for training, and the validation dataloader will be identical to the
    train dataloader."""
    seed: int = 114514
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""
    shuffle: bool = True

    file_prefixes: str = "data_part"
    "e.g. 'data_part,data_part_2'"
    prefix_weights: str = "1.0"
    "e.g. '1.0,0.5'"
    pair_file_prefixes: str = "pair_data_part"

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self) -> None:
        if self.split_names is not None and len(self.split_names) != 2:
            raise ValueError("If provided `split_names` must be a tuple of two strings, for example: ('train', 'val').")

    def connect(
        self, tokenizer = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.file_prefixes = self.file_prefixes.split(",")
        self.pair_file_prefixes = self.pair_file_prefixes.split(",")
        self.prefix_weights = [float(el) for el in self.prefix_weights.split(",")]
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self, fabric) -> DataLoader:
        datasets = []
        for prefix, pair_data_prefix in zip(self.file_prefixes, self.pair_file_prefixes):
            filenames = sorted(glob.glob(str(self.data_path) + f"/{prefix}*"))
            pair_filenames = sorted(glob.glob(str(self.data_path) + f"/{pair_data_prefix}*"))
            assert len(filenames) == len(pair_filenames), "Number of files must match"

            filename_pairs = list(zip(filenames, pair_filenames)) 
            random.seed(self.seed)
            if self.shuffle:
                random.shuffle(filename_pairs)

            dataset = PairedPackedDataset(
                filename_pairs,
                # n_chunks control the buffer size. 
                # Note that the buffer size also impacts the random shuffle
                # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
                n_chunks=2,
                block_size=self.seq_length,
                shuffle=self.shuffle,
                seed=self.seed+fabric.global_rank,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
            )
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError(
                f"No data found at {self.data_path}. Make sure you ran prepare_redpajama.py to create the dataset."
            )

        weights = self.prefix_weights
        sum_weights = sum(weights)
        weights = [el / sum_weights for el in weights]

        combined_dataset = CombinedDataset(datasets=datasets, seed=self.seed, weights=weights)

        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def val_dataloader(self, fabric) -> DataLoader:
        return None




class PairedPackedDataset(IterableDataset):
    def __init__(
        self, filename_pairs, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0
    ):
        self._filename_pairs = filename_pairs
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filename_pairs) // num_shards * num_shards
        filename_pairs = self._filename_pairs[shard_id:max_num_files:num_shards]

        return PairedPackedDatasetIterator(
            filename_pairs=filename_pairs,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )

class PairedPackedDatasetIterator:
    def __init__(self, filename_pairs, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filename_pairs = filename_pairs
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._pair_data_dtype = None

        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []
        self._pair_data_mmaps = []
        self._pair_data_buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()
        for mmap in self._pair_data_mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        # load file in _filenames, if no file are left, load from the beginning
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []
        self._pair_data_mmaps = []
        self._pair_data_buffers = []

        load_count = 0
        while load_count < self._n_chunks:
            if self._file_idx >= len(self._filename_pairs):
                self._file_idx = 0
            filename, pair_filename = self._filename_pairs[self._file_idx]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._pair_data_dtype, pair_chunk_size = self._read_header(pair_filename)
                assert pair_chunk_size == self._chunk_size, "Chunk size of pair data must match"
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            pair_data_mmap = np.memmap(pair_filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))
            self._pair_data_mmaps.append(pair_data_mmap)
            self._pair_data_buffers.append(memoryview(pair_data_mmap))

            load_count += 1
            self._file_idx += 1

        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers
        del self._pair_data_mmaps
        del self._pair_data_buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)

        pair_data_buffer = self._pair_data_buffers[chunk_id]
        pair_data_offset = np.dtype(self._pair_data_dtype).itemsize * elem_id
        pair_data_arr = np.frombuffer(pair_data_buffer, dtype=self._pair_data_dtype, count=self._block_size, offset=pair_data_offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64)), torch.from_numpy(pair_data_arr.astype(self._pair_data_dtype))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)




