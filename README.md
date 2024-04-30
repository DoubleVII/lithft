<div align="center">


# ⚡️🤗LitHFT

**Pretrain, finetune any LLMs from huggingface on your own data.**

</div>

## What is LitHFT

LitHFT is a tool developed based on [LitGPT](https://github.com/Lightning-AI/litgpt) for pre-training and fine-tuning Language Model (LLMs) from [Huggingface](https://huggingface.co/).

In comparison, LitGPT is specifically designed for certain types of LLMs and supports over 20 commonly used LLMs. However, it is not applicable to other types of models in Huggingface, such as [Qwen](https://huggingface.co/Qwen). LitHFT is suitable for native Huggingface models without any need for checkpoint conversion, but the trade-off is less training optimization. 

|Comparison|LitHFT|LitGPT|
|----|----|----|
|LLMs|Any|20+|
|Optimization|Deepspeed|FSDP|
|Dataloader|packed data from [TinyLLama](https://github.com/jzhang38/TinyLlama)|litdata|


## Install LitHFT

```bash
git clone https://github.com/DoubleVII/lithft.git
cd lithft
pip install -e .
```

## Warning
This project has only been tested on Qwen and Mistral.

|Model|Device|Throughput / s / device|
|----|----|----|
|Qwen1.5-1.8B|A800-40G|24.5k tokens|
|Mistral-7B|A100-80G|3.5k tokens|


# Quick start

## Finetune

### Data
We use [`ParquetData`](./litgpt/data/parquet_sft_data.py) as an example here, which contains two columns: `prompt` and `response`. You can use any `Datamodule` supported by LitGPT.

### Launch training

```bash
huggingface-cli download "Qwen/Qwen1.5-1.8B" --local-dir Qwen1.5-1.8B --local-dir-use-symlinks False

MODEL_DIR=Qwen1.5-1.8B
DATA_PATH=sft_data.parquet # columns: `prompt` and `response`

fabric run model \
--accelerator=cuda \
--devices=8 \
launch/finetune.py \
--checkpoint_dir $MODEL_DIR \
--data ParquetData \
--data.data_path $DATA_PATH \
--train.learning_rate=1e-5 \
--train.lr_warmup_steps=100 \
--train.micro_batch_size=16 \
--train.epochs=1 \
--train.save_interval=10000 \
--train.global_batch_size=64 \
--train.log_interval=1 \
--out_dir out
```


### Convert to Huggingface
The model's state dict is not changed during training, it is simply extracted:
```bash
python3 litgpt/scripts/convert_hf_fast.py out/final/lit_model.pth/checkpoint/mp_rank_00_model_states.pt Qwen1.5-1.8B-finetuned
```
The saved model file is `Qwen1.5-1.8B-finetuned/pytorch_model.bin`.
You may need to copy some metadata to `Qwen1.5-1.8B-finetuned` to load the model directly via `AutoModelForCausalLM`.

## Pretrain

### Data
For pretraining, LitHFT uses dataloader from [TinyLLama](https://github.com/jzhang38/TinyLlama) (with some modifications). Use the following script to process all parquet files under `pretrain_data` into binary data. All parquet files should contain the column `text`.


```bash
huggingface-cli download "Qwen/Qwen1.5-1.8B" --local-dir Qwen1.5-1.8B --local-dir-use-symlinks False

python scripts/prepare_packed_data.py --source_path pretrain_data --destination_path bin/pretrain_data --tokenizer_path Qwen1.5-1.8B --prefix data_part
```

### Launch distributed training

For rank=0 use the following command to start, for the other nodes you should modify the value of `NODE_RANK`.

```bash
NODE_RANK=0
MODEL_DIR=Qwen1.5-1.8B
DATA_PATH=bin/pretrain_data

fabric run model \
--node-rank=$NODE_RANK \
--main-address=$RANK0_ADDR \
--main-port=$RANK0_PORT \
--accelerator=cuda \
--devices=8 \
--num-nodes=2 \
launch/pretrain.py \
--model_config $MODEL_DIR \
--data PackedData \
--data.data_path $DATA_PATH \
--data.shuffle False \
--data.file_prefixes data_part \
--train.learning_rate 4e-4 \
--train.lr_warmup_steps=200 \
--train.micro_batch_size=3 \
--train.max_tokens=1000000000000 \
--train.save_interval=10000 \
--train.log_interval=1 \
--zero3=False \
--out_dir out
```

### Convert to Huggingface

Same as the [finetuning section](#convert-to-huggingface).

### Continue training

Continue training can be performed by loading a pre-trained model, specifying the model directory by `initial_checkpoint_dir` flag:


```bash
NODE_RANK=0
MODEL_DIR=Qwen1.5-1.8B
DATA_PATH=bin/pretrain_data

fabric run model \
--node-rank=$NODE_RANK \
--main-address=$RANK0_ADDR \
--main-port=$RANK0_PORT \
--accelerator=cuda \
--devices=8 \
--num-nodes=2 \
launch/pretrain.py \
--initial_checkpoint_dir $MODEL_DIR \
--data PackedData \
--data.data_path $DATA_PATH \
--data.shuffle False \
--data.file_prefixes data_part \
--train.learning_rate 4e-4 \
--train.lr_warmup_steps=200 \
--train.micro_batch_size=3 \
--train.max_tokens=1000000000000 \
--train.save_interval=10000 \
--train.log_interval=1 \
--zero3=False \
--out_dir out
```



## License

LitHFT is released under the [MIT](https://github.com/DoubleVII/lithft/blob/main/LICENSE) license. Additionally, this project comes with LitGPT’s [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.