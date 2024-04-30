<div align="center">

# ⚡️🤗LitHFT
**在你自己的数据上对任何来自huggingface的大型语言模型(LLMs)进行预训练和微调。**

| [English](https://github.com/DoubleVII/lithft) | [中文简体](docs/README_CN.md) |
</div>

## 什么是LitHFT

LitHFT是基于[LitGPT](https://github.com/Lightning-AI/litgpt)开发的工具，用于对[Huggingface](https://huggingface.co/)中的语言模型(LLMs)进行预训练和微调。相比之下，LitGPT专为某些类型的LLMs设计，支持20多种常用的LLMs。然而，它不适用于Huggingface中的其他类型模型，例如[Qwen](https://huggingface.co/Qwen)。LitHFT适用于原生Huggingface模型，无需进行检查点转换，但代价是训练优化较少。

|对比|LitHFT|LitGPT|
|----|----|----|
|LLMs|任何|20+|
|优化|Deepspeed|FSDP|
|数据加载器|来自[TinyLLama](https://github.com/jzhang38/TinyLlama)的打包数据|litdata|

## 安装LitHFT

```bash
git clone https://github.com/DoubleVII/lithft.git
cd lithft
pip install -e .
```

## 警告

该项目仅在Qwen和Mistral模型上测试过。

|模型|设备|吞吐量 / s / 设备|
|----|----|----|
|Qwen1.5-1.8B|A800-40G|24.5k tokens|
|Mistral-7B|A100-80G|3.5k tokens|

# 快速开始

## 微调

### 数据

这里我们使用[`ParquetData`](./litgpt/data/parquet_sft_data.py)作为示例，包含两列：`prompt`和`response`。你可以使用LitGPT支持的任何`Datamodule`。

### 启动训练

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

### 转换为Huggingface格式

训练期间模型的状态字典不会改变，简单提取即可：

```bash
python3 litgpt/scripts/convert_hf_fast.py out/final/lit_model.pth/checkpoint/mp_rank_00_model_states.pt Qwen1.5-1.8B-finetuned
```

保存的模型文件是`Qwen1.5-1.8B-finetuned/pytorch_model.bin`。

你可能需要复制一些元数据到`Qwen1.5-1.8B-finetuned`，以便通过`AutoModelForCausalLM`直接加载模型。

## 预训练

### 数据

对于预训练，LitHFT使用来自[TinyLLama](https://github.com/jzhang38/TinyLlama)（经过一些修改）的数据加载器。使用以下脚本将`pretrain_data`下的所有parquet文件处理为二进制数据。所有parquet文件应包含列`text`。

```bash
huggingface-cli download "Qwen/Qwen1.5-1.8B" --local-dir Qwen1.5-1.8B --local-dir-use-symlinks False

python scripts/prepare_packed_data.py --source_path pretrain_data --destination_path bin/pretrain_data --tokenizer_path Qwen1.5-1.8B --prefix data_part
```

### 启动分布式训练

对于`rank=0`的节点，使用以下命令启动，对于其他节点，你应该修改`NODE_RANK`的值。

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

### 转换为Huggingface格式

和[微调部分](#convert-to-huggingface)一样。

### 继续训练

通过加载预训练的模型，并通过`initial_checkpoint_dir`标志指定模型目录来执行继续训练：

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

## 许可证

LitHFT采用[MIT](https://github.com/DoubleVII/lithft/blob/main/LICENSE)许可证发布。此外，此项目还附带LitGPT的[Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE)许可证。