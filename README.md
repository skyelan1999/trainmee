# TrainMee

TrainMee 是面向 Apple Silicon 的本地 LoRA 增量训练与 OpenAI 兼容 API 项目。它使用 MLX-LM/Metal 在 macOS 原生运行，支持将人工审核过的其他模型输出整理为训练样本，再通过命令行测试和 OpenAI 兼容 API 使用微调后的模型。

## 模型和商业许可

默认模型为 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，模型卡标注 Apache-2.0。仓库不包含模型权重；初始化脚本会从 Hugging Face 下载并转换为 MLX 格式。请同时遵守模型仓库附带的许可和第三方组件声明。

API 默认对外模型 ID 为 `trainmee`，实际提供的基座为 Qwen3-4B。可通过 `TRAINMEE_MODEL_ID` 自定义此 ID。

项目默认把 Python 环境、Hugging Face 下载缓存、基础模型、adapter、合并模型和日志放在项目目录的 `state/`，其中运行生成物由 `.gitignore` 排除，不会推入 Git。仓库只提交代码、文档和训练数据样例。

## 适用环境

- Apple Silicon Mac；当前使用环境为 M5、24GB 统一内存。
- macOS、Python 3.12、Xcode Command Line Tools。
- 首次初始化需要网络，需为 Qwen3-4B 原始权重和 MLX 转换文件预留约 10GB 以上磁盘空间。

## 安装 Python 环境

```bash
cd trainmee
python3.12 -m venv state/venv
source state/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

## 下载并初始化模型

```bash
./scripts/init-model.command
```

脚本默认从 `Qwen/Qwen3-4B` 下载模型，缓存放在 `state/huggingface/`，转换结果放在 `state/base-model/`。若要指定同系列的另一个兼容 Hugging Face 模型，可设置 `TRAINMEE_HF_MODEL` 后运行。仓库不需要 Git LFS，也不上传模型镜像或权重。

## 准备训练数据

编辑 `data/train.jsonl` 和 `data/valid.jsonl`。每行是一条完整 JSON 记录，例如：

```json
{"messages":[{"role":"user","content":"问题及必要上下文"},{"role":"assistant","content":"筛选并确认后的目标回答"}]}
```

仓库附带的数据只是格式示例，不是有效训练集。建议将约 5%–10% 的样本独立保留用于验证。不要把未经审核的模型输出直接作为正确答案；错误、重复或互相矛盾的样本都会影响训练结果。

开始训练前可检查格式：

```bash
source state/venv/bin/activate
python scripts/check_dataset.py
```

## 训练 LoRA

先运行一次模型初始化，然后开始训练：

```bash
./scripts/init-model.command
./scripts/train.command
```

默认运行 100 次迭代、batch size 1、训练末端 8 层、最大序列长度 2048，并启用梯度检查点和 prompt masking，以控制 M5 24GB 统一内存的使用。增加迭代数：

```bash
TRAINMEE_ITERS=300 ./scripts/train.command
```

adapter 和训练日志保存在 `state/adapters/` 与 `state/logs/`。已有 adapter 时脚本会停止，避免覆盖。要继续训练，可设置：

```bash
TRAINMEE_RESUME_ADAPTER="$PWD/state/adapters/qwen3-4b-lora/adapters.safetensors" ./scripts/train.command
```

训练数据由你筛选、整理，项目不会自动读取其他聊天记录或自动吸收未审核回答。训练 loss 下降本身不能证明模型回答质量变好；应保留独立验证问题并人工比较训练前后的结果。当前仓库的 `data/train.jsonl` 只有 2 条示例、`data/valid.jsonl` 只有 1 条示例，适合验证流程，不代表有实际领域效果。正式使用前，请替换为你筛选过的真实问答数据。

## 终端聊天测试

启动多轮终端聊天器：

```bash
./scripts/chat.command
```

输入问题后会逐段显示模型回答。输入 `/clear` 清空当前对话，输入 `/exit` 退出。聊天器直接在本机加载已训练 adapter，不需要启动 API 或 WebUI。它优先用合并模型，其次用基础模型加 LoRA adapter；如果尚未训练，则用 Qwen3-4B 基座。

也可以用单条命令做一次问答：

```bash
./scripts/ask.command "请简要说明 LoRA 是什么。"
```

生成长度默认 512 tokens；可用 `TRAINMEE_MAX_TOKENS=1024 ./scripts/chat.command` 调整。

## 可选 OpenAI 兼容 API

需要让其他本机程序调用时，先合并 adapter 并启动 API：

```bash
./scripts/fuse.command
./scripts/serve.command
```

API 默认仅监听本机 `127.0.0.1:8080`，模型 ID 为 `trainmee`，路径为 `/v1`。本机连通性检查：

```bash
curl http://127.0.0.1:8080/v1/models
```

停止服务可运行 `./scripts/stop.command`，或在服务终端按 Ctrl+C。

## 目录结构

```text
trainmee/
├── data/                   # JSONL 训练/验证样例
├── scripts/                # 模型初始化、训练、聊天测试、合并与 API 脚本
├── state/                  # 本机环境、缓存、模型、adapter 和日志（Git 忽略）
├── requirements.lock.txt
└── README.md
```

参考：[Qwen3-4B 模型卡与许可证](https://huggingface.co/Qwen/Qwen3-4B)、[MLX-LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)、[MLX-LM Server 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)。
