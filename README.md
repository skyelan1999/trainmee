# TrainMee

TrainMee 是面向 Apple Silicon 的本地 LoRA 增量训练与 OpenAI 兼容 API 项目。它使用 MLX-LM/Metal 在 macOS 原生运行，支持将人工审核过的其他模型输出整理为训练样本，再通过 OpenWebUI 调用微调后的模型。

## 模型和商业许可

默认模型为 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，模型卡标注 Apache-2.0。仓库不包含模型权重；初始化脚本会从 Hugging Face 下载并转换为 MLX 格式。请同时遵守模型仓库附带的许可和第三方组件声明。

API 对外模型 ID 暂保留为 `qwen25-trainmee`，以兼容之前配置 OpenWebUI 的模型名；它实际提供的基座已改为 Qwen3-4B。该 ID 可通过 `TRAINMEE_MODEL_ID` 修改。

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

训练数据由你筛选、整理，项目不会自动读取 OpenWebUI 历史或自动吸收未审核回答。训练 loss 下降本身不能证明模型回答质量变好；应保留独立验证问题并人工比较训练前后的结果。

## 合并并启动 API

训练后合并 adapter：

```bash
./scripts/fuse.command
```

启动 API：

```bash
./scripts/serve.command
```

默认监听 `127.0.0.1:8080`，OpenAI API 路径为 `/v1`。检查模型列表与对话接口：

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen25-trainmee","messages":[{"role":"user","content":"你好，请简要介绍你自己。"}],"max_tokens":120}'
```

OpenWebUI 添加 OpenAI 连接：

- OpenWebUI 在 Mac Docker 容器中：`http://host.docker.internal:8080/v1`
- OpenWebUI 直接运行在 Mac：`http://127.0.0.1:8080/v1`
- API Key：可填 `trainmee-local` 占位值（本地代理目前不校验）
- 模型 ID：`qwen25-trainmee`

如果 Docker 内的 OpenWebUI 无法连接，在 Mac 上停止默认服务后以 `TRAINMEE_HOST=0.0.0.0 ./scripts/serve.command` 启动。API 当前没有鉴权，只在可信网络使用，不要暴露到公共网络。停止服务运行 `./scripts/stop.command`，或在服务终端按 Ctrl+C。

## 目录结构

```text
trainmee/
├── data/                   # JSONL 训练/验证样例
├── scripts/                # 模型初始化、训练、合并、API 和停止脚本
├── state/                  # 本机环境、缓存、模型、adapter 和日志（Git 忽略）
├── requirements.lock.txt
└── README.md
```

参考：[Qwen3-4B 模型卡与许可证](https://huggingface.co/Qwen/Qwen3-4B)、[MLX-LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)、[MLX-LM Server 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)。
