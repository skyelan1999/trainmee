# trainee

a local ai for xiaomi

## TrainMee：Mac 上微调 Qwen2.5-3B-Instruct

TrainMee 是面向 Apple Silicon 的本地增量微调目录。它使用 MLX-LM 在 macOS 原生环境里训练 LoRA，并通过兼容 OpenAI Chat Completions 的 HTTP API 提供模型给 OpenWebUI。API 代理将对外模型 ID 固定为 `qwen25-trainmee`。训练不在 Docker 容器内运行，因为这条 Mac GPU 路线使用 Apple MLX/Metal，不使用 CUDA。代码和样例数据位于此目录；Python 环境、模型缓存、adapter、合并模型和日志默认放在 `~/Library/Application Support/TrainMee/`，避免大文件进入 iCloud 同步。

## 适用环境

- Apple Silicon；当前机器为 M5、24GB 统一内存。
- macOS 原生 Python 3.12，Apple Silicon。
- 建议先关闭占用大量内存的应用。首次下载模型需要网络和数 GB 磁盘空间。

## 1. 安装环境

克隆仓库后，从项目目录创建本机环境：

```bash
cd trainee
STATE="$HOME/Library/Application Support/TrainMee"
mkdir -p "$STATE"
python3.12 -m venv "$STATE/venv"
source "$STATE/venv/bin/activate"
export HF_HOME="$STATE/huggingface"
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

需要 Python 3.12。不要用系统 Python 3.9 创建环境。

## 2. 准备训练数据

编辑 `data/train.jsonl` 和 `data/valid.jsonl`。每行是一条完整 JSON 记录，建议先人工审查其他模型生成的答案，并保留原问题、必要的上下文和最终目标回答：

```json
{"messages":[{"role":"user","content":"原始问题及必要上下文"},{"role":"assistant","content":"筛选、纠正后的目标回答"}]}
```

多轮对话可以在 `messages` 中依次放入 `user`、`assistant` 消息，最后一条必须是目标 `assistant` 回答。JSONL 要求每条记录单独占一行。不要将未经筛选的模型输出直接作为正确答案；重复、矛盾、错误或格式差的样本会被模型学进去。

仓库附带的数据只是格式示例，**不是你的有效训练数据**。开始正式训练前请替换成自己的样本。将约 5%–10% 的样本独立放在 `valid.jsonl`，不要同时复制到训练集。

检查格式：

```bash
"$HOME/Library/Application Support/TrainMee/venv/bin/python" scripts/check_dataset.py
```

## 3. 运行 LoRA 微调

```bash
./scripts/train.command
```

默认跑 100 次迭代作冒烟式首轮训练，batch size 为 1、仅训练末端 8 层、最大序列长度 2048，并启用梯度检查点和 prompt masking，以控制 24GB 统一内存占用。训练日志写入 `~/Library/Application Support/TrainMee/logs/`，adapter 写入 `~/Library/Application Support/TrainMee/adapters/qwen25-3b-lora/`。旧 adapter 存在时脚本会停止，避免覆盖。要调整迭代次数，可运行：

```bash
TRAINMEE_ITERS=300 ./scripts/train.command
```

后续把新一批审核后的问答追加到 `train.jsonl`、更新验证样本后，可从既有 adapter 继续训练：

```bash
TRAINMEE_RESUME_ADAPTER="$HOME/Library/Application Support/TrainMee/adapters/qwen25-3b-lora/adapters.safetensors" ./scripts/train.command
```

继续训练前建议先复制备份整个 adapter 目录。每轮训练都使用同一个 Qwen2.5-3B 基座，并保留此前 adapter 文件；不要把不同基座模型的 adapter 混用。当前工作流由你手工筛选、整理模型问答后启动训练，不会自动读取 OpenWebUI 对话或未经审核地自我训练。

这不是固定的最终超参数。观察日志中的 loss 和验证表现，再决定是否增加数据或迭代次数；只看训练 loss 下降不能证明模型回答质量提高。

## 4. 合并 adapter 并启动 API

先合并 adapter，生成可独立加载的 MLX 模型目录：

```bash
./scripts/fuse.command
```

还没有训练模型时，也可以直接启动基座模型 API；训练并 fuse 后再次启动同一命令，会自动改为提供合并后的微调模型。启动时模型权重会自动下载到 `~/Library/Application Support/TrainMee/huggingface/`。保持服务终端运行：

```bash
./scripts/serve.command
```

默认监听 `127.0.0.1:8080`。先查出服务公布的模型 ID，再用 curl 检查 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen25-trainmee","messages":[{"role":"user","content":"你好，请简要介绍你自己。"}],"max_tokens":120}'
```

合并后的服务路径是刻意采用的：MLX-LM 原生服务支持 OpenAI 风格 API；先 fuse 可避免服务端直接加载 adapter 时受版本行为影响。

## 5. 配置 OpenWebUI

在 OpenWebUI 的管理员设置中添加 OpenAI 连接：

- API Base URL：`http://host.docker.internal:8080/v1`（OpenWebUI 在 Mac 上的 Docker 容器内运行时；需先用下面的命令改为监听所有本机网卡）
- API Key：填任意占位值，例如 `trainmee-local`（本地 MLX 服务默认不校验此值）
- 模型：选择 `/v1/models` 返回的模型 ID

如果 OpenWebUI 直接运行在同一台 Mac 上，保持默认监听地址，Base URL 用 `http://127.0.0.1:8080/v1`。若 OpenWebUI 在 Mac 的 Docker 容器里，先停止默认服务，再用下面的命令启动：

```bash
TRAINMEE_HOST=0.0.0.0 ./scripts/serve.command
```

MLX-LM 的服务默认没有 API 鉴权。只在可信的本机/私有网络使用 `0.0.0.0`，并按需配置 macOS 防火墙；不要把无认证的推理 API 暴露到公共网络。

当前 API 的连接信息（基座模型版本）：

- OpenWebUI 在 Mac 的 Docker 容器中：`http://host.docker.internal:8080/v1`
- OpenWebUI 直接运行在这台 Mac：`http://127.0.0.1:8080/v1`
- 模型 ID：`qwen25-trainmee`
- API Key：可填 `trainmee-local` 作为占位值；本地服务目前不校验 Key

停止 API：在另一个终端运行 `./scripts/stop.command`。如果服务在当前终端前台运行，也可以在该终端按 `Ctrl+C`。

## 常用操作

```bash
# 停止服务：在服务终端按 Ctrl+C
# 查看训练日志
ls -lt "$HOME/Library/Application Support/TrainMee/logs"
# 重新训练前备份或移走旧 adapter，避免误覆盖
mv "$HOME/Library/Application Support/TrainMee/adapters/qwen25-3b-lora" "$HOME/Library/Application Support/TrainMee/adapters/qwen25-3b-lora.backup"
# 指定本地 MLX 格式基础模型服务
TRAINMEE_MODEL="/path/to/mlx-model" ./scripts/serve.command
```

## 目录说明

```text
trainmee/
├── README.md
├── requirements.txt
├── requirements.lock.txt
├── data/                 # train.jsonl、valid.jsonl
├── scripts/
│   ├── check_dataset.py
│   ├── train.command
│   ├── fuse.command
│   ├── serve.command
│   ├── api_proxy.py
│   └── stop.command
```

本机运行状态目录 `~/Library/Application Support/TrainMee/`：`venv/`、`huggingface/`、`adapters/`、`models/`、`logs/`。

模型来源及许可证请查看 [Qwen2.5-3B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。MLX-LM 的训练、数据和 fuse 参数说明见 [官方 LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)，API 说明见 [官方 Server 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)。
