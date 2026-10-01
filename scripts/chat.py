#!/usr/bin/env python3
"""Interactive local terminal chat for the TrainMee MLX model."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    state_root = Path(os.environ.get("TRAINMEE_STATE_DIR", root / "state"))
    fused_model = state_root / "models" / "qwen3-4b-trainmee"
    base_model = state_root / "base-model"
    adapter = state_root / "adapters" / "qwen3-4b-lora"

    if fused_model.is_dir():
        model_path = fused_model
        adapter_path = None
        model_label = "Qwen3-4B + LoRA"
    elif (base_model / "config.json").is_file():
        model_path = base_model
        adapter_path = adapter if (adapter / "adapters.safetensors").is_file() else None
        model_label = "Qwen3-4B + LoRA" if adapter_path else "Qwen3-4B 基座"
    else:
        print("模型尚未初始化。先运行 ./scripts/init-model.command", file=sys.stderr)
        return 1

    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        print(f"加载 MLX-LM 失败：{exc}\n请安装 state/venv 中的项目依赖。", file=sys.stderr)
        return 1

    print(f"正在加载 {model_label}…", flush=True)
    kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
    model, tokenizer = load(str(model_path), **kwargs)
    messages: list[dict[str, str]] = []
    max_tokens = int(os.environ.get("TRAINMEE_MAX_TOKENS", "512"))
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)

    print("TrainMee 本地聊天测试器（/clear 清空上下文，/exit 退出）")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not question:
            continue
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("已退出。")
            return 0
        if question.lower() == "/clear":
            messages.clear()
            print("已清空对话上下文。")
            continue

        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        print("TrainMee：", end="", flush=True)
        answer_parts: list[str] = []
        try:
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if response.text:
                    print(response.text, end="", flush=True)
                    answer_parts.append(response.text)
            print()
        except KeyboardInterrupt:
            print("\n（本轮生成已中断）")
        answer = "".join(answer_parts).strip()
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.pop()


if __name__ == "__main__":
    raise SystemExit(main())
