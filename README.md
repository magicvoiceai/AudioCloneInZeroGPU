---
title: Qwen3-TTS Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: zero-a10g
---

# Qwen3-TTS AudioClone (AudioCloneInZeroGPU)

本项目是一个基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的交互式语音克隆与设计演示系统，专为 Hugging Face ZeroGPU 环境优化。它集成了先进的 ASR（语音识别）、Voice Design（语音设计）和 Voice Clone（语音克隆）技术。

## 🌟 主要功能

- **ASR (Whisper)**: 利用 OpenAI 的 Whisper 模型进行精准的语音到文本转换，支持多种模型规格（Base 到 Large-v3）。
- **Voice Design (语音设计)**: 只需通过一段文本描述（例如：“用一种惊讶的语气说话”），即可生成具有特定风格和音色的语音。
- **Voice Clone (语音克隆)**: 
    - **极速提取**: 能够从一段参考音频中快速提取音色特征。
    - **特征文件支持**: 支持保存特征文件（.pt），后续可直接使用该特征生成语音，无需重复提取。
    - **多语言支持**: 自动识别并支持中文、英文、日文、韩文等多种语言的克隆生成。

## 🛠️ 技术亮点

- **模型架构**: 基于 Qwen3-TTS-12Hz 系列模型（0.6B 和 1.7B 版本），提供高质量的语音合成。
- **ZeroGPU 优化**: 针对 Hugging Face ZeroGPU 进行了专门的内存与推理加速优化。
- **简繁转换**: 内置 OpenCC，支持自动将繁体中文识别结果转换为简体，提升用户体验。

## 🚀 快速开始

1. **环境准备**: 确保已安装 `requirements.txt` 中的所有依赖。
2. **运行 Demo**: 
   ```bash
   python app.py
   ```
3. **功能演示**:
   - 上传参考音频并输入目标文本，即可开始音色克隆。
   - 在语音设计板块，输入你的创意描述，让 AI 为你定制声音。

---

## 🔗 更多资源与高级服务

如果你对 AI 语音克隆技术感兴趣，或者需要更专业、更稳定、功能更全的在线语音服务，欢迎访问我们的官方平台：

👉 **[MagicVoice - 全球领先的 AI 语音克隆平台](https://magicvoice.online)**

在 MagicVoice，您可以体验到更极致的语音克隆精度、更丰富的音色库以及更便捷的商业化语音生成服务。

---

## 📄 开源协议

本项目遵循 Apache-2.0 开源协议。
