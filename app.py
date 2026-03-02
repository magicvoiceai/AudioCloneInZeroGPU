# coding=utf-8
import os
import sys
import logging
import spaces
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import functools
import uuid
import random
import whisper
import librosa
from opencc import OpenCC

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Qwen3-TTS-Demo")

# 初始化简繁转换器
cc = OpenCC('t2s')

HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)

MODEL_SIZES = ["0.6B", "1.7B"]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_path(model_type: str, model_size: str) -> str:
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

@functools.lru_cache(maxsize=1)
def load_model(model_type, model_size):
    path = get_model_path(model_type, model_size)
    return Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda",
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        attn_implementation="kernels-community/flash-attn3"
    )

@functools.lru_cache(maxsize=1)
def load_whisper_model(model_name="large-v3"):
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max)
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio):
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

@spaces.GPU
def infer_voice_design(part, language, voice_description):
    voice_design_model = load_model("VoiceDesign","1.7B")
    seed_everything(42)
    wavs, sr = voice_design_model.generate_voice_design(
        text=part,
        language=language,
        instruct=voice_description.strip(),
        non_streaming_mode=True,
        max_new_tokens=2048,
    )
    return wavs[0], sr

@spaces.GPU
def infer_voice_clone(part, language, audio_tuple, ref_text, use_xvector_only):
    tts = load_model("Base", "0.6B")
    voice_clone_prompt = tts.create_voice_clone_prompt(
        ref_audio=audio_tuple,
        ref_text=ref_text.strip() if ref_text else None,
        x_vector_only_mode=use_xvector_only
    )
    wavs, sr = tts.generate_voice_clone(
        text=part,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=2048,
        seed=42, 
        temperature=0.3,
        top_p=0.85
    )
    return wavs[0], sr

@spaces.GPU
def infer_voice_clone_from_prompt(part, language, prompt_file_path):
    loaded_data = torch.load(prompt_file_path, map_location='cuda', weights_only=False)
    if isinstance(loaded_data, list) and len(loaded_data) > 0 and isinstance(loaded_data[0], VoiceClonePromptItem):
        voice_clone_prompt = loaded_data
    elif isinstance(loaded_data, list) and len(loaded_data) > 0 and isinstance(loaded_data[0], dict):
        voice_clone_prompt = [VoiceClonePromptItem(**item) for item in loaded_data]
    else:
         voice_clone_prompt = loaded_data
    if isinstance(voice_clone_prompt, list):
        for item in voice_clone_prompt:
            if item.ref_code is not None and item.ref_code.ndim == 3:
                item.ref_code = item.ref_code.squeeze(0)
    tts = load_model("Base", "0.6B")
    wavs, sr = tts.generate_voice_clone(
        text=part,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=2048,
        seed=42, 
        temperature=0.3,
        top_p=0.85
    )
    return wavs[0], sr

@spaces.GPU
def extract_voice_clone_prompt(ref_audio, ref_text, use_xvector_only):
    tts = load_model("Base", "0.6B")
    seed_everything(42)
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "错误：需要参考音频。"
    r_text = ref_text
    uxo = use_xvector_only
    if not r_text or (isinstance(r_text, str) and not r_text.strip()):
        whisper_size = "base"
        try:
            whisper_model = load_whisper_model(whisper_size)
            audio_data, sr = audio_tuple
            if sr != 16000:
                whisper_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            else:
                whisper_audio = audio_data
            result = whisper_model.transcribe(whisper_audio)
            
            res_val = result.get("text", "")
            if isinstance(res_val, list) and len(res_val) > 0:
                res_val = res_val[0]
            if not isinstance(res_val, str):
                res_val = str(res_val)
            r_text = cc.convert(res_val.strip())
            uxo = False
        except Exception as e:
            logger.error(f"Whisper 识别失败: {str(e)}", exc_info=True)
            uxo = True
            # return None, f"错误：语音识别失败且未提供参考文本。{str(e)}"
    
    r_text_str = ""
    if isinstance(r_text, str):
        r_text_str = r_text.strip()
    elif isinstance(r_text, list) and len(r_text) > 0 and isinstance(r_text[0], str):
        r_text_str = r_text[0].strip()
        
    logger.info(f"语音识别成功 ：{r_text_str}")
    voice_clone_prompt_items = tts.create_voice_clone_prompt(
        ref_audio=audio_tuple,
        ref_text=r_text_str if r_text_str else None,
        x_vector_only_mode=uxo
    )
    prompt_data = []
    for item in voice_clone_prompt_items:
        prompt_data.append({
            "ref_code": item.ref_code,
            "ref_spk_embedding": item.ref_spk_embedding,
            "x_vector_only_mode": item.x_vector_only_mode,
            "icl_mode": item.icl_mode,
            "ref_text": item.ref_text
        })
    file_id = str(uuid.uuid4())[:8]
    file_path = f"voice_clone_prompt_{file_id}.pt"
    torch.save(prompt_data, file_path)
    return file_path

def generate_voice_design(text, language, voice_description):
    if not text or not text.strip():
        return None, "错误：文本不能为空。"
    if not voice_description or not voice_description.strip():
        return None, "错误：语音描述不能为空。"
    try:
        wav, sr = infer_voice_design(text.strip(), language, voice_description)
        return (sr, wav), "语音设计生成成功！"
    except Exception as e:
        logger.error(f"Voice Design 生成失败: {str(e)}", exc_info=True)
        return None, f"错误: {e}"

def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only):
    t_text = target_text.strip() if isinstance(target_text, str) else ""
    if not t_text:
        return None, "错误：目标文本不能为空。"
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "错误：需要参考音频。"
    r_text = ref_text.strip() if isinstance(ref_text, str) else ""
    if not use_xvector_only and not r_text:
        return None, "错误：未启用 '仅使用 x-vector' 时需要参考文本。"
    try:
        wav, sr = infer_voice_clone(t_text, language, audio_tuple, r_text, use_xvector_only)
        return (sr, wav), "语音克隆生成成功！"
    except Exception as e:
        logger.error(f"Voice Clone 生成失败: {str(e)}", exc_info=True)
        return None, f"错误: {e}"

def generate_voice_clone_from_prompt_file(prompt_file_path, target_text, language):
    t_text = target_text.strip() if isinstance(target_text, str) else ""
    if not t_text:
        return None, "错误：目标文本不能为空。"
    if not prompt_file_path:
        return None, "错误：需要提供音频特征文件。"
    try:
        wav, sr = infer_voice_clone_from_prompt(t_text, language, prompt_file_path)
        return (sr, wav), "语音克隆生成成功（使用特征文件）！"
    except Exception as e:
        logger.error(f"Voice Clone 生成失败: {str(e)}", exc_info=True)
        return None, f"错误: {e}"

@spaces.GPU
def infer_whisper_audio(audio_path, model_size="base"):
    if not audio_path:
        return "错误：请上传音频文件或进行录音。"
    try:
        model = load_whisper_model(model_size)
        result = model.transcribe(audio_path)
        
        res_val = result.get("text", "")
        if isinstance(res_val, list) and len(res_val) > 0:
            res_val = res_val[0]
        if not isinstance(res_val, str):
            res_val = str(res_val)
        
        return cc.convert(res_val.strip())
    except Exception as e:
        logger.error(f"Whisper 识别失败: {str(e)}", exc_info=True)
        return f"识别出错: {e}"

def build_ui():
    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    with gr.Blocks(theme=theme, title="Qwen3-TTS Demo") as demo:
        gr.Markdown("# Qwen3-TTS Demo")
        with gr.Tabs():
            with gr.Tab("ASR (Whisper)"):
                with gr.Row():
                    with gr.Column():
                        asr_audio_input = gr.Audio(label="输入音频", type="filepath", sources=["microphone", "upload"])
                        asr_model_size = gr.Dropdown(label="Whisper 模型大小", choices=["base", "small", "medium", "large-v3"], value="base")
                        asr_btn = gr.Button("开始识别", variant="primary")
                    with gr.Column():
                        asr_text_output = gr.Textbox(label="识别结果", lines=10, show_copy_button=True)
                asr_btn.click(infer_whisper_audio, inputs=[asr_audio_input, asr_model_size], outputs=[asr_text_output])
            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column():
                        design_text = gr.Textbox(label="目标文本", lines=4, value="It's in the top drawer... wait, it's empty?")
                        design_language = gr.Dropdown(label="语言", choices=LANGUAGES, value="Auto")
                        design_instruct = gr.Textbox(label="语音描述", lines=3, value="Speak in an incredulous tone.")
                        design_btn = gr.Button("开始生成", variant="primary")
                    with gr.Column():
                        design_audio_out = gr.Audio(label="生成音频", type="numpy")
                        design_status = gr.Textbox(label="状态", interactive=False)
                design_btn.click(generate_voice_design, inputs=[design_text, design_language, design_instruct], outputs=[design_audio_out, design_status],api_name="generate_voice_design")
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### 1. 提取音频特征")
                with gr.Row():
                    with gr.Column():
                        extract_ref_audio = gr.Audio(label="参考音频", type="numpy")
                        extract_ref_text = gr.Textbox(label="参考文本", lines=2)
                        extract_xvector = gr.Checkbox(label="仅使用 x-vector", value=False)
                        extract_btn = gr.Button("提取音频特征", variant="primary")
                    with gr.Column():
                        extract_file_out = gr.File(label="特征文件 (.pt)")
                extract_btn.click(extract_voice_clone_prompt, inputs=[extract_ref_audio, extract_ref_text, extract_xvector], outputs=[extract_file_out],api_name="extract_voice_clone_prompt")
                gr.Markdown("### 2. 使用特征文件生成")
                with gr.Row():
                    with gr.Column():
                        prompt_file = gr.File(label="特征文件 (.pt)")
                        prompt_target_text = gr.Textbox(label="目标文本", lines=4)
                        prompt_language = gr.Dropdown(label="语言", choices=LANGUAGES, value="Auto")
                        prompt_btn = gr.Button("使用特征文件生成", variant="primary")
                    with gr.Column():
                        prompt_audio_out = gr.Audio(label="生成音频", type="numpy")
                        prompt_status = gr.Textbox(label="状态", interactive=False)
                prompt_btn.click(generate_voice_clone_from_prompt_file, inputs=[prompt_file, prompt_target_text, prompt_language], outputs=[prompt_audio_out, prompt_status],api_name="generate_voice_clone_from_prompt")
                gr.Markdown("---")

                # Section 3: Traditional Voice Clone (Original)
                gr.Markdown("### 3. 传统音色克隆（直接使用参考音频）")
                gr.Markdown("直接上传参考音频生成语音（每次都需要提取特征）。")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="参考音频",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="参考文本",
                            lines=2,
                            placeholder="输入参考音频中的确切文字...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="仅使用 x-vector",
                            value=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="目标文本",
                            lines=4,
                            placeholder="输入要让克隆音色说话的文字...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="语言",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                        clone_btn = gr.Button("克隆并生成", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="生成的音频", type="numpy")
                    clone_status = gr.Textbox(label="状态", lines=2, interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                    outputs=[clone_audio_out, clone_status],
                    api_name="generate_voice_clone"
                )

    return demo

if __name__ == "__main__":
    build_ui().launch()
