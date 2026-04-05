

import sys

sys.path.append("/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main")  # 把项目根目录 GLM-4-Voice-main 加入路径


import os
import json
import re
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from TTS.api import TTS

# ----------------------------
# Utilities & deterministic
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量启用确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_into_sentences(text: str) -> List[str]:
    """
    按常见句尾标点分句（保留句尾标点），返回句子列表；
    若文本为空或没有句尾标点则返回整段作为单句。
    """
    text = (text or "").strip()
    if not text:
        return []
    # 保留 !?. 后的空格作为切分点
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]

# ----------------------------
# TTS (load once)
# ----------------------------
def get_tts_model(model_name: str = "tts_models/en/vctk/vits"):
    os.environ["PHONEMIZER_ESPEAK_PATH"] = "py-espeak-ng"
    global _TTS_MODEL
    if "_TTS_MODEL" not in globals():
        print("🔊 Loading TTS model:", model_name)
        _TTS_MODEL = TTS(model_name)
    return _TTS_MODEL

def text_to_audio_file(text: str, out_path: str, speaker: str = "p347"):
    tts = get_tts_model()
    # 固定随机性参数以尽量复现
    tts.tts_to_file(
        text=text,
        speaker=speaker,
        file_path=out_path,
        speed=1.0,
        noise_scale=0.0,
        noise_scale_w=0.0,
        length_scale=1.0
    )

# ----------------------------
# Whisper encoder (load once)
# ----------------------------
def load_whisper_encoder(tokenizer_path: str, device: str = "cuda"):
    print("🔁 Loading WhisperVQEncoder and WhisperFeatureExtractor from:", tokenizer_path)
    model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    return model, extractor

def audio_file_to_tokens(audio_path: str, whisper_model, feature_extractor) -> List[int]:
    tokens = extract_speech_token(whisper_model, feature_extractor, [audio_path])[0]
    return tokens

# ----------------------------
# Helpers for audio token text
# ----------------------------
def audio_tokens_to_str(tokens: List[int]) -> str:
    return "<|begin_of_audio|>" + "".join([f"<|audio_{t}|>" for t in tokens]) + "<|end_of_audio|>"

def tts_and_tokens_for_sentences(
    text: str,
    base_wav_prefix: str,
    whisper_model,
    feature_extractor,
    speaker: str = "p347",
    remove_temp_wavs: bool = False
) -> Tuple[List[int], List[str]]:
    """
    将文本分句，每句合成 wav 并提取 token，返回：
      - 全部 audio token 的拼接
      - 分句文本列表（便于后续拼接回 combined_text）
    """
    sentences = split_into_sentences(text)
    all_audio_tokens: List[int] = []
    sentence_texts: List[str] = []
    for s_idx, sentence in enumerate(sentences):
        wav_path = f"{base_wav_prefix}_sent{s_idx}.wav"
        try:
            text_to_audio_file(sentence, wav_path, speaker=speaker)
        except Exception as e:
            print(f"[error] TTS failed for sentence {s_idx}: {e}")
            audio_tokens = []
        else:
            try:
                audio_tokens = audio_file_to_tokens(wav_path, whisper_model, feature_extractor)
            except Exception as e:
                print(f"[error] audio->tokens failed for sentence {s_idx}: {e}")
                audio_tokens = []
        all_audio_tokens.extend(audio_tokens)
        sentence_texts.append(sentence)

        if remove_temp_wavs and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

    return all_audio_tokens, sentence_texts

def tts_and_tokens_for_whole(
    text: str,
    wav_path: str,
    whisper_model,
    feature_extractor,
    speaker: str = "p347",
    remove_temp_wavs: bool = False
) -> List[int]:
    """
    将整段文本直接 TTS 成单个 wav 并提取 token（用于“Instruction 转纯 audio”）
    """
    try:
        text_to_audio_file(text, wav_path, speaker=speaker)
    except Exception as e:
        print(f"[error] TTS failed for whole text: {e}")
        tokens = []
    else:
        try:
            tokens = audio_file_to_tokens(wav_path, whisper_model, feature_extractor)
        except Exception as e:
            print(f"[error] audio->tokens failed for whole text: {e}")
            tokens = []

    if remove_temp_wavs and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return tokens

# ----------------------------
# Interleave (global)
# ----------------------------
def interleave_text_audio_global(
    text: str,
    tokenizer,
    audio_token_ids: List[int],
    text_block_size: int = 13,
    audio_block_size: int = 26
) -> str:
    """
    全局交错文本 token 与音频 token：
    每 text_block_size 个 text token 插入 audio_block_size 个 audio token。
    最后一个 text block 插入剩余所有 audio token。
    """
    text_tokens = tokenizer.tokenize(text)
    total_text = len(text_tokens)
    interleaved_parts = []
    text_idx = 0
    audio_idx = 0
    total_audio = len(audio_token_ids)

    while text_idx < total_text:
        block_tokens = text_tokens[text_idx:text_idx + text_block_size]
        block_str = tokenizer.convert_tokens_to_string(block_tokens)
        interleaved_parts.append(block_str)
        text_idx += len(block_tokens)

        # 是否是最后一个 text block
        is_last_text_block = (text_idx >= total_text)
        if is_last_text_block:
            block_audio = audio_token_ids[audio_idx:]
        else:
            block_audio = audio_token_ids[audio_idx: audio_idx + audio_block_size]

        if block_audio:
            interleaved_parts.append(audio_tokens_to_str(block_audio))
            audio_idx += len(block_audio)

    # 如果文本为空但仍有音频 token（极少见）也追加音频
    if total_text == 0 and total_audio > 0:
        interleaved_parts.append(audio_tokens_to_str(audio_token_ids))

    return "".join(interleaved_parts)

# ----------------------------
# 旧：仅处理 {"text": "..."} 的数据集
# ----------------------------
def process_dataset(
    input_json_path: str,
    output_json_path: str,
    text_tokenizer_path: str,
    audio_tokenizer_path: str,
    temp_audio_dir: str,
    device: str = "cuda",
    speaker: str = "p347",
    text_block_size: int = 13,
    audio_block_size: int = 26,
    seed: int = 1234,
    remove_temp_wavs: bool = False
):
    os.makedirs(temp_audio_dir, exist_ok=True)
    set_seed(seed)
    # load text tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, trust_remote_code=True)
    # load whisper encoder & extractor once
    whisper_model, feature_extractor = load_whisper_encoder(audio_tokenizer_path, device=device)

    # load dataset
    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)


    processed_data = []
    first_sentence_records = []

    for idx, sample in tqdm(list(enumerate(dataset)), total=len(dataset), desc="Processing(text-only JSON)"):
        text = (sample.get("text") or "").strip()
        if not text:
            processed_data.append({"text": ""})
            first_sentence_records.append({
                "first_sentence_text": "",
                "first_sentence_audio_tokens": "<|begin_of_audio|><|end_of_audio|>"
            })
            continue

        # 1) 分句
        all_audio_tokens, sentence_texts = tts_and_tokens_for_sentences(
            text=text,
            base_wav_prefix=os.path.join(temp_audio_dir, f"sample_{idx}"),
            whisper_model=whisper_model,
            feature_extractor=feature_extractor,
            speaker=speaker,
            remove_temp_wavs=remove_temp_wavs
        )

        # 记录首句（可选）
        if sentence_texts:
            first_audio_str = audio_tokens_to_str(all_audio_tokens[:len(all_audio_tokens)])  # 简单保留全部
            first_sentence_records.append({
                "first_sentence_text": sentence_texts[0],
                "first_sentence_audio_tokens": first_audio_str
            })
        else:
            first_sentence_records.append({
                "first_sentence_text": "",
                "first_sentence_audio_tokens": "<|begin_of_audio|><|end_of_audio|>"
            })

        # 3) 拼回整段文本
        combined_text = " ".join(sentence_texts).strip()

        # 4) 全局交错
        interleaved_text = interleave_text_audio_global(
            combined_text,
            tokenizer,
            all_audio_tokens,
            text_block_size=text_block_size,
            audio_block_size=audio_block_size
        )
        processed_data.append({"text": interleaved_text})

    # 保存
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    first_audio_path = os.path.splitext(output_json_path)[0] + "_first_sentence_audio.json"
    with open(first_audio_path, "w", encoding="utf-8") as f:
        json.dump(first_sentence_records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Interleaved dataset saved to: {output_json_path}")
    print(f"✅ First-sentence audio tokens saved to: {first_audio_path}")

# ----------------------------
# 新：处理 esnli_test.json（Instruction & Response）
# ----------------------------
def process_esnli_dataset(
    input_json_path: str,
    output_json_path: str,
    text_tokenizer_path: str,
    audio_tokenizer_path: str,
    temp_audio_dir: str,
    device: str = "cuda",
    speaker: str = "p347",
    text_block_size: int = 13,
    audio_block_size: int = 26,
    seed: int = 1234,
    remove_temp_wavs: bool = False,
    # 版本 B 下 Response 的输出方式：默认仍用交替；如需纯音频，改为 "audio_only"
    response_b_mode: str = "interleave"
):
    """
    输入：形如
    [
      {"Instruction": "...", "Response": "..."},
      ...
    ]

    输出：每个输入样本扩增为两个样本：
      - 版本 A：{"variant":"A","source_id":i, "Instruction": <纯文本>, "Response": <13/26交替>}
      - 版本 B：{"variant":"B","source_id":i, "Instruction": <纯音频>, "Response": <13/26交替 或 纯音频(可选)>}
    """
    assert response_b_mode in ("interleave", "audio_only")

    os.makedirs(temp_audio_dir, exist_ok=True)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, trust_remote_code=True)
    whisper_model, feature_extractor = load_whisper_encoder(audio_tokenizer_path, device=device)

    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)


    dataset = dataset[:1000]
    # dataset = dataset[:5]

    out_records = []

    for idx, sample in tqdm(list(enumerate(dataset)), total=len(dataset), desc="Processing(ESNLI)"):
        instr = (sample.get("Instruction") or "").strip()
        resp  = (sample.get("Response") or "").strip()

        # ---------- 处理 Response（用于 A & B 共用，默认交替） ----------
        # 分句 -> TTS -> tokens
        resp_tokens, resp_sentence_texts = tts_and_tokens_for_sentences(
            text=resp,
            base_wav_prefix=os.path.join(temp_audio_dir, f"esnli_{idx}_resp"),
            whisper_model=whisper_model,
            feature_extractor=feature_extractor,
            speaker=speaker,
            remove_temp_wavs=remove_temp_wavs
        )
        resp_text_combined = " ".join(resp_sentence_texts).strip()

        # 交替输出（标准）
        resp_interleaved = interleave_text_audio_global(
            text=resp_text_combined,
            tokenizer=tokenizer,
            audio_token_ids=resp_tokens,
            text_block_size=text_block_size,
            audio_block_size=audio_block_size
        )
        # 纯音频输出（B 可选）
        # resp_audio_only = audio_tokens_to_str(resp_tokens)
        resp_audio_only = None

        # ---------- 版本 A：Instruction 纯文本；Response 交替 ----------
        rec_A = {
            "variant": "A",
            "source_id": idx,
            "Instruction": instr,
            "Response": resp_interleaved
        }
        out_records.append(rec_A)

        # ---------- 版本 B：Instruction 纯音频；Response 默认交替（或按开关纯音频） ----------
        instr_tokens_whole = tts_and_tokens_for_whole(
            text=instr,
            wav_path=os.path.join(temp_audio_dir, f"esnli_{idx}_instr.wav"),
            whisper_model=whisper_model,
            feature_extractor=feature_extractor,
            speaker=speaker,
            remove_temp_wavs=remove_temp_wavs
        )
        instr_audio_only = audio_tokens_to_str(instr_tokens_whole)

        rec_B = {
            "variant": "B",
            "source_id": idx,
            "Instruction": instr_audio_only,
            "Response": resp_interleaved if response_b_mode == "interleave" else resp_audio_only
        }
        out_records.append(rec_B)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. ESNLI processed dataset saved to: {output_json_path}")

# ----------------------------
# RUN 示例（请按需修改路径）
# ----------------------------
if __name__ == "__main__":
    # === 示例 1：处理 esnli_test.json（按你的新需求） ===
    esnli_input_json  = "/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/muon/data_process/cose/dataset/cose_train.json"
    esnli_output_json = "/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/muon/data_process/cose/dataset/cose_train_processed_2.json"
    text_tokenizer_path = "/home/huang341/myspace/GLM-VOICE/glm-9b-voice-model"
    audio_tokenizer_path = "/home/huang341/myspace/GLM-VOICE/encoder"
    temp_audio_dir = "/home/huang341/myspace/GLM_unlearn/GLM-4-Voice-main/eval/temp_outputs"

    set_seed(1234)
    process_esnli_dataset(
        input_json_path=esnli_input_json,
        output_json_path=esnli_output_json,
        text_tokenizer_path=text_tokenizer_path,
        audio_tokenizer_path=audio_tokenizer_path,
        temp_audio_dir=temp_audio_dir,
        device="cuda",
        speaker="p347",
        text_block_size=13,
        audio_block_size=26,
        seed=1234,
        remove_temp_wavs=False,
        # 如果想让 B 版本的 Response 也变成纯音频，这里改成 "audio_only"
        response_b_mode="interleave"
    )

    # === 示例 2：保留原有能力，处理 {"text": "..."} 的 JSON ===
    # input_json = "/home/huang341/myspace/GLM_unlearn/datasets/short_45_add_q.json"
    # output_json = "/home/huang341/myspace/GLM_unlearn/datasets/short_45_add_q_o3_il.json"
    # process_dataset(
    #     input_json_path=input_json,
    #     output_json_path=output_json,
    #     text_tokenizer_path=text_tokenizer_path,
    #     audio_tokenizer_path=audio_tokenizer_path,
    #     temp_audio_dir=temp_audio_dir,
    #     device="cuda",
    #     speaker="p347",
    #     text_block_size=13,
    #     audio_block_size=26,
    #     seed=1234,
    #     remove_temp_wavs=False
    # )


# 这个是对于training json