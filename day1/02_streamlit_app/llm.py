# llm.py
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            batch_size=4, # バッチサイズを設定 (調整可能)
            num_workers=2 # データローダーのワーカー数 (調整可能)
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して単一の質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        messages = [
            {"role": "user", "content": user_question},
        ]
        outputs = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

        assistant_response = ""
        if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
            if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
                last_message = outputs[0]["generated_text"][-1]
                if last_message.get("role") == "assistant":
                    assistant_response = last_message.get("content", "").strip()
            elif isinstance(outputs[0]["generated_text"], str):
                full_text = outputs[0]["generated_text"]
                prompt_end = user_question
                response_start_index = full_text.find(prompt_end) + len(prompt_end)
                possible_response = full_text[response_start_index:].strip()
                if "<start_of_turn>model" in possible_response:
                    assistant_response = possible_response.split("<start_of_turn>model\n")[-1].strip()
                else:
                    assistant_response = possible_response

        if not assistant_response:
            print("Warning: Could not extract assistant response. Full output:", outputs)
            assistant_response = "回答の抽出に失敗しました。"

        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s") # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0
    


def generate_batch_responses(pipe, user_questions):
    """LLMを使用して複数の質問に対する回答をまとめて生成する"""
    if pipe is None:
        return ["モデルがロードされていないため、回答を生成できません。" for _ in user_questions], [0.0] * len(user_questions)

    try:
        start_time = time.time()
        batch_inputs = [{"role": "user", "content": q} for q in user_questions]
        outputs = pipe(batch_inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        end_time = time.time()
        total_response_time = end_time - start_time
        average_response_time = total_response_time / len(user_questions) if user_questions else 0

        responses = []
        for output in outputs:
            assistant_response = ""
            if output and isinstance(output, list) and output[0].get("generated_text"):
                if isinstance(output[0]["generated_text"], list) and len(output[0]["generated_text"]) > 0:
                    last_message = output[0]["generated_text"][-1]
                    if last_message.get("role") == "assistant":
                        assistant_response = last_message.get("content", "").strip()
                elif isinstance(output[0]["generated_text"], str):
                    full_text = output[0]["generated_text"]
                    assistant_response = full_text.strip() # より高度な抽出が必要な場合あり

            if not assistant_response:
                print("Warning: Could not extract assistant response from batch output:", output)
                assistant_response = "回答の抽出に失敗しました。"
            responses.append(assistant_response)

        print(f"Generated {len(responses)} responses in {total_response_time:.2f}s (average {average_response_time:.2f}s)")
        return responses, [average_response_time] * len(responses) # 各応答に平均時間を返す

    except Exception as e:
        st.error(f"バッチ回答生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return [f"エラーが発生しました: {str(e)}" for _ in user_questions], [0.0] * len(user_questions)
