# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()
elif st.session_state.page == "バッチチャット":
    if pipe:
        st.subheader("複数の質問を入力してください (1行に1質問)")
        batch_input = st.text_area("質問リスト", height=200)
    if st.button("まとめて送信"):
        if batch_input:
            questions = [q.strip() for q in batch_input.split('\n') if q.strip()]
        if questions:
            st.info(f"送信された質問数: {len(questions)}")
            with st.spinner("まとめて回答を生成中です..."):
                responses, response_times = llm.generate_batch_responses(pipe, questions)
                st.subheader("回答")
                for i, (question, response, response_time) in enumerate(zip(questions, responses, esponse_times)):
                    st.markdown(f"**質問 {i+1}:** {question}")
                    st.markdown(f"**回答 {i+1}:** {response}")
                    st.info(f"応答時間: {response_time:.2f}秒 (平均)")
                    st.markdown("---")
        else:
            st.warning("質問が入力されていません。")
    else:
        st.warning("質問を入力してください。")
else:
    st.error("バッチチャット機能を利用できません。モデルの読み込みに失敗しました。")

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")



# elif st.session_state.page == "バッチチャット":
#   if pipe:
#         st.subheader("複数の質問を入力してください (1行に1質問)")
#         batch_input = st.text_area("質問リスト", height=200)
#         if st.button("まとめて送信"):
# +            if batch_input:
# +                questions = [q.strip() for q in batch_input.split('\n') if q.strip()]
# +                if questions:
# 
# st.info(f"送信された質問数: {len(questions)}")
# 
# with st.spinner("まとめて回答を生成中です..."):
# 
#     responses, response_times = llm.generate_batch_responses(pipe, questions)
# 
#     st.subheader("回答")
# 
#     for i, (question, response, response_time) in enumerate(zip(questions, responses, response_times)):
# 
#         st.markdown(f"**質問 {i+1}:** {question}")
# 
#         st.markdown(f"**回答 {i+1}:** {response}")
# 
#         st.info(f"応答時間: {response_time:.2f}秒 (平均)")
# 
#         st.markdown("---")
# +                else:
# 
# st.warning("質問が入力されていません。")
# +            else:
# +                st.warning("質問を入力してください。")
# +    else:
# +        st.error("バッチチャット機能を利用できません。モデルの読み込みに失敗しました。")