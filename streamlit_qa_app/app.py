"""
streamlit run app.py --server.address 0.0.0.0
"""

from __future__ import annotations

import streamlit as st
import os

import faiss
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI
import streamlit as st
import pandas as pd
import os
from time import time
from datasets.download import DownloadManager
from datasets import load_dataset  # type: ignore


WIKIPEDIA_JA_DS = "singletongue/wikipedia-utils"
WIKIPEDIA_JS_DS_NAME = "passages-c400-jawiki-20230403"
WIKIPEDIA_JA_EMB_DS = "hotchpotch/wikipedia-passages-jawiki-embeddings"

EMB_MODEL_PQ = {
    "intfloat/multilingual-e5-small": 96,
    "intfloat/multilingual-e5-base": 192,
    "intfloat/multilingual-e5-large": 256,
    "cl-nagoya/sup-simcse-ja-base": 192,
    "pkshatech/GLuCoSE-base-ja": 192,
}

EMB_MODEL_NAMES = list(EMB_MODEL_PQ.keys())

OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
]

E5_QUERY_TYPES = [
    "passage",
    "query",
]

DEFAULT_QA_PROMPT = """
## Instruction

Prepare an explanatory statement for the question, including as much detailed explanation as possible.
Avoid speculations or information not contained in the contexts. Heavily favor knowledge provided in the documents before falling back to baseline knowledge or other contexts. If searching the contexts didn"t yield any answer, just say that.

Responses must be given in Japanese.

## Contexts

{contexts}

## Question

{question}
""".strip()


if os.getenv("SPACE_ID"):
    USE_HF_SPACE = True
    os.environ["HF_HOME"] = "/data/.huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/data/.huggingface"
else:
    USE_HF_SPACE = False

# for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


@st.cache_resource
def get_model(name: str, max_seq_length=512):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_seq_length
    return model


@st.cache_resource
def get_wikija_ds(name: str = WIKIPEDIA_JS_DS_NAME):
    ds = load_dataset(path=WIKIPEDIA_JA_DS, name=name, split="train")
    return ds


@st.cache_resource
def get_faiss_index(
    index_name: str, ja_emb_ds: str = WIKIPEDIA_JA_EMB_DS, name=WIKIPEDIA_JS_DS_NAME
):
    target_path = f"faiss_indexes/{name}/{index_name}"
    dm = DownloadManager()
    index_local_path = dm.download(
        f"https://huggingface.co/datasets/{ja_emb_ds}/resolve/main/{target_path}"
    )
    index = faiss.read_index(index_local_path)
    index.nprobe = 128
    return index


def text_to_emb(model, text: str, prefix: str):
    return model.encode([prefix + text], normalize_embeddings=True)


def search(
    faiss_index, emb_model, ds, question: str, search_text_prefix: str, top_k: int
):
    start_time = time()
    emb = text_to_emb(emb_model, question, search_text_prefix)
    emb_exec_time = time() - start_time
    scores, indexes = faiss_index.search(emb, top_k)
    faiss_seartch_time = time() - emb_exec_time - start_time
    scores = scores[0]
    indexes = indexes[0]
    results = []
    for idx, score in zip(indexes, scores):  # type: ignore
        idx = int(idx)
        passage = ds[idx]
        results.append((score, passage))
    return results, emb_exec_time, faiss_seartch_time


def to_contexts(passages):
    contexts = ""
    for passage in passages:
        title = passage["title"]
        text = passage["text"]
        # section = passage["section"]
        contexts += f"- {title}: {text}\n"
    return contexts


def qa(
    openai_api_key: str,
    question: str,
    passages: list,
    model_name: str,
    temperature: int,
    qa_prompt: str,
    max_tokens=2000,
):
    client = OpenAI(api_key=openai_api_key)
    contexts = to_contexts(passages)
    prompt = qa_prompt.format(contexts=contexts, question=question)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        yield delta.content or ""


def generate_answer(
    openai_api_key,
    buf,
    question,
    passages,
    model_name,
    temperature,
    qa_prompt,
    max_tokens,
):
    buf.write("⏳回答の生成中...")
    texts = ""
    for char in qa(
        openai_api_key=openai_api_key,
        question=question,
        passages=passages,
        model_name=model_name,
        temperature=temperature,
        qa_prompt=qa_prompt,
        max_tokens=max_tokens,
    ):
        texts += char
        buf.write(texts)


def to_df(scores, passages):
    df = pd.DataFrame(passages)
    df["text"] = df["text"]
    df["score"] = scores
    df_rows = ["score", "title", "text", "section"]
    df = df[df_rows]
    return df


def app():
    st.title("Wikipedia 日本語 - RAGを使った検索Q&A")
    md_text = """
    [RAG用途に使える、Wikipedia 日本語の embeddings とベクトル検索用の faiss index を作った](https://secon.dev/entry/2023/12/04/080000-wikipedia-ja-embeddings/) の検索 & 質疑応答Q&Aのデモです。Wikipedia 2023年4月3日時点のデータを使用しています。
    """
    st.markdown(md_text)

    st.text_area(
        "Question",
        key="question",
        value="楽曲『約束はいらない』でデビューした、声優は誰?",
    )
    if not OPENAI_API_KEY:
        st.text_input(
            "OpenAI API Key",
            key="openai_api_key",
            type="password",
            placeholder="※ OpenAI API Key 未入力時は回答を生成せずに、検索のみ実行します",
        )
    else:
        st.session_state.openai_api_key = OPENAI_API_KEY

    with st.expander("オプション"):
        option_cols_main = st.columns(2)
        with option_cols_main[0]:
            st.selectbox("Emb Model", EMB_MODEL_NAMES, index=0, key="emb_model_name")
        with option_cols_main[1]:
            st.selectbox(
                "OpenAI Model", OPENAI_MODEL_NAMES, index=0, key="openai_model_name"
            )
        emb_model_name = st.session_state.emb_model_name
        option_cols_sub = st.columns(2)
        with option_cols_sub[0]:
            st.number_input("Top K", value=5, key="top_k", min_value=1, max_value=20)
        with option_cols_sub[1]:
            if "-e5-" in emb_model_name:
                st.radio(
                    "Passage or Query (e5 only)",
                    E5_QUERY_TYPES,
                    index=0,
                    key="e5_query_or_passage",
                    horizontal=True,
                )
                e5_query_or_passage = st.session_state.e5_query_or_passage
                index_emb_model_name = (
                    f"{emb_model_name.split('/')[-1]}-{e5_query_or_passage}"
                )
                search_text_prefix = f"{e5_query_or_passage}: "
            else:
                index_emb_model_name = emb_model_name.split("/")[-1]
                search_text_prefix = ""
        option_cols = st.columns(3)
        with option_cols[0]:
            st.slider("Temperature", 0.0, 1.0, value=0.8, key="temperature")
        with option_cols[1]:
            st.slider("nprobe", 16, 1024, value=128, key="nprobe")
        with option_cols[2]:
            st.number_input(
                "max_tokens", value=2000, key="max_tokens", min_value=1, max_value=16000
            )
        st.text_area("QA Prompt", value=DEFAULT_QA_PROMPT, key="qa_prompt")

    loading_placeholder = st.empty()
    loading_placeholder.text("⏳ Loading - Embedding Model...")
    emb_model = get_model(st.session_state.emb_model_name)
    loading_placeholder.text("⏳ Loading - Faiss Index...")
    emb_model_pq = EMB_MODEL_PQ[emb_model_name]
    index_name = f"{index_emb_model_name}/index_IVF2048_PQ{emb_model_pq}.faiss"
    faiss_index = get_faiss_index(index_name=index_name)
    faiss_index.nprobe = st.session_state.nprobe
    loading_placeholder.text("⏳ Loading - Huggingface Dataset...")
    ds = get_wikija_ds()
    loading_placeholder.empty()

    if st.button("Search"):
        answer_header = st.empty()
        answer_text_buffer = st.empty()

        question = st.session_state.question
        top_k = st.session_state.top_k
        scores = []
        passages = []
        search_results, emb_exec_time, faiss_seartch_time = search(
            faiss_index,
            emb_model,
            ds,
            question,
            search_text_prefix=search_text_prefix,
            top_k=top_k,
        )
        st.subheader("Search Results: ")
        st.write(
            f"⏱️ generate embedding: {emb_exec_time*1000:.2f}ms /  faiss search: {faiss_seartch_time*1000:.2f}ms"
        )
        for score, passage in search_results:
            scores.append(score)
            passages.append(passage)
        df = to_df(scores, passages)
        st.dataframe(df, hide_index=True)

        openai_api_key = st.session_state.openai_api_key
        if openai_api_key:
            openai_api_key = openai_api_key.strip()
            answer_header.subheader("Answer: ")
            openai_model_name = st.session_state.openai_model_name
            temperature = st.session_state.temperature
            qa_prompt = st.session_state.qa_prompt
            max_tokens = st.session_state.max_tokens
            generate_answer(
                openai_api_key=openai_api_key,
                buf=answer_text_buffer,
                question=question,
                passages=passages,
                model_name=openai_model_name,
                temperature=temperature,
                qa_prompt=qa_prompt,
                max_tokens=max_tokens,
            )


if __name__ == "__main__":
    app()
