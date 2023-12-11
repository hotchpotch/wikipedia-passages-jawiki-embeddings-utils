from __future__ import annotations

import streamlit as st
import os

import faiss
from sentence_transformers import SentenceTransformer
import torch
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

E5_QUERY_TYPES = [
    "passage",
    "query",
]

# for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model(name: str, max_seq_length=512):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_seq_length
    return model


def get_wikija_ds(name: str = WIKIPEDIA_JS_DS_NAME):
    ds = load_dataset(path=WIKIPEDIA_JA_DS, name=name, split="train")
    return ds


def get_faiss_index(
    index_name: str, ja_emb_ds: str = WIKIPEDIA_JA_EMB_DS, name=WIKIPEDIA_JS_DS_NAME
):
    target_path = f"faiss_indexes/{name}/{index_name}"
    dm = DownloadManager()
    index_local_path = dm.download(
        f"https://huggingface.co/datasets/{ja_emb_ds}/resolve/main/{target_path}"
    )
    index = faiss.read_index(index_local_path)
    index.nprobe = 256
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
