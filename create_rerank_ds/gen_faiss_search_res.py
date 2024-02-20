"""
faissで検索した結果をまず作成する
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import time

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset  # type: ignore
from datasets.download import DownloadManager
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

# from sentence_transformers import SentenceTransformer
PROJECT_ROOT = Path(__file__).parent.parent
RERANK_DS_WORK_DIR = PROJECT_ROOT / "workdir/rerank_ws"
RERANK_DS_WORK_DIR.mkdir(parents=True, exist_ok=True)


JAQKET_V1_DEV_URLS = [
    "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev1_questions.json",
    "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev2_questions.json",
]


WIKIPEDIA_JA_DS = "singletongue/wikipedia-utils"
WIKIPEDIA_JS_DS_NAME = "passages-c400-jawiki-20230403"
WIKIPEDIA_JA_EMB_DS = "hotchpotch/wikipedia-passages-jawiki-embeddings"

EMB_MODEL_PQ = {
    # "intfloat/multilingual-e5-small": 96,
    # "intfloat/multilingual-e5-base": 192,
    "intfloat/multilingual-e5-large": 256,
    "cl-nagoya/sup-simcse-ja-base": 192,
    "pkshatech/GLuCoSE-base-ja": 192,
    "text-embedding-3-small-dim512": 128,
}

EMB_MODEL_NAMES = list(EMB_MODEL_PQ.keys())

SEARCH_TOP_K = 500

# for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = ArgumentParser()
args.add_argument("-m", "--emb_model_name", type=str, default=None)
args.add_argument("-d", "--debug", action="store_true")
args.add_argument("-r", "--reranking", action="store_true")
args.add_argument("--use_gpu", action="store_true")

parsed_args = args.parse_args()


def get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_model(name: str, max_seq_length=512):
    if name.startswith("text-embedding"):
        # text-embedding は text-embedding-3-small-dim512 と後ろに dim がついている
        # -dim{n} を正規表現で取得する
        m = re.search(r"-dim(\d+)$", name)
        if m:
            target_dim = int(m.group(1))
        else:
            raise ValueError(f"invalid model name: {name}")
        # -dim{n} を削除する
        name = name.replace(f"-dim{target_dim}", "")
        model = OpenAIEmbeddings(
            model=name, tiktoken_model_name="cl100k_base", dimensions=target_dim
        )

        def model_to_embs_oai(texts: list[str]):
            # texts を1000個ずつに分割して、embs を取得する
            embs = []
            for i in range(0, len(texts), 1000):
                embs += model.embed_documents(texts[i : i + 1000])
            embs = model.embed_documents(texts)
            # to numpy
            embs = np.array(embs)
            return embs

        return model_to_embs_oai
    else:
        device = get_device_name()
        model = SentenceTransformer(name, device=device)
        model.max_seq_length = max_seq_length
        model.encode(["none"])  # warmup

        def model_to_embs(texts: list[str]):
            embs = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embs

        return model_to_embs


def get_wikija_ds(name: str = WIKIPEDIA_JS_DS_NAME):
    ds = load_dataset(path=WIKIPEDIA_JA_DS, name=name, split="train")  # type: ignore
    return ds


def get_faiss_index(
    index_name: str,
    ja_emb_ds: str = WIKIPEDIA_JA_EMB_DS,
    name=WIKIPEDIA_JS_DS_NAME,
    use_gpu=False,
):
    target_path = f"faiss_indexes/{name}/{index_name}"
    dm = DownloadManager()
    index_local_path = dm.download(
        f"https://huggingface.co/datasets/{ja_emb_ds}/resolve/main/{target_path}"
    )
    faiss_index = faiss.read_index(index_local_path)
    if use_gpu:  # and getattr(faiss, "StandardGpuResources", None):
        gpu_res = faiss.StandardGpuResources()  # type: ignore
        co = faiss.GpuClonerOptions()  # type: ignore
        # here we are using a over 64-byte PQ, so we must set the lookup tables to
        # 16 bit float (this is due to the limited temporary memory).
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index, co)  # type: ignore
    faiss_index.nprobe = 256
    return faiss_index


def texts_to_embs(model_to_embs_fn, texts: list[str], prefix: str):
    texts = [prefix + text for text in texts]
    start_time = time()
    embs = model_to_embs_fn(texts)
    end_time = time()
    return embs, end_time - start_time


def faiss_search_by_embs(faiss_index, embs, top_k=5):
    start_time = time()
    D, I = faiss_index.search(embs, top_k)
    end_time = time()
    search_sec = end_time - start_time
    return D, I, search_sec


# jaqket v1
@dataclass
class JaqketQuestionV1:
    qid: str
    question: str
    answer_entity: str
    label: int
    answer_candidates: list[str]
    original_question: str


def load_jaqket_v1(urls):
    res: list[JaqketQuestionV1] = []
    for url in urls:
        with urllib.request.urlopen(url) as f:
            # f は 1行ごとに処理
            data = [json.loads(line.decode("utf-8")) for line in f]
        for d in data:
            # label position
            d["label"] = d["answer_candidates"].index(d["answer_entity"])
            # if -1
            if d["label"] == -1:
                raise ValueError(
                    f"answer_entity not found in answer_candidates: {d['answer_entity']}, {d['answer_candidates']}"
                )
            res.append(
                JaqketQuestionV1(
                    qid=d["qid"],
                    question=d["question"],
                    answer_entity=d["answer_entity"],
                    label=d["label"],
                    answer_candidates=d["answer_candidates"],
                    original_question=d["original_question"],
                )
            )
    return res


def find_label_by_indexes(idxs, jaqket: JaqketQuestionV1, wiki_ds) -> int:
    INT_MAX = 2**31 - 1
    answer_candidates = jaqket.answer_candidates
    candidate_found_indexes = []
    titles = []
    texts = []
    for idx in idxs[0:top_k]:
        data = wiki_ds[idx]
        title = data["title"]
        text = data["text"]
        titles.append(title)
        texts.append(text)
    target_text = " ".join(titles + texts)
    for candidate in answer_candidates:
        pos = target_text.find(candidate)
        if pos == -1:
            candidate_found_indexes.append(INT_MAX)
        else:
            candidate_found_indexes.append(pos)
    min_index = min(candidate_found_indexes)
    if min_index == INT_MAX:
        return -1
    else:
        return candidate_found_indexes.index(min_index)


def predict_by_indexes(indexes, jaqket_ds, wiki_ds):
    pred_labels = []
    total = len(indexes)
    for idxs, jaqket in tqdm(zip(indexes, jaqket_ds), total=total):
        # tolist がある場合は実行する
        if hasattr(idxs, "tolist"):
            idxs = idxs.tolist()
        pred_label = find_label_by_indexes(idxs, jaqket, wiki_ds)
        pred_labels.append(pred_label)
    return pred_labels


print("load wikija datasets")
ds = get_wikija_ds()

jacket_v1 = load_jaqket_v1(JAQKET_V1_DEV_URLS)

if parsed_args.debug:
    print("RUN: debug mode")
    jacket_v1 = jacket_v1[:100]

if parsed_args.emb_model_name:
    target_emb_models = [parsed_args.emb_model_name]
else:
    target_emb_models = EMB_MODEL_NAMES

use_gpu = parsed_args.use_gpu

# rerankingは e5 でのみ実行
if parsed_args.reranking:
    print("reranking is e5 only")
    target_emb_models = [m for m in target_emb_models if "e5" in m]
    print("reranking target_emb_models: ", target_emb_models)

results = []

for emb_model_name in target_emb_models:
    if "-e5-" in emb_model_name:
        query_prefix = True
    else:
        query_prefix = False

    print("load model: ", emb_model_name)
    model_to_embs_fn = get_model(emb_model_name)

    name = f"{emb_model_name}"
    if parsed_args.debug:
        name = f"[debug]{name}"
    index_emb_model_name = f"{emb_model_name.split('/')[-1]}"
    if query_prefix:
        index_emb_model_name += f"-query"
        search_text_prefix = f"query: "
    else:
        # e5 以外のモデルは prefix なし
        search_text_prefix = ""
    emb_model_pq = EMB_MODEL_PQ[emb_model_name]
    print(f"--- {name} ---")

    print("load faiss index: ", index_emb_model_name)
    index_name = f"{index_emb_model_name}/index_IVF2048_PQ{emb_model_pq}.faiss"
    faiss_index = get_faiss_index(index_name, use_gpu=use_gpu)

    jaqket = jacket_v1
    question_embs, gen_embs_sec = texts_to_embs(
        model_to_embs_fn,
        texts=[q.question for q in jaqket],
        prefix=search_text_prefix,
    )
    gen_embs_sec = round(gen_embs_sec, 2)
    print("question_embs.shape: ", question_embs.shape)  # type: ignore
    print("gen embs sec: ", gen_embs_sec)
    scores, indexes, search_sec = faiss_search_by_embs(
        faiss_index, question_embs, top_k=SEARCH_TOP_K
    )
    # 一旦、検索結果を保存する
    results.append(
        {
            "name": name,
            "indexes": indexes,
            "scores": scores,
        }
    )
    del faiss_index
    torch.cuda.empty_cache()
    del model_to_embs_fn
    torch.cuda.empty_cache()

# save to pickle
import pickle

if parsed_args.debug:
    filename = f"search_results_debug.pkl.gz"
else:
    filename = f"search_results.pkl.gz"

with open(RERANK_DS_WORK_DIR / filename, "wb") as f:
    pickle.dump(results, f)

# # 最後に results を df にして、table で全て表示
# pd.set_option("display.max_rows", None)
# df = pd.DataFrame(results)
# print(df)

# # csv で保存
# if parsed_args.debug:
#     filename = f"eval_jaqket_{jacket_target}_debug"
# else:
#     filename = f"eval_jaqket_{jacket_target}"

# if parsed_args.cross_encoder_reranking:
#     filename_ce_name = parsed_args.cross_encoder_reranking.split("/")[-1]
#     filename += "_ce_" + filename_ce_name

# filename = filename + ".csv"
# print("save csv: ", filename)
# df.to_csv(filename, index=False)
