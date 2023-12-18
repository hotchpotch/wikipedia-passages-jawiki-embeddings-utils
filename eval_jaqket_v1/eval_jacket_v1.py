"""
jaqket v1 のデータセットを使って Q&A 回答の評価

$ python eval_jaqket_v1/eval_jacket_v1.py -k 5
"""

from __future__ import annotations

import os
import faiss
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import os
from time import time
from datasets.download import DownloadManager
from datasets import load_dataset  # type: ignore
import time
from dataclasses import dataclass
import json
import urllib.request
from tqdm import tqdm
from argparse import ArgumentParser

JAQKET_V1_TRAIN_URLS = [
    "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/train_questions.json",
]

JAQKET_V1_DEV_URLS = [
    "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev1_questions.json",
    "https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev2_questions.json",
]


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

SEARCH_TOP_K = 100

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
        gpu_res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # here we are using a over 64-byte PQ, so we must set the lookup tables to
        # 16 bit float (this is due to the limited temporary memory).
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index, co)
    faiss_index.nprobe = 256
    return faiss_index


def texts_to_embs(model, texts: list[str], prefix: str, show_progress_bar=True):
    texts = [prefix + text for text in texts]
    model.encode(["none"])  # warmup
    start_time = time.time()
    embs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=show_progress_bar
    )
    end_time = time.time()
    return embs, end_time - start_time


def faiss_search_by_embs(faiss_index, embs, top_k=5):
    start_time = time.time()
    D, I = faiss_index.search(embs, top_k)
    end_time = time.time()
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


def reranking_by_e5(
    model, idxs: list[int], jaqket: JaqketQuestionV1, wiki_ds
) -> list[int]:
    """
    e5 のモデルで question に近い文章を抽出し、再ランキングする
    """
    question = "query: " + jaqket.question
    passages = []
    for idx in idxs:
        data = wiki_ds[idx]
        title = data["title"]
        text = data["text"]
        section = data["section"]
        if section == "__LEAD__":
            section = "概要"
        passage = f"passage: # {title}\n\n## {section}\n\n### {text}"
        passages.append(passage)
    target_texts = [question] + passages
    embs, gen_embs_sec = texts_to_embs(
        model, target_texts, prefix="", show_progress_bar=False
    )
    # cosine similarityでソートする
    scores = embs[0].dot(embs[1:].T)
    sorted_idxs = scores.argsort()[::-1]
    # idxs を sorted_idxs 順序に並び替える
    sorted_idxs = [idxs[i] for i in sorted_idxs]
    return sorted_idxs


def reranking_indexes_by_e5(model, indexes, jaqket_ds, wiki_ds, top_k: int):
    reranked_indexes = []
    total = len(indexes)
    for idxs, jaqket in tqdm(zip(indexes, jaqket_ds), total=total):
        reranked_index = reranking_by_e5(model, idxs.tolist(), jaqket, wiki_ds)
        reranked_indexes.append(reranked_index[0:top_k])
    return reranked_indexes


args = ArgumentParser()
args.add_argument("-m", "--emb_model_name", type=str, default=None)
args.add_argument("-k", "--top_k", type=list, default=[1, 3, 5, 10, 20, 50, 100])
args.add_argument("-d", "--debug", action="store_true")
args.add_argument("-r", "--reranking", action="store_true")
args.add_argument("--use_gpu", action="store_true")

parsed_args = args.parse_args()
print("load wikija datasets")
ds = get_wikija_ds()

jacket_v1_dev = load_jaqket_v1(JAQKET_V1_DEV_URLS)
# jacket_v1_train = load_jaqket_v1(JAQKET_V1_TRAIN_URLS)

if parsed_args.debug:
    print("RUN: debug mode")
    jacket_v1_dev = jacket_v1_dev[:100]
    # jacket_v1_train = jacket_v1_train[:100]

jacket_v1 = {"dev": jacket_v1_dev}
# jacket_v1 = {"train": jacket_v1_train, "dev": jacket_v1_dev}

if parsed_args.emb_model_name:
    target_emb_models = [parsed_args.emb_model_name]
else:
    target_emb_models = EMB_MODEL_NAMES

top_k_s = parsed_args.top_k
use_gpu = parsed_args.use_gpu

# rerankingは e5 でのみ実行
if parsed_args.reranking:
    print("reranking is e5 only")
    target_emb_models = [m for m in target_emb_models if "e5" in m]
    print("reranking target_emb_models: ", target_emb_models)

results = []

for emb_model_name in target_emb_models:
    if "-e5-" in emb_model_name:
        query_passage = ["query", "passage"]
    else:
        query_passage = [""]

    print("load model: ", emb_model_name)
    model = get_model(emb_model_name)
    model.max_seq_length = 512

    for query_or_passage in query_passage:
        name = f"{emb_model_name}"
        if parsed_args.debug:
            name = f"[debug]{name}"
        index_emb_model_name = f"{emb_model_name.split('/')[-1]}"
        if query_or_passage:
            name = f"[{query_or_passage}]{name}"
            index_emb_model_name += f"-{query_or_passage}"
            # 検索するための prefix は元データが passage でも "query: " を指定する
            search_text_prefix = f"query: "
        else:
            # e5 以外のモデルは prefix なし
            search_text_prefix = ""
        emb_model_pq = EMB_MODEL_PQ[emb_model_name]
        print(f"\n\n--- {name} ---")

        print("load faiss index: ", index_emb_model_name)
        index_name = f"{index_emb_model_name}/index_IVF2048_PQ{emb_model_pq}.faiss"
        faiss_index = get_faiss_index(index_name, use_gpu=use_gpu)

        for target_split_name in jacket_v1.keys():
            jaqket = jacket_v1[target_split_name]
            print("gen embs: ", target_split_name)
            question_embs, gen_embs_sec = texts_to_embs(
                model, texts=[q.question for q in jaqket], prefix=search_text_prefix
            )
            gen_embs_sec = round(gen_embs_sec, 2)
            print("question_embs.shape: ", question_embs.shape)  # type: ignore
            print("gen embs sec: ", gen_embs_sec)
            scores, indexes, search_sec = faiss_search_by_embs(
                faiss_index, question_embs, top_k=SEARCH_TOP_K
            )
            search_sec = round(search_sec, 2)
            print("faiss search sec: ", search_sec)
            if parsed_args.reranking:
                print("reranking by e5")
                indexes = reranking_indexes_by_e5(
                    model, indexes, jaqket, ds, SEARCH_TOP_K
                )
            top_k_accuracies = []
            top_k_no_match_rates = []
            for top_k in top_k_s:
                target_indexes = indexes[:, 0:top_k]  # type: ignore
                pred_labels = predict_by_indexes(target_indexes, jaqket, ds)
                # pred labels に含まれる、-1 (見つからなかったデータ)の割合
                no_match_rate = sum([1 for l in pred_labels if l == -1]) / len(
                    pred_labels
                )
                # print("no match rate: ", no_match_rate)
                correct_count = sum(
                    [
                        1 if pred_label == q.label else 0
                        for pred_label, q in zip(pred_labels, jaqket)
                    ]
                )
                accuracy = correct_count / len(jaqket)
                # acc, no_match_rate を 0.xxxx に丸める
                top_k_accuracies.append(round(accuracy, 4))
                top_k_no_match_rates.append(round(no_match_rate, 4))
                # print("accuracy: ", accuracy)
                # append results
            # top_k_s と top_k_accuracies を、acc@1, acc@3 のようなdict keyにする
            top_k_accuracies = dict(
                zip([f"acc@{k}" for k in top_k_s], top_k_accuracies)
            )
            # NMR@1, NMR@3 のようなdict keyにする
            top_k_no_match_rates = dict(
                zip([f"NMR@{k}" for k in top_k_s], top_k_no_match_rates)
            )
            results.append(
                {
                    "name": name,
                    "ds_target": target_split_name,
                    **top_k_accuracies,
                    **top_k_no_match_rates,
                    "search_sec": search_sec,
                    "gen_embs_sec": gen_embs_sec,
                }
            )
        del faiss_index
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()


# 最後に results を df にして、table で全て表示
pd.set_option("display.max_rows", None)
df = pd.DataFrame(results)
print(df)

# csv で保存
if parsed_args.debug:
    df.to_csv(f"eval_jaqket_debug_top_{top_k}.csv", index=False)
else:
    df.to_csv(f"eval_jaqket_top_{top_k}.csv", index=False)
