from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import tiktoken
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Convert datasets to embeddings",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-t",
    "--target_dataset",
    type=str,
    default="singletongue/wikipedia-utils",
    help="target huggingface dataset",
)
# -n, name
parser.add_argument(
    "-n",
    "--name",
    type=str,
    default="passages-c400-jawiki-20230403",
    help="huggingface datasets name(subdir)",
)

# model_name
parser.add_argument(
    "-m",
    "--model_name",
    default="text-embedding-3-small",
    type=str,
    help="openai embedding model name",
)
# dim
parser.add_argument(
    "--dim",
    type=int,
    default=512,
    help="dimensions",
)

parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="debug mode, use small dataset",
)
parser.add_argument(
    "--passage_template",
    type=str,
    required=False,
    default="# {title}\n\n## {section}\n\n### {text}",
    help="template for passage, {title}, {section}, {text} are replaced",
)

# output_name
parser.add_argument(
    "-w",
    "--working_dir",
    type=str,
    default="outputs",
    help="working_dir dir",
)
# only count tokens
parser.add_argument(
    "-c",
    "--only_count_tokens",
    action="store_true",
    required=False,
    default=False,
)
args = parser.parse_args()


@dataclass
class EmbConfig:
    model_name: str
    dim: int


args = parser.parse_args()


TEMPLATE = args.passage_template

target_ds = load_dataset(args.target_dataset, args.name, split="train")


def data_to_passage(data, template=TEMPLATE, prefix=""):
    title = data["title"]
    section = data["section"]
    if section == "__LEAD__":
        section = "概要"
    text = data["text"]
    formatted = template.format(title=title, section=section, text=text)
    return prefix + formatted


print("---- example formatted passages ----")
for i in range(3):
    print(data_to_passage(target_ds[i]))  # type: ignore
    print("-" * 20)

emb_config = EmbConfig(
    model_name=args.model_name,
    dim=args.dim,
)
embs_dir = f"embs{'_debug' if args.debug else ''}"

model_name_for_embs_dir = emb_config.model_name.split("/")[-1] + f"-dim{emb_config.dim}"

working_dir_embs_path = Path(
    "/".join([args.working_dir, embs_dir, args.name, model_name_for_embs_dir])
)
working_dir_embs_path.mkdir(parents=True, exist_ok=True)

print("output embs path:", working_dir_embs_path)

MODEL = OpenAIEmbeddings(
    model=emb_config.model_name,
    tiktoken_model_name="cl100k_base",
    dimensions=emb_config.dim,
)


def to_embs(
    texts: list[str], group_size=1024, model=MODEL
) -> Generator[np.ndarray, None, None]:
    group = []
    for text in texts:
        group.append(text)
        if len(group) == group_size:
            embeddings = model.embed_documents(group)
            yield embeddings  # type: ignore
            group = []
    if len(group) > 0:
        embeddings = model.embed_documents(group)
        yield embeddings  # type: ignore


def ds_to_embs(
    ds,
    text_fn,
    group_size: int,
):
    texts = []
    total = len(ds)
    pbar = tqdm(total=total)
    # text は group_size 件ごとに処理する
    for i in range(0, total, group_size):
        if get_npz_file_path(i).exists():
            pbar.update(group_size)
            yield None, i, pbar
            continue
        texts = []
        for data in ds.select(range(i, min(i + group_size, total))):
            data: dict = data
            text = text_fn(data)
            texts.append(text)
        embs = []
        for group_embs in to_embs(texts):
            embs.append(group_embs)
            pbar.update(len(group_embs))
        embs = np.concatenate(embs)
        yield embs, i, pbar


def count_tokens_data(ds) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for data in tqdm(ds, total=len(ds)):
        text = data_to_passage(data)
        c = len(encoding.encode(text))
        total += c
    # passage-400 で 1490618785
    return total


def get_npz_file_path(idx: int) -> Path:
    filename = f"{idx}.npz"
    filepath = working_dir_embs_path / filename
    return filepath


ds = target_ds  # type: ignore

if args.debug:
    print("debug mode")
    ds = ds.select(range(1998))  # type: ignore
    print("small dataset len:", len(ds))
    group_size = 1_000
else:
    print("dataset len:", len(ds))  # type: ignore
    group_size = 100_000

if args.only_count_tokens:
    print("counting tokens...")
    total = count_tokens_data(ds)
    print("total tokens:", total)
else:
    for embs, idx, pbar in ds_to_embs(ds, data_to_passage, group_size=group_size):
        filepath = get_npz_file_path(idx)
        if embs is None:
            print("skip, file exists:", filepath)
            continue
        pbar.desc = f"saving...: {str(filepath)}"
        np.savez_compressed(filepath, embs=embs.astype(np.float16))
        pbar.desc = ""
