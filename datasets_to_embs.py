from __future__ import annotations
from dataclasses import dataclass
from typing import Generator
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import argparse
from pathlib import Path
from datasets import load_dataset

parser = argparse.ArgumentParser(
    description="Convert datasets to embeddings",
formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="debug mode, use small dataset",
)
# model_name
parser.add_argument(
    "-m",
    "--model_name",
    default="intfloat/multilingual-e5-small",
    type=str,
    help="huggingface(sentence transformer) model name",
)
parser.add_argument(
    "-p",
    "--passage_prefix",
    type=str,
    required=False,
    default="",
    help="prefix string for passage, ' or 'query: ' or 'passage: ' or 'prefix-foo-bar: '",
)
parser.add_argument(
    "--passage_template",
    type=str,
    required=False,
    default="# {title}\n\n## {section}\n\n{text} ",
    help="template for passage, {title}, {section}, {text} are replaced",
)
# max_seq_length
parser.add_argument(
    "-l",
    "--max_seq_length",
    type=int,
    required=False,
    default=512,
    help="max sequence length",
)
# output_name
parser.add_argument(
    "-w",
    "--working_dir",
    type=str,
    default="outputs",
    help="working_dir dir",
)
args = parser.parse_args()


@dataclass
class EmbConfig:
    model_name: str
    passage_prefix: str
    max_seq_length: int


args = parser.parse_args()


TEMPLATE = args.passage_template
PASSAGE_PREFIX = args.passage_prefix

passage_prefixes_e5 = ("query: ", "passage: ")

if "-e5-" in args.model_name:
    if args.passage_prefix not in passage_prefixes_e5:
        raise ValueError("passage_prefix should be one of", passage_prefixes_e5)

target_ds = load_dataset(args.target_dataset, 'passages-c400-jawiki-20230403', split='train')

def data_to_passage(data, template=TEMPLATE, prefix=PASSAGE_PREFIX):
    title = data['title']
    section = data['section'] 
    if section == "__LEAD__":
        section = "概要"
    text = data['text']
    formatted = template.format(title=title, section=section, text=text)
    return prefix + formatted

print("---- example formatted passages ----")
for i in range(3):
    print(data_to_passage(target_ds[i])) # type: ignore
    print("-" * 20)

emb_config = EmbConfig(
    model_name=args.model_name,
    passage_prefix=args.passage_prefix,
    max_seq_length=args.max_seq_length,
)
embs_dir = f"embs{'_debug' if args.debug else ''}"

model_name_for_embs_dir = emb_config.model_name.split("/")[-1]

if "-e5-" in model_name_for_embs_dir:
    if emb_config.passage_prefix == "query: ":
        model_name_for_embs_dir += "-query"
    elif emb_config.passage_prefix == "passage: ":
        model_name_for_embs_dir += "-passage"
    else:
        raise ValueError("passage_prefix should be one of", passage_prefixes_e5)


working_dir_embs_path = Path("/".join([args.working_dir, embs_dir, args.name, model_name_for_embs_dir]))
working_dir_embs_path.mkdir(parents=True, exist_ok=True)

print("output embs path:", working_dir_embs_path)

MODEL = SentenceTransformer(emb_config.model_name)
MODEL.max_seq_length = emb_config.max_seq_length


def to_embs(texts: list[str], group_size=1024, model=MODEL) -> Generator[np.ndarray, None, None]:
    group = []
    for text in texts:
        group.append(text)
        if len(group) == group_size:
            embeddings = model.encode(
                group,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            yield embeddings  # type: ignore
            group = []
    if len(group) > 0:
        embeddings = model.encode(
            group, normalize_embeddings=True, show_progress_bar=False
        )
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


if torch.cuda.is_available():
    print("use cuda")
    MODEL.to("cuda")
elif torch.backends.mps.is_available(): # type: ignore
    print("use mps (apple selicon)")
    MODEL.to("mps")
else:
    print("!! Warning: use cpu")

ds = target_ds # type: ignore

if args.debug:
    print("debug mode")
    ds = ds.select(range(19998))  # type: ignore
    print("small dataset len:", len(ds))
    group_size = 10_000
else:
    print("dataset len:", len(ds)) # type: ignore
    group_size = 100_000

for embs, idx, pbar in ds_to_embs(ds, data_to_passage, group_size=group_size):
    filename = f"{idx}.npz"
    filepath = working_dir_embs_path / filename
    pbar.desc = f"saving...: {str(filepath)}"
    np.savez_compressed(filepath, embs=embs.astype(np.float16))
    pbar.desc = ""
