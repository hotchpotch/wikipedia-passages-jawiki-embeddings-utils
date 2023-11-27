from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import faiss

parser = argparse.ArgumentParser(
    description="Convert embs to faiss index",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-w",
    "--working_dir",
    type=str,
    default="outputs",
    help="working_dir dir, ex: ~/works/",
)
parser.add_argument(
    "-t",
    "--target_embs_name",
    type=str,
    required=True,
    help="ex: passages-c400-jawiki-20230403/multilingual-e5-small-passage",
)
# m
parser.add_argument(
    "-f",
    "--factory",
    type=str,
    required=False,
    default="IVF2048,PQ96",
    help="faiss index_factory() option",
)

# use_gpu
parser.add_argument(
    "-g",
    "--use_gpu",
    action="store_true",
    help="use gpu",
)
# force override
parser.add_argument(
    "--force",
    action="store_true",
    help="force override existing index file",
)
args = parser.parse_args()


working_path = Path(args.working_dir)
# not found
if not working_path.exists():
    print(f"working_dir not found: {working_path}")
    exit(1)

target_embs_path = working_path.joinpath("embs").joinpath(args.target_embs_name)
embs_npz = list(target_embs_path.glob("*.npz"))
embs_npz.sort(key=lambda x: int(x.stem))

if len(embs_npz) == 0:
    print(f"target embs not found: {target_embs_path}")
    exit(1)
else:
    print(f"input {len(embs_npz)} embs(*.npz) found: {target_embs_path}")


def get_index_filename(factory_name: str) -> str:
    factory_name = factory_name.replace(",", "_")
    return f"index_{factory_name}.faiss"


def gen_faiss_index(factory_name: str, dim: int, use_gpu: bool):
    faiss_index = faiss.index_factory(dim, factory_name)
    if use_gpu:  # and getattr(faiss, "StandardGpuResources", None):
        gpu_res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # here we are using a over 64-byte PQ, so we must set the lookup tables to
        # 16 bit float (this is due to the limited temporary memory).
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index, co)
        return faiss_index
    else:
        return faiss_index


output_faiss_path = (
    working_path.joinpath("faiss_indexes")
    .joinpath(args.target_embs_name)
    .joinpath(get_index_filename(args.factory))
)

output_faiss_path.parent.mkdir(parents=True, exist_ok=True)

# output_faiss_path がすでにある場合
if output_faiss_path.exists():
    if args.force:
        print("force override existing index file")
        print(f"[found] -> {output_faiss_path}")
    else:
        print("index file already exists, skip")
        print(f"[found] -> {output_faiss_path}")
        exit(0)


# pbar = tqdm(total=len(input_embs_npz))
# emb_total = 0
all_embs = []
for idx, npz_file in enumerate(tqdm(embs_npz)):
    with np.load(npz_file) as data:
        e = data["embs"].astype("float16")
    all_embs.append(e)
embs: np.ndarray = np.concatenate(all_embs, axis=0, dtype="float32")
del all_embs

# if idx == 0:
dim = embs.shape[1]
if args.use_gpu:
    print("use gpu for faiss index")
faiss_index = gen_faiss_index(args.factory, dim, args.use_gpu)
print(f"start training faiss index, shape: {embs.shape}")
faiss_index.train(embs)  # type: ignore
print(f"start adding embs to faiss index")
faiss_index.add(embs)  # type: ignore
# pbar.update(1)
print(f"added embs: {embs.shape[0]}")
# pbar.set_description(f"added embs: {emb_total}")
# pbar.close()

faiss_index.nprobe = 256  # type: ignore
if args.use_gpu:
    faiss_index = faiss.index_gpu_to_cpu(faiss_index)  # type: ignore
faiss.write_index(faiss_index, str(output_faiss_path))  # type: ignore
print("output faiss index file:", output_faiss_path)
# output_faiss_path のファイルサイズを MB で表示
print(
    "output faiss index file size (MB):",
    int(output_faiss_path.stat().st_size / 1024 / 1024),
)
