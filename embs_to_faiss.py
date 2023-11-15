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
    "-m",
    "--m",
    type=int,
    required=False,
    default=32,
    help="faiss param: m, subvector",
)
# mbit
parser.add_argument(
    "-b",
    "--mbit",
    type=int,
    required=False,
    default=8,
    help="faiss param: mbit, bits_per_idx",
)
# nlist
parser.add_argument(
    "-n",
    "--nlist",
    type=int,
    required=False,
    default=512,
    help="faiss param: nlist",
)
# use_gpu
parser.add_argument(
    "-g",
    "--use_gpu",
    action="store_true",
    help="use gpu",
)
# no quantization
parser.add_argument(
    "--flat-l2",
    action="store_true",
    help="use FlatL2, no quantization",
)
# force override
parser.add_argument(
    "--force",
    action="store_true",
    help="force override existing index file",
)
args = parser.parse_args()


@dataclass
class FaissConfig:
    m: int = 32  # subvector
    mbit: int = 8  # bits_per_idx
    nlist: int = 512  # nlist
    flat_l2: bool = True


args = parser.parse_args()

faiss_config = FaissConfig(
    m=args.m,
    mbit=args.mbit,
    nlist=args.nlist,
    flat_l2=args.flat_l2,
)

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


def get_index_filename(config: FaissConfig) -> str:
    if config.flat_l2:
        return f"index_flat_l2.faiss"
    return f"index_m{config.m}_mbit{config.mbit}_nlist{config.nlist}.faiss"


def gen_faiss_index(config: FaissConfig, dim: int, use_gpu: bool):
    if use_gpu and config.m > 48:
        return gen_faiss_index_f16_lookup(config, dim, use_gpu)
    # quantizer = faiss.IndexFlatL2(dim)
    if config.flat_l2:
        faiss_index = faiss.IndexFlatL2(dim)
    else:
        # faiss_index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, 16)
        # faiss_index.hnsw.efConstruction = 512
        # index.hnsw.efSearch = 128
        # quantizer = faiss.IndexHNSWFlat(dim, 32)
        quantizer = faiss.IndexFlatL2(dim)

        faiss_index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            config.nlist,
            config.m,
            config.mbit,
        )
        # faiss_index = faiss.IndexIVFFlat(quantizer, dim, 16384)
        # faiss_index.cp.min_points_per_centroid = 5  # quiet warning
        # faiss_index.quantizer_trains_alone = 2
    if use_gpu and getattr(faiss, "StandardGpuResources", None):
        gpu_res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)
        return faiss_index
    else:
        return faiss_index


def gen_faiss_index_f16_lookup(config: FaissConfig, dim: int, use_gpu: bool):
    # use float16 lookup tables
    gpu_resource = faiss.StandardGpuResources()  # GPUリソースの初期化
    gpu_index_config = faiss.GpuIndexIVFPQConfig()  # IVFPQインデックスの設定用オブジェクト
    gpu_index_config.useFloat16LookupTables = True  # Float16ルックアップテーブルを使用する
    quantizer = faiss.IndexFlatL2(dim)  # 量子化器の定義
    index = faiss.GpuIndexIVFPQ(
        gpu_resource,
        # quantizer,
        dim,
        config.nlist,
        config.m,
        config.mbit,
        faiss.METRIC_L2,
        gpu_index_config,  # IVFPQインデックスの設定用オブジェクト
    )  # GPU上のIVFPQインデックスの作成
    return index


output_faiss_path = (
    working_path.joinpath("faiss_indexes")
    .joinpath(args.target_embs_name)
    .joinpath(get_index_filename(faiss_config))
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
if faiss_config.nlist is None:
    faiss_config.nlist = int(np.sqrt(len(embs) * (len(embs_npz) - 1)))
    if faiss_config.nlist < 1:
        faiss_config.nlist = 100
print(f"faiss_config: {faiss_config}")
if args.use_gpu:
    print("use gpu for faiss index")
faiss_index = gen_faiss_index(faiss_config, dim, args.use_gpu)
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
