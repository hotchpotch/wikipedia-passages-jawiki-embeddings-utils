"""
working_dir から、対象のembsに対して、同じオプションのIndexを作成する
-f オプションのPQの値は dim から自動で選ばれる
$ python all_embs_to_faiss.py -w $WORKING_DIR --use_gpu

"""

# ToDo: dim をみてPQをセットする?

import argparse
import subprocess
import sys
import numpy as np
from pathlib import Path

IVF_NLIST = 2048


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        default="outputs",
        help="working_dir dir, ex: ~/works/",
    )
    args, unknown_args = parser.parse_known_args()
    return args.working_dir, unknown_args


def run_subprocess(working_dir, target_embs_name, other_args, ivf_nlist, pq):
    args = [
        "python",
        "embs_to_faiss.py",
        "-w",
        working_dir,
        "-t",
        target_embs_name,
        "-f",
        f"IVF{ivf_nlist},PQ{pq}",
    ] + other_args
    print(f"args: {args}")

    subprocess.run(args)


if __name__ == "__main__":
    working_dir, other_args = parse_args()
    working_dirs = Path(working_dir).glob("embs/*/*")
    # dir だけ抽出
    dirs = [d for d in working_dirs if d.is_dir()]
    # unique
    dirs = list(set(dirs))
    for d in dirs:
        # d/0.npz を取得
        npz_file = d / "0.npz"
        if not npz_file.exists():
            print(f"{npz_file} is not exists")
            continue
        # npz_file から、shape を取得
        with np.load(npz_file) as data:
            e = data["embs"]
        dim = e.shape[1]

        # dim が96よりも高いなら、other_args に --use_gpu があれば削除
        if dim > 96:
            if "--use_gpu" in other_args:
                other_args.remove("--use_gpu")

        target_embs_name = "/".join(str(d).split("/")[-2:])
        print(f"target_embs_name: {target_embs_name}")
        run_subprocess(
            working_dir,
            target_embs_name,
            other_args,
            ivf_nlist=IVF_NLIST,
            pq=int(dim / 4),
        )
