"""
working_dir から、対象のembsに対して、同じオプションのIndexを作成する
$ python all_embs_to_faiss.py -w $WORKING_DIR --use_gpu -f "IVF2048,PQ96"

"""

import argparse
import subprocess
import sys
from pathlib import Path


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


def run_subprocess(working_dir, target_embs_name, other_args):
    args = [
        "python",
        "embs_to_faiss.py",
        "-w",
        working_dir,
        "-t",
        target_embs_name,
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
        target_embs_name = "/".join(str(d).split("/")[-2:])
        print(f"target_embs_name: {target_embs_name}")
        run_subprocess(working_dir, target_embs_name, other_args)
