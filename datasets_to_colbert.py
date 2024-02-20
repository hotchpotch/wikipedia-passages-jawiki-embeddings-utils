# %%
# import faiss

# faiss.Kmeans

# %%
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("bclavie/JaColBERT")

# %%
from datasets import load_dataset

ds = load_dataset(
    "singletongue/wikipedia-utils", "passages-c400-jawiki-20230403", split="train"
)

# %%
ds_head = ds  # ds.select(range(1000))

# %%
TEMPLATE = "# {title}\n\n## {section}\n\n### {text}"


def data_to_passage(data, template=TEMPLATE, prefix=""):
    title = data["title"]
    section = data["section"]
    if section == "__LEAD__":
        section = "概要"
    text = data["text"]
    formatted = template.format(title=title, section=section, text=text)
    return prefix + formatted


# %%
texts = [data_to_passage(data) for data in ds_head]

# %%
# check text len in texts
import numpy as np

# np.mean([len(text) for text in texts])
np.max([len(text) for text in texts])

# %%
RAG.index(
    collection=texts,
    index_name="passages-c400-jawiki-20230403",
    document_ids=[str(i) for i in range(len(texts))],
    max_document_length=512,
    split_documents=False,
)

# %%
RAG.search("常用漢字とは？")

# %%
# RAG.from_index(".ragatouille/colbert/indexes/tmp/")
