# wikipedia-passages-jawiki-embeddings-utils

wikipedia 日本語の文を、各種日本語の embeddings や faiss index へと変換するスクリプト等。

- [RAG用途に使える、Wikipedia 日本語の embeddings とベクトル検索用の faiss index を作った](https://secon.dev/entry/2023/12/04/080000-wikipedia-ja-embeddings/)
- [HuggingFace Space 上のデモ](https://huggingface.co/spaces/hotchpotch/wikipedia-japanese-rag-qa)

## 大元のデータ

- https://huggingface.co/datasets/singletongue/wikipedia-utils

## 生成したデータ

- https://huggingface.co/datasets/hotchpotch/wikipedia-passages-jawiki-embeddings

## Web UI の実行例

```
cd streamlit_qa_app
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```


## 変換例

```
export WORKING_DIR=/home/hotchpotch/src/huggingface.co/datasets/hotchpotch/wikipedia-passages-jawiki-embeddings/
python datasets_to_embs.py -w $WORKING_DIR -n passages-c400-jawiki-20230403 -p 'query: ' -m "intfloat/multilingual-e5-large"
python embs_to_faiss.py -w $WORKING_DIR -t passages-c400-jawiki-20230403/multilingual-e5-large-passage -f "IVF2048,PQ256"
```
