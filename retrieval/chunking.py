from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        #每个文本块最多切成 512 个单位。这里的“单位”可能是 token，也可能是字符或词，取决于用的 splitter。
        #chunk_size 控制每块信息量
            # 太小：上下文被切碎，检索到的片段可能不完整
            # 太大：一个 chunk 混入太多主题，检索不够精准，还更费 embedding / 上下文窗口
        chunk_overlap=64, 


        #相邻两个文本块之间，重复保留 64 个单位
        #chunk_overlap 保证上下文连续
            # 防止一句话或一个概念刚好被切断在边界上
            # 提高召回率，尤其对跨段落信息更有帮助
        #代价也很明显：
            # overlap 越大，重复内容越多
            # 向量数量会变多
            # 存储、embedding 成本、检索冗余都会上升
        chunk_size=512, 
        
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for doc in documents:
        text = doc["content"]

        split_texts = splitter.split_text(text)

        for index, chunk_text in enumerate(split_texts):
            chunks.append({
                "title": doc.get("title", ""),
                "content": chunk_text,
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "date": doc.get("date", ""),
                "chunk_id": f"{doc.get('source', 'unknown')}_{index}",
                "chunk_index": index,
            })

    return chunks


# ======Example Usage=====
raw_documents_path = Path(__file__).resolve().parents[1] / "data" / "raw_documents.jsonl"

with raw_documents_path.open(encoding="utf-8") as file:
        documents = [json.loads(file.readline())] #读取文件的第一行
        
chunks = chunk_documents(documents)

print("num chunks:", len(chunks))
print(chunks)


