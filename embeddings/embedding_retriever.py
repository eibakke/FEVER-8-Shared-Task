#IMPORTS
import argparse
import json
import os
import time
import numpy as np
import heapq
import queue
from datetime import datetime, timedelta
from pathlib import Path
import torch
from threading import Thread, Event

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack import Document

#Defining global variables
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_store(path_kb: str):
    """Function that takes in path to folder with the knowledge base.
    Returns an InMemoryDocumentStore item. This element is also stored in a file in the datastore"""
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    #Setting up pipeline (same as seminar 11, https://github.uio.no/in5550/2025/blob/main/labs/11/rag_example.ipynb)
    indexing = Pipeline()
    indexing.add_component("clean",   DocumentCleaner())
    indexing.add_component("split",   DocumentSplitter(split_by="sentence", split_length=1))
    indexing.add_component("embed",   SentenceTransformersDocumentEmbedder(
                                        model=MODEL_NAME, meta_fields_to_embed=["url"]))
    indexing.add_component("writer",  DocumentWriter(doc_store,
                                        policy=DuplicatePolicy.OVERWRITE))
    indexing.connect("clean",  "split")
    indexing.connect("split",  "embed")
    indexing.connect("embed",  "writer")

    #Due to issues with OOM, we run in batches
    batch_size=100 #FIRST TEST 
    batch = []
    for root, _, files in os.walk(path_kb):
        for fn in files:
            if not fn.endswith((".jsonl", ".json")):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as fh:
                for line in fh:
                    data = json.loads(line)
                    url  = data["url"]
                    for txt in data["url2text"]:
                        batch.append(Document(content=txt, meta={"url": url}))
                        if len(batch) == batch_size:
                            indexing.run({"clean": {"documents": batch}})
                            batch.clear()
    if batch:
        indexing.run({"clean": {"documents": batch}})

    #Storing object in file
    Path("data_store/miniLM_cosine.pkl").parent.mkdir(parents=True, exist_ok=True)
    doc_store.save("data_store/miniLM_cosine.pkl")
 
    return doc_store

def build_or_load(path_kb: str, store_fp="data_store/miniLM_cosine.pkl"):
    """Function checks if there is a datastore element in the given folder.
    If yes this is imported. If not, we call build store function"""
    if Path(store_fp).exists():
        return InMemoryDocumentStore.load(store_fp)
    return build_store(path_kb)

def retrieve_top_k_sentences(query: str, top_k: int):
    q_vec = EMBEDDER.run(query)["embedding"]
    docs  = SEARCHER.topk(q_vec, top_k)
    return [d.content for d in docs], [d.meta["url"] for d in docs]

def process_single_example(idx, example, args, result_queue, counter):
    try:
        start_time = time.time()
        
        query = example["claim"] + " " + " ".join(example['questions'])
        
        processing_time = time.time() - start_time
        print(f"Top {args.top_k} retrieved. Time elapsed: {processing_time:.2f}s")
        sents, urls = retrieve_top_k_sentences(query, args.top_k)

        elapsed = time.time() - start_time
        print(f"Top-{args.top_k} retrieved in {elapsed:.2f}s")

        result = {
            "claim_id": idx,
            "claim": example["claim"],
            f"top_{args.top_k}":  [{"sentence": s, "url": u} for s, u in zip(sents, urls)],
            "questions": example["questions"]
        }
        
        result_queue.put((idx, result))
        return True
    except Exception as e:
        print(f"Error processing example {idx}: {str(e)}")
        result_queue.put((idx, None))
        return False

def writer_thread(output_file, result_queue, next_index, stop_event):
    pending_results = []
    
    with open(output_file, "w", encoding="utf-8") as f:
        while not (stop_event.is_set() and result_queue.empty()):
            try:
                idx, result = result_queue.get(timeout=1)
                
                if result is not None:
                    heapq.heappush(pending_results, (idx, result))
                
                while pending_results and pending_results[0][0] == next_index:
                    _, result_to_write = heapq.heappop(pending_results)
                    f.write(json.dumps(result_to_write, ensure_ascii=False) + "\n")
                    f.flush()
                    next_index += 1
                    
            except queue.Empty:
                continue

def format_time(seconds):
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))

class GPUSearcher:
    def __init__(self, doc_store, device="cuda"):
        embs  = np.stack([d.embedding for d in doc_store._embedding_id_to_embedding.values()])
        self.doc_ids = np.fromiter(doc_store._embedding_id_to_doc_id.values(), dtype=np.int64)
        self.matrix  = torch.from_numpy(embs).to(device)
        self.matrix  = torch.nn.functional.normalize(self.matrix, dim=1)
        self.device  = device

    @torch.inference_mode()
    def topk(self, query_vec: np.ndarray, k: int):
        q = torch.from_numpy(query_vec).to(self.device)
        q = torch.nn.functional.normalize(q, dim=0)
        sims = self.matrix @ q                    # (N,)
        vals, idx = torch.topk(sims, k)
        doc_ids   = self.doc_ids[idx.cpu()]
        return [DOC_STORE.get_document_by_id(i) for i in doc_ids]

def main(args):
    script_start = time.time()
    start_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")

    with open(args.target_data, "r", encoding="utf-8") as fh:
        target_examples = json.load(fh)

    if args.end == -1:
        args.end = len(target_examples)

    files_to_process    = list(range(args.start, args.end))
    examples_to_process = [(idx, target_examples[idx]) for idx in files_to_process]
    print(f"Total examples to process: {len(examples_to_process)}")

    args.total_examples = len(examples_to_process)

    counter = 0  
    lock = None                       
    result_queue = queue.Queue()
    stop_event = Event()
    writer = Thread(
        target=writer_thread,
        args=(args.json_output, result_queue, args.start, stop_event)
    )
    writer.start()

    results = []
    for idx, example in examples_to_process:
        ok = process_single_example(idx, example, args, result_queue, counter)
        counter += 1                         
        results.append(ok)

    stop_event.set()
    writer.join()

    successful = sum(results)
    print(f"\nSuccessfully processed {successful} out of {len(files_to_process)} examples")
    print(f"Results written to {args.json_output}")
        
    # Calculate and display timing information
    total_time = time.time() - script_start
    avg_time = total_time / len(files_to_process)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)} (HH:MM:SS)")
    print(f"Average time per example: {avg_time:.2f} seconds")
    if successful > 0:
        print(f"Processing speed: {successful / total_time:.2f} examples per second")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Retrieve top-k sentences with FAISS in parallel"
    )
    parser.add_argument(
        "-k",
        "--knowledge_store_dir",
        type=str,
        default="data_store/knowledge_store",
        help="The path of the knowledge_store_dir containing json files with all the retrieved sentences.",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default="data_store/hyde_fc.json",
        help="The path of the file that stores the claim.",
    )
    parser.add_argument(
        "-o",
        "--json_output",
        type=str,
        default="data_store/dev_retrieval_top_k.json",
        help="The output dir for JSON files to save the top 100 sentences for each claim.",
    )
    parser.add_argument(
        "--top_k",
        default=5, #5000,
        type=int,
        help="How many documents should we pick out with dense retriever",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Starting index of the files to process.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="End index of the files to process.",
    )

    args = parser.parse_args()
    
    global DOC_STORE, SEARCHER, EMBEDDER          # declare globals
    DOC_STORE  = build_or_load(args.knowledge_store_dir)
    SEARCHER   = GPUSearcher(DOC_STORE, device=DEVICE)
    EMBEDDER   = SentenceTransformersTextEmbedder(model=MODEL_NAME, device=DEVICE)
    EMBEDDER.warm_up()
    
    main(args)
