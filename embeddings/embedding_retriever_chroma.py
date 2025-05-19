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

#from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
#from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack import Document

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.utils import ComponentDevice

#Defining global variables
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_DEVICE = ComponentDevice.from_str("cuda")
EMBEDDING_DIM = 384

def _noop_embed(texts):
    # return zero vectors – they’ll never be queried
    return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

def build_store(path_kb, db_dir = "data_store/chroma_db", collection = "miniLM_cosine"):
    #https://afolabi-lagunju.medium.com/building-a-q-a-chatbot-on-your-documents-with-haystack-2-x-488738eb5206
    #reopen from disk if it exists

    if Path(db_dir, "chroma_new.sqlite3").exists():
        return ChromaDocumentStore(
            persist_path=db_dir,
            collection_name=collection,
            embedding_function="default"    
        )
     
    doc_store = ChromaDocumentStore(
        persist_path=db_dir,
        collection_name=collection,
        embedding_function="default")   

    indexing = Pipeline()
    indexing.add_component("clean",  DocumentCleaner())
    indexing.add_component("split",  DocumentSplitter(split_by="sentence", split_length=1))
    indexing.add_component("embed",  SentenceTransformersDocumentEmbedder(
                                        model=MODEL_NAME,
                                        device=GPU_DEVICE,
                                        batch_size=2048,
                                        #precision="float16",
                                        model_kwargs={"torch_dtype": torch.float16}
                                        ))
    indexing.add_component("writer", DocumentWriter(doc_store, policy=DuplicatePolicy.OVERWRITE))
    indexing.connect("clean", "split")
    indexing.connect("split", "embed")
    indexing.connect("embed", "writer")

    
    batch=[]
    BATCH_SIZE=5000
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
                        if len(batch) == BATCH_SIZE:
                            indexing.run({"clean": {"documents": batch}})
                            batch.clear()
                    
    if batch:
        indexing.run({"clean": {"documents": batch}})

    doc_store.persist()
    return doc_store


def retrieve_top_k_sentences(query: str, top_k: int):
    q_vec = EMBEDDER.run(query)["embedding"] 
    result = RETRIEVER.run(query_embedding=q_vec, top_k=top_k)
    docs = result["documents"]
    return [d.content for d in docs], [d.meta["url"] for d in docs]

def process_single_example(idx, example, args, result_queue, counter):
    try:
        start_time = time.time()
        questions = example.get("questions", [])
        query = example["claim"] + " " + " ".join(questions)
        
        processing_time = time.time() - start_time
        print(f"Top {args.top_k} retrieved. Time elapsed: {processing_time:.2f}s")
        sents, urls = retrieve_top_k_sentences(query, args.top_k)

        elapsed = time.time() - start_time
        print(f"Top-{args.top_k} retrieved in {elapsed:.2f}s")

        result = {
            "claim_id": idx,
            "claim": example["claim"],
            f"top_{args.top_k}":  [{"sentence": s, "url": u} for s, u in zip(sents, urls)],
            "questions": questions
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
    
    global DOC_STORE, SEARCHER, EMBEDDER      
    DOC_STORE = build_store(args.knowledge_store_dir)
    EMBEDDER = SentenceTransformersTextEmbedder(
    model=MODEL_NAME,
    device=GPU_DEVICE,     
    batch_size=512,       
    #precision="float16",
    progress_bar=False
    )
    EMBEDDER.warm_up() 
    EMBEDDER._backend.model.half() 
    RETRIEVER = ChromaEmbeddingRetriever(
        document_store=DOC_STORE,
        query_embedder=EMBEDDER,
        top_k=args.top_k
    )

    main(args)
