import argparse
import json
import os
import time
import numpy as np
from functools import partial
import heapq
from threading import Thread, Event
import queue
from datetime import datetime, timedelta

import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock, cpu_count

##
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack import Document
import json, os
##

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def worker_init(path_kb, model_name):
    global DOC_STORE, EMBEDDER
    DOC_STORE  = build_store(path_kb)
    EMBEDDER   = SentenceTransformersTextEmbedder(model=model_name)
    EMBEDDER.warm_up()


def build_store(path_kb: str):
    """Index everything once – run this only when KB changes."""
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

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

    #Indexing
    for fn in os.listdir(path_kb):
        docs = []
        with open(os.path.join(path_kb, fn), encoding="utf-8") as fh:
            for line in fh:
                data = json.loads(line)
                for url, txt in zip(data["url"], data["url2text"]):
                    docs.append(Document(content=txt, meta={"url": url}))
        indexing.run({"clean": {"documents": docs}})

    return doc_store

def init_retriever(doc_store, top_k):
    embedder  = SentenceTransformersTextEmbedder(model=MODEL_NAME)
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store, top_k=top_k)
    return embedder, retriever

def retrieve_top_k_sentences(query: str, top_k: int):
    q_vec = EMBEDDER.run(query)["embedding"]
    #docs  = DOC_STORE.query_by_embedding(q_vec, top_k=top_k)["documents"]
    #return [d.content for d in docs], [d.meta["url"] for d in docs]
    docs  = DOC_STORE.query_by_embedding(q_vec, top_k=top_k)["documents"]
    seen=set(); uniq=[]
    for d in docs:
        if d.content not in seen:
            seen.add(d.content)
            uniq.append(d)
        if len(uniq) == top_k:
            break
    return [d.content for d in uniq], [d.meta["url"] for d in uniq]



def process_single_example(idx, example, args, result_queue, counter, lock):
    try:
        with lock:
            current_count = counter.value + 1
            counter.value = current_count
            print(f"\nProcessing claim {idx}... Progress: {current_count} / {args.total_examples}")
        
        start_time = time.time()
        
        query = example["claim"] + " " + " ".join(example['hypo_fc_docs'])
        
        processing_time = time.time() - start_time
        print(f"Top {args.top_k} retrieved. Time elapsed: {processing_time:.2f}s")
        sents, urls = retrieve_top_k_sentences(query, args.top_k)

        elapsed = time.time() - start_time
        print(f"Top-{args.top_k} retrieved in {elapsed:.2f}s")

        result = {
            "claim_id": idx,
            "claim": example["claim"],
            f"top_{args.top_k}":  [{"sentence": s, "url": u} for s, u in zip(sents, urls)],
            "hypo_fc_docs": example["hypo_fc_docs"]
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

    worker_init(args.knowledge_store_dir, MODEL_NAME)

    with Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        args.total_examples = len(files_to_process)
        
        result_queue = mp.Queue()
        
        stop_event = Event()
        writer = Thread(
            target=writer_thread,
            args=(args.json_output, result_queue, args.start, stop_event)
        )
        writer.start()

        results = []
        for idx, example in examples_to_process:
            ok = process_single_example(
                idx, example, args,
                result_queue, counter, lock
            )
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
    mp.set_start_method('spawn', force=True) #Handlelig GPU issue

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
        default=5000,
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
    
    main(args)
