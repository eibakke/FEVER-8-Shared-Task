import argparse
import json
import os
import time
import numpy as np
from functools import partial
import heapq
from threading import Thread, Event
import queue
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback

import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock, cpu_count

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Document

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR = "/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task/embedding_data_store"
DEBUG = True  # Enable detailed debugging


def debug_print(message):
    """Print debug messages with timestamp"""
    if DEBUG:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def worker_init(model_name):
    """Initialize only the embedder in the worker"""
    global EMBEDDER
    try:
        debug_print(f"Initializing worker with model {model_name}...")
        EMBEDDER = SentenceTransformersTextEmbedder(model=model_name)
        debug_print("Warming up embedder...")
        EMBEDDER.warm_up()
        print("Worker initialized with embedder")
        return True
    except Exception as e:
        print(f"Error initializing worker: {str(e)}")
        traceback.print_exc()
        return False


def build_claim_store(knowledge_file, claim_id):
    """Build document store for a specific claim"""
    store_path = os.path.join(STORE_DIR, MODEL_NAME, f"store_{claim_id}.json")

    # Return existing store if available
    if os.path.exists(store_path):
        debug_print(f"Loading existing document store for claim {claim_id}")
        try:
            doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            doc_store.load_from_disk(store_path)
            debug_print(f"Successfully loaded document store with {doc_store.count_documents()} documents")
            return doc_store
        except Exception as e:
            debug_print(f"Error loading stored document store: {str(e)}")
            # If loading fails, continue to build a new one
            pass

    debug_print(f"Building document store for claim {claim_id}")
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    debug_print("Creating indexing pipeline...")
    indexing = Pipeline()
    indexing.add_component("clean", DocumentCleaner())
    indexing.add_component("split", DocumentSplitter(split_by="sentence", split_length=1))
    indexing.add_component("embed", SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME, meta_fields_to_embed=["url"]))
    indexing.add_component("writer", DocumentWriter(doc_store,
                                                    policy=DuplicatePolicy.OVERWRITE))
    indexing.connect("clean", "split")
    indexing.connect("split", "embed")
    indexing.connect("embed", "writer")

    # Track unique documents
    all_content_seen = set()
    total_docs = 0
    unique_docs = 0

    # Process only the specific knowledge file for this claim
    try:
        if not os.path.exists(knowledge_file):
            debug_print(f"Knowledge file not found: {knowledge_file}")
            return None

        debug_print(f"Reading knowledge file: {knowledge_file}")

        # Get file size
        file_size = os.path.getsize(knowledge_file)
        debug_print(f"Knowledge file size: {file_size / (1024 * 1024):.2f} MB")

        docs = []
        with open(knowledge_file, encoding="utf-8") as fh:
            line_count = 0
            for line in fh:
                line_count += 1
                if line_count % 10 == 0:
                    debug_print(f"Processed {line_count} lines, collected {total_docs} documents")

                try:
                    data = json.loads(line)
                    # Check data structure
                    if "url" not in data or "url2text" not in data:
                        debug_print(f"Invalid data structure in line {line_count}")
                        continue

                    # Normalize URL handling
                    url_value = data["url"]
                    url_list = [url_value] * len(data["url2text"]) if isinstance(url_value, str) else data["url"]

                    for url, txt in zip(url_list, data["url2text"]):
                        total_docs += 1

                        # Normalize content for deduplication
                        content_key = txt.strip().lower()

                        # Only add unique content
                        if content_key not in all_content_seen and txt.strip():  # Skip empty content
                            all_content_seen.add(content_key)
                            docs.append(Document(content=txt, meta={"url": url}))
                            unique_docs += 1

                except json.JSONDecodeError:
                    debug_print(f"Invalid JSON in line {line_count}")
                except Exception as e:
                    debug_print(f"Error processing line {line_count}: {str(e)}")

        debug_print(f"Document collection complete. Indexing {len(docs)} documents...")

        if len(docs) == 0:
            debug_print("No documents collected, skipping indexing")
            return None

        # Index documents in smaller batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            debug_print(f"Indexing batch {i // batch_size + 1}/{len(docs) // batch_size + 1} ({len(batch)} documents)")
            indexing.run({"clean": {"documents": batch}})
            debug_print(f"Batch {i // batch_size + 1} indexed")

        print(f"Claim {claim_id}: {unique_docs} unique documents from {total_docs} total")

        # Save document store using Haystack's to_disk method
        debug_print(f"Saving document store to {store_path}")
        os.makedirs(store_path, exist_ok=True)
        doc_store.save_to_disk(store_path)
        debug_print("Document store saved successfully")

        return doc_store

    except Exception as e:
        print(f"Error building document store for claim {claim_id}: {str(e)}")
        traceback.print_exc()
        return None


def retrieve_top_k_sentences(claim_id, query, top_k, knowledge_store_dir):
    """Retrieve top-k sentences for a claim using its specific document store"""
    try:
        # Get the knowledge file path for this claim
        knowledge_file = os.path.join(knowledge_store_dir, f"{claim_id}.json")
        debug_print(f"Knowledge file path: {knowledge_file}")

        if not os.path.exists(knowledge_file):
            print(f"ERROR: Knowledge file not found: {knowledge_file}")
            return [], []

        # Build/load document store for this claim
        debug_print("Building/loading document store...")
        doc_store = build_claim_store(knowledge_file, claim_id)
        if doc_store is None:
            print(f"ERROR: Failed to build document store for claim {claim_id}")
            return [], []

        # Verify document store has documents
        doc_count = doc_store.count_documents()
        debug_print(f"Document store contains {doc_count} documents")
        if doc_count == 0:
            print(f"ERROR: Document store is empty for claim {claim_id}")
            return [], []

        # Get query embedding and retrieve documents
        debug_print(f"Embedding query: {query[:100]}...")
        q_vec = EMBEDDER.run(query)["embedding"]
        debug_print(f"Query embedding shape: {np.array(q_vec).shape}")

        debug_print("Querying document store...")
        results = doc_store.query_by_embedding(q_vec, top_k=top_k)
        docs = results["documents"]

        # Print score information for debugging
        if len(docs) > 0:
            scores = results.get("scores", [])
            if scores:
                debug_print(f"Top score: {scores[0]}, Bottom score: {scores[-1]}")

        debug_print(f"Retrieved {len(docs)} documents")

        return [d.content for d in docs], [d.meta["url"] for d in docs]
    except Exception as e:
        print(f"Error in retrieve_top_k_sentences for claim {claim_id}: {str(e)}")
        traceback.print_exc()
        return [], []


def process_single_example(idx, example, args, result_queue, counter, lock):
    try:
        with lock:
            current_count = counter.value + 1
            counter.value = current_count
            print(f"\nProcessing claim {idx}... Progress: {current_count} / {args.total_examples}")

        start_time = time.time()
        debug_print(f"Claim {idx}: Starting processing")

        query = example["claim"] + " " + " ".join(example['hypo_fc_docs'])
        debug_print(f"Claim {idx}: Query constructed")

        # Retrieve using claim-specific document store
        debug_print(f"Claim {idx}: Calling retrieve_top_k_sentences")
        sents, urls = retrieve_top_k_sentences(
            idx, query, args.top_k, args.knowledge_store_dir
        )
        debug_print(f"Claim {idx}: Retrieved {len(sents)} sentences")

        processing_time = time.time() - start_time
        print(f"Claim {idx}: Top {len(sents)}/{args.top_k} retrieved. Time elapsed: {processing_time:.2f}s")

        result = {
            "claim_id": idx,
            "claim": example["claim"],
            f"top_{args.top_k}": [{"sentence": s, "url": u} for s, u in zip(sents, urls)],
            "hypo_fc_docs": example['hypo_fc_docs']
        }

        debug_print(f"Claim {idx}: Putting result in queue")
        result_queue.put((idx, result))
        debug_print(f"Claim {idx}: Processing complete")
        return True
    except Exception as e:
        print(f"Error processing example {idx}: {str(e)}")
        traceback.print_exc()
        result_queue.put((idx, None))
        return False


def process_examples_in_main_process(examples_to_process, args, result_queue, counter, lock):
    """Process examples in the main process to avoid pool issues"""
    debug_print("Processing examples in main process...")
    results = []

    # Initialize in main process
    worker_init(MODEL_NAME)

    for idx, example in examples_to_process:
        result = process_single_example(idx, example, args, result_queue, counter, lock)
        results.append(result)

    return results


def writer_thread(output_file, result_queue, next_index, stop_event):
    pending_results = []

    with open(output_file, "w", encoding="utf-8") as f:
        while not (stop_event.is_set() and result_queue.empty()):
            try:
                idx, result = result_queue.get(timeout=1)
                debug_print(f"Writer: Got result for claim {idx}")

                if result is not None:
                    heapq.heappush(pending_results, (idx, result))
                    debug_print(f"Writer: Added result for claim {idx} to heap")

                while pending_results and pending_results[0][0] == next_index:
                    _, result_to_write = heapq.heappop(pending_results)
                    f.write(json.dumps(result_to_write, ensure_ascii=False) + "\n")
                    f.flush()
                    debug_print(f"Writer: Wrote result for claim {next_index}")
                    next_index += 1

            except queue.Empty:
                # No need to print this, it will spam the logs
                pass
            except Exception as e:
                print(f"Error in writer thread: {str(e)}")
                traceback.print_exc()


def format_time(seconds):
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))


def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")

    # Create store directory
    debug_print(f"Creating store directory: {STORE_DIR}")
    os.makedirs(STORE_DIR, exist_ok=True)

    debug_print(f"Loading target data from: {args.target_data}")
    with open(args.target_data, "r", encoding="utf-8") as fh:
        target_examples = json.load(fh)

    if args.end == -1:
        args.end = len(target_examples)

    files_to_process = list(range(args.start, args.end))
    examples_to_process = [(idx, target_examples[idx]) for idx in files_to_process]
    print(f"Total examples to process: {len(examples_to_process)}")

    with Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        args.total_examples = len(files_to_process)

        result_queue = manager.Queue()

        stop_event = Event()
        debug_print("Starting writer thread...")
        writer = Thread(
            target=writer_thread,
            args=(args.json_output, result_queue, args.start, stop_event)
        )
        writer.start()

        # Process in main process for small batches or debugging
        if len(examples_to_process) <= 5 or DEBUG:
            debug_print("Processing in main process due to small batch size or debug mode")
            results = process_examples_in_main_process(
                examples_to_process, args, result_queue, counter, lock
            )
        else:
            # Process examples in parallel for larger batches
            debug_print(f"Processing in parallel with {args.workers} workers")
            worker_init(MODEL_NAME)  # Initialize in main process first

            process_func = partial(
                process_single_example,
                args=args,
                result_queue=result_queue,
                counter=counter,
                lock=lock
            )

            with Pool(processes=args.workers,
                      initializer=worker_init,
                      initargs=(MODEL_NAME,)) as pool:
                results = pool.starmap(process_func, examples_to_process)

        debug_print("All processing complete, stopping writer thread")
        stop_event.set()
        writer.join()

        successful = sum(1 for r in results if r)
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
    mp.set_start_method('spawn', force=True)  # Handling GPU issue

    parser = argparse.ArgumentParser(
        description="Retrieve top-k sentences with per-claim document stores"
    )
    parser.add_argument(
        "-k",
        "--knowledge_store_dir",
        type=str,
        default="data_store/knowledge_store",
        help="Path to knowledge store directory"
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default="data_store/hyde_fc.json",
        help="File containing claims to process"
    )
    parser.add_argument(
        "-o",
        "--json_output",
        type=str,
        default="data_store/dev_retrieval_top_k.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--top_k",
        default=5000,
        type=int,
        help="Number of documents to retrieve per claim"
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Starting index of examples to process"
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="Ending index of examples to process"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output"
    )

    args = parser.parse_args()

    # Set debug mode based on argument
    DEBUG = args.debug

    main(args)