"""
Materialize a bounded Stack-Edu subset into a local parquet with source text.

Stack-Edu's Hub dataset stores Software Heritage blob ids, not file contents.
This script downloads the metadata parquet shards from Hugging Face and fetches
the gzipped file bodies from Software Heritage's public S3 bucket.

Example:

python -m scripts.materialize_stackedu \
  --language Python \
  --target-bytes 8000000000 \
  --num-workers 64
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import time
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir


LANGUAGES = {
    "C",
    "CSharp",
    "Cpp",
    "Go",
    "Java",
    "JavaScript",
    "Markdown",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "SQL",
    "Shell",
    "Swift",
    "TypeScript",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize Stack-Edu source files from SWH S3")
    parser.add_argument("--language", type=str, default="Python", choices=sorted(LANGUAGES))
    parser.add_argument("--output", type=str, default=None, help="output parquet path")
    parser.add_argument("--target-bytes", type=int, default=8_000_000_000, help="decoded source bytes to materialize")
    parser.add_argument("--val-docs", type=int, default=4096, help="extra successful docs to reserve for validation")
    parser.add_argument("--max-docs", type=int, default=-1, help="cap successful train+val docs (-1 = unlimited)")
    parser.add_argument("--min-length-bytes", type=int, default=64)
    parser.add_argument("--max-length-bytes", type=int, default=65_536)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--allowed-license-types", type=str, default="", help="comma-separated license_type allowlist; empty = no filter")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--metadata-limit", type=int, default=-1, help="debug cap on metadata rows scanned")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download_metadata_shards(language, base_dir):
    shards_dir = os.path.join(base_dir, "task_data", "HuggingFaceTB--stack-edu", language, "train")
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/HuggingFaceTB/stack-edu/parquet/{language}/train"
                with urlopen_with_retry(listing_url, timeout=60, retries=8) as response:
                    shard_urls = json.loads(response.read())
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    path = os.path.join(shards_dir, filename)
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        print(f"Reusing metadata shard {path}", flush=True)
                        filenames.append(filename)
                        continue
                    print(f"Downloading metadata shard {shard_url} ...", flush=True)
                    tmp_path = path + ".tmp"
                    with urlopen_with_retry(shard_url, timeout=300, retries=10) as response:
                        with open(tmp_path, "wb") as f:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                    os.replace(tmp_path, path)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    return [os.path.join(shards_dir, filename) for filename in filenames]


def urlopen_with_retry(url, timeout, retries):
    last_error = None
    for attempt in range(retries + 1):
        try:
            headers = {"User-Agent": "nanochat-stackedu-prep/0.1"}
            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HF_KEY")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )
            if hf_token and "huggingface.co" in url:
                headers["Authorization"] = f"Bearer {hf_token}"
            request = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retry_after = exc.headers.get("Retry-After")
            if exc.code != 429 or attempt >= retries:
                raise
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(300, 15 * (2**attempt))
            print(f"HTTP 429 for {url}; retrying in {delay}s ({attempt + 1}/{retries})", flush=True)
            time.sleep(delay)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            delay = min(120, 5 * (2**attempt))
            print(f"URL error for {url}: {exc}; retrying in {delay}s ({attempt + 1}/{retries})", flush=True)
            time.sleep(delay)
    raise last_error


def decode_content(blob_id, encoding, timeout, retries):
    url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nanochat-stackedu-prep/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            raw = gzip.decompress(payload)
            enc = encoding or "utf-8"
            try:
                return raw.decode(enc)
            except (LookupError, UnicodeDecodeError):
                return raw.decode("utf-8", errors="replace")
        except (OSError, EOFError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.25 * (2**attempt))
    raise last_error


def fetch_row(row, timeout, retries):
    try:
        text = decode_content(row["blob_id"], row.get("src_encoding"), timeout, retries)
    except Exception:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or "\x00" in text:
        return None
    row = dict(row)
    row["text"] = text + "\n"
    return row


def write_batch(writer, rows):
    if not rows:
        return 0
    columns = {
        "text": [row["text"] for row in rows],
        "blob_id": [row["blob_id"] for row in rows],
        "language": [row.get("language") for row in rows],
        "repo_name": [row.get("repo_name") for row in rows],
        "path": [row.get("path") for row in rows],
        "src_encoding": [row.get("src_encoding") for row in rows],
        "length_bytes": [row.get("length_bytes") for row in rows],
        "score": [row.get("score") for row in rows],
        "int_score": [row.get("int_score") for row in rows],
        "license_type": [row.get("license_type") for row in rows],
    }
    table = pa.table(columns)
    writer.write_table(table)
    return table.num_rows


def iter_candidates(metadata_paths, args, allowed_license_types):
    scanned = 0
    emitted = 0
    columns = [
        "blob_id",
        "language",
        "repo_name",
        "path",
        "src_encoding",
        "length_bytes",
        "score",
        "int_score",
        "license_type",
    ]
    for path in metadata_paths:
        table = pq.read_table(path, columns=columns)
        for batch in table.to_batches(max_chunksize=8192):
            rows = batch.to_pylist()
            for row in rows:
                if args.metadata_limit > 0 and scanned >= args.metadata_limit:
                    return
                scanned += 1
                length = row["length_bytes"] or 0
                score = row["score"] or 0.0
                if length < args.min_length_bytes or length > args.max_length_bytes:
                    continue
                if score < args.min_score:
                    continue
                if allowed_license_types and row.get("license_type") not in allowed_license_types:
                    continue
                emitted += 1
                yield row
        del table


def main():
    args = parse_args()
    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or get_base_dir()
    output = args.output
    if output is None:
        output = os.path.join(
            base_dir,
            "task_data",
            "HuggingFaceTB--stack-edu",
            "materialized",
            args.language,
            f"stackedu_{args.language.lower()}_budget.parquet",
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    done_path = output + ".done.json"
    if os.path.exists(output) and os.path.exists(done_path) and not args.overwrite:
        print(f"Found existing materialized dataset: {output}", flush=True)
        return
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"{output} exists without {done_path}; pass --overwrite to replace it")

    tmp_output = output + ".tmp"
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    allowed_license_types = {
        item.strip() for item in args.allowed_license_types.split(",") if item.strip()
    }
    metadata_paths = download_metadata_shards(args.language, base_dir)
    target_success_docs = None if args.max_docs < 0 else args.max_docs
    target_train_bytes = args.target_bytes
    target_total_bytes = target_train_bytes

    schema = pa.schema([
        ("text", pa.string()),
        ("blob_id", pa.string()),
        ("language", pa.string()),
        ("repo_name", pa.string()),
        ("path", pa.string()),
        ("src_encoding", pa.string()),
        ("length_bytes", pa.int64()),
        ("score", pa.float64()),
        ("int_score", pa.int64()),
        ("license_type", pa.string()),
    ])

    writer = pq.ParquetWriter(tmp_output, schema=schema, compression="zstd")
    pending = []
    write_rows = []
    success_docs = 0
    failed_docs = 0
    decoded_bytes = 0
    train_bytes = 0
    started = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}

            def submit(row):
                future = executor.submit(fetch_row, row, args.timeout, args.retries)
                futures[future] = row

            for row in iter_candidates(metadata_paths, args, allowed_license_types):
                if target_success_docs is not None and success_docs >= target_success_docs:
                    break
                if success_docs >= args.val_docs and train_bytes >= target_train_bytes:
                    break
                submit(row)
                if len(futures) < args.num_workers * 4:
                    continue
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    futures.pop(future)
                    result = future.result()
                    if result is None:
                        failed_docs += 1
                        continue
                    success_docs += 1
                    decoded_bytes += len(result["text"].encode("utf-8"))
                    if success_docs > args.val_docs:
                        train_bytes += len(result["text"].encode("utf-8"))
                    write_rows.append(result)
                    if len(write_rows) >= args.batch_size:
                        write_batch(writer, write_rows)
                        write_rows.clear()
                if success_docs > 0 and success_docs % (args.batch_size * 4) == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"docs={success_docs:,} train_bytes={train_bytes:,}/{target_total_bytes:,} "
                        f"failed={failed_docs:,} rate={decoded_bytes/elapsed/1024/1024:.2f} MiB/s",
                        flush=True,
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    failed_docs += 1
                    continue
                if target_success_docs is not None and success_docs >= target_success_docs:
                    continue
                if success_docs >= args.val_docs and train_bytes >= target_train_bytes:
                    continue
                success_docs += 1
                decoded_bytes += len(result["text"].encode("utf-8"))
                if success_docs > args.val_docs:
                    train_bytes += len(result["text"].encode("utf-8"))
                write_rows.append(result)
                if len(write_rows) >= args.batch_size:
                    write_batch(writer, write_rows)
                    write_rows.clear()

        write_batch(writer, write_rows)
    finally:
        writer.close()

    os.replace(tmp_output, output)
    summary = {
        "language": args.language,
        "output": output,
        "success_docs": success_docs,
        "failed_docs": failed_docs,
        "decoded_bytes": decoded_bytes,
        "train_bytes_after_val_docs": train_bytes,
        "val_docs": min(args.val_docs, success_docs),
        "target_bytes": args.target_bytes,
        "elapsed_sec": time.time() - started,
        "args": vars(args),
    }
    with open(done_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
