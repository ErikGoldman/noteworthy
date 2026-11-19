"""Datasets to and from the project's Cloudflare R2 bucket, so a cloud GPU gets them at datacenter
speed instead of through a home upload link.

usage: r2_data.py push PATH [PATH...]     (paths relative to the repo root, e.g. data/seq516g_data)
       r2_data.py pull PATH [PATH...] [--into DIR]   (into the same paths under DIR, by default
                                                      this checkout's root)
       r2_data.py list
       r2_data.py files PATH [PATH...]   (each file R2 holds under PATH, and its size)

Each PATH is stored as tar shards of about SHARD_BYTES plus a manifest (PATH/manifest.json,
written last) giving each shard's SHA-256 (also beside each shard, KEY.sha256, once it is
complete). Symlinks are stored as the files they point to: the datasets link into data/raw by
absolute path, which another machine does not have. A shard's key is a hash of its members'
paths, sizes and mtimes, so a push that was cut off skips the shards already up, and a push of
unchanged data uploads nothing but the manifest. A pull downloads each shard whole and checks
its SHA-256 before unpacking it.

Credentials come from the repo's .env: OMR_R2_ACCOUNT_ID and OMR_R2_TOKEN, an R2 API token.
Cloudflare derives the S3 keys from such a token: the access key ID is the token's ID, the
secret its value's SHA-256 (developers.cloudflare.com/r2/api/tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "omr-data"
SHARD_BYTES = 512 << 20
# Part of every shard's key: a change to how shards are packed must not find the old ones "already up".
SHARD_FORMAT = "tar 1, symlinks dereferenced"
# Shards pulled at once, each in parallel ranged parts: enough to fill a datacenter link
# without a pile of half-downloaded shards on disk.
PULL_WORKERS = 3
# Multipart parts of 64 MB, four at a time: on a slow link a failed part costs little to resend.
TRANSFER = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=4)


def env() -> dict[str, str]:
    values = dict(os.environ)
    for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            values.setdefault(key.strip(), value.strip().strip("'\""))
    return values


def token_id(account: str, token: str) -> str:
    request = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{account}/tokens/verify",
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["result"]["id"]


def bucket_client():
    values = env()
    account, token = values["OMR_R2_ACCOUNT_ID"], values["OMR_R2_TOKEN"]
    return boto3.client("s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com", region_name="auto",
                        aws_access_key_id=token_id(account, token),
                        aws_secret_access_key=hashlib.sha256(token.encode()).hexdigest())


@dataclass
class Member:
    path: Path      # where it is found; a symlink is stored as its target (TarWriter dereferences)
    name: str       # its name in the tar, relative to the repo root
    size: int
    mtime: int


def members(top: Path) -> list[Member]:
    """The files under top (a directory or one file), each named by its path from the repo root."""
    paths = [ROOT / top] if (ROOT / top).is_file() else \
        [Path(d) / f for d, _, files in sorted(os.walk(ROOT / top, followlinks=True)) for f in sorted(files)]
    return [Member(p, str(p.relative_to(ROOT)), p.stat().st_size, int(p.stat().st_mtime)) for p in paths]


def shards(found: list[Member]) -> list[list[Member]]:
    out, current, size = [], [], 0
    for member in found:
        if current and size + member.size > SHARD_BYTES:
            out.append(current)
            current, size = [], 0
        current.append(member)
        size += member.size
    return out + [current] if current else out


def shard_key(top: Path, shard: list[Member]) -> str:
    listing = "\n".join([SHARD_FORMAT] + [f"{m.name}\t{m.size}\t{m.mtime}" for m in shard])
    identity = hashlib.sha256(listing.encode()).hexdigest()
    return f"{top}/shards/{identity[:24]}.tar"


class HashingPipe:
    """The read end of a pipe a tar is written into, hashing and counting what passes."""

    def __init__(self, fd: int):
        self.file, self.sha, self.size = os.fdopen(fd, "rb"), hashlib.sha256(), 0

    def read(self, n: int = -1) -> bytes:
        data = self.file.read(n)
        self.sha.update(data)
        self.size += len(data)
        return data


class TarWriter(threading.Thread):
    """Writes the shard's tar into a pipe; error holds what stopped it, if anything did."""

    def __init__(self, fd: int, shard: list[Member]):
        super().__init__()
        self.fd, self.shard, self.error = fd, shard, None

    def run(self) -> None:
        try:
            with os.fdopen(self.fd, "wb") as out, tarfile.open(fileobj=out, mode="w|", dereference=True) as tar:
                for member in self.shard:
                    tar.add(member.path, arcname=member.name, recursive=False)
        except Exception as error:  # noqa: BLE001 - raised again by upload_shard, in the main thread
            self.error = error


def upload_shard(client, key: str, shard: list[Member]) -> tuple[str, int]:
    """Streams the shard's tar to the bucket; its SHA-256 and size. Its record (KEY.sha256) is
    written only if the tar was written whole: a tar cut short still uploads as a valid object."""
    read_end, write_end = os.pipe()
    writer = TarWriter(write_end, shard)
    writer.start()
    pipe = HashingPipe(read_end)
    try:
        client.upload_fileobj(pipe, BUCKET, key, Config=TRANSFER)
    finally:
        pipe.file.close()  # a failed upload stops reading; closing lets the writer fail rather than block
        writer.join()
    if writer.error:
        raise writer.error
    sha = pipe.sha.hexdigest()
    client.put_object(Bucket=BUCKET, Key=key + ".sha256", Body=f"{sha} {pipe.size}".encode())
    return sha, pipe.size


def uploaded(client, key: str) -> tuple[str, int] | None:
    """The SHA-256 and size of a shard already in the bucket, or None. Its record (KEY.sha256) is
    written once the upload is complete, so a shard cut off part way counts as missing."""
    try:
        sha, size = client.get_object(Bucket=BUCKET, Key=key + ".sha256")["Body"].read().decode().split()
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise
    return sha, int(size)


def push(client, top: Path) -> None:
    planned = shards(members(top))
    total = sum(m.size for shard in planned for m in shard)
    print(f"{top}: {sum(map(len, planned))} files, {total / 1e9:.1f} GB, {len(planned)} shards", flush=True)
    manifest = []
    for i, shard in enumerate(planned):
        key = shard_key(top, shard)
        done = uploaded(client, key)
        sha, size = done or upload_shard(client, key, shard)
        manifest.append({"key": key, "sha256": sha, "bytes": size, "files": len(shard)})
        print(f"  shard {i + 1}/{len(planned)} {'already up' if done else 'uploaded'} ({size / 1e6:.0f} MB)", flush=True)
    client.put_object(Bucket=BUCKET, Key=f"{top}/manifest.json", Body=json.dumps({"shards": manifest}, indent=1).encode())
    print(f"{top}: manifest written", flush=True)


class CorruptShard(Exception):
    pass


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pull_shard(client, shard: dict, into: Path, staging: Path) -> None:
    """Downloads the shard into staging, then unpacks it under into once its SHA-256 matches.
    Unpacking straight from the stream timed the connection out: R2 drops a read left idle
    while tarfile writes tens of thousands of small files."""
    tar_path = staging / Path(shard["key"]).name
    try:
        client.download_file(BUCKET, shard["key"], str(tar_path), Config=TRANSFER)
        if file_sha256(tar_path) != shard["sha256"]:
            raise CorruptShard(f"{shard['key']}: SHA-256 mismatch; nothing unpacked, pull again")
        with tarfile.open(tar_path) as tar:
            tar.extractall(into, filter="data")
    finally:
        tar_path.unlink(missing_ok=True)


def pull(client, top: Path, into: Path) -> None:
    manifest = json.loads(client.get_object(Bucket=BUCKET, Key=f"{top}/manifest.json")["Body"].read())["shards"]
    total = sum(s["bytes"] for s in manifest)
    print(f"{top}: {len(manifest)} shards, {total / 1e9:.1f} GB", flush=True)
    into.mkdir(parents=True, exist_ok=True)
    # On into's filesystem, not /tmp: a cloud machine's root disk may not hold a few shards.
    with tempfile.TemporaryDirectory(dir=into, prefix=".r2_pull-") as staging, ThreadPoolExecutor(PULL_WORKERS) as pool:
        for i, _ in enumerate(pool.map(lambda shard: pull_shard(client, shard, into, Path(staging)), manifest)):
            print(f"  shard {i + 1}/{len(manifest)} unpacked", flush=True)


class RangedObject:
    """A bucket object read as a seekable file, each read a ranged GET: tarfile seeks past each
    member's data, so a shard's listing costs a request per member, not the shard's bytes."""

    def __init__(self, client, key: str):
        self.client, self.key, self.position = client, key, 0

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence != os.SEEK_SET:
            raise ValueError("only absolute seeks")
        self.position = offset
        return offset

    def tell(self) -> int:
        return self.position

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        end = "" if n < 0 else self.position + n - 1
        try:
            data = self.client.get_object(Bucket=BUCKET, Key=self.key, Range=f"bytes={self.position}-{end}")["Body"].read()
        except ClientError as error:
            if error.response["Error"]["Code"] == "InvalidRange":  # read at the object's end
                return b""
            raise
        self.position += len(data)
        return data


def stored_files(client, top: Path) -> dict[str, int]:
    """Each file R2 holds under top, by its name from the repo root, with its size."""
    manifest = json.loads(client.get_object(Bucket=BUCKET, Key=f"{top}/manifest.json")["Body"].read())["shards"]
    files = {}
    for shard in manifest:
        with tarfile.open(fileobj=RangedObject(client, shard["key"]), mode="r:") as tar:
            files.update((m.name, m.size) for m in tar if m.isfile())
    return files


def not_stored(client, top: Path) -> list[Member]:
    """The local files under top that R2 does not hold at the same size."""
    stored = stored_files(client, top)
    return [m for m in members(top) if stored.get(m.name) != m.size]


def list_datasets(client) -> None:
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for item in page.get("Contents", []):
            if item["Key"].endswith("/manifest.json"):
                shards_of = json.loads(client.get_object(Bucket=BUCKET, Key=item["Key"])["Body"].read())["shards"]
                print(f"{item['Key'][:-len('/manifest.json')]}: {sum(s['bytes'] for s in shards_of) / 1e9:.1f} GB, "
                      f"pushed {item['LastModified']:%Y-%m-%d %H:%M}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull", "list", "files"])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--into", type=Path, default=ROOT, help="pull: the root the paths are unpacked under")
    args = ap.parse_args()
    if args.action != "list" and not args.paths:
        ap.error(f"{args.action} needs at least one path")
    client = bucket_client()
    if args.action == "list":
        list_datasets(client)
    for top in args.paths:
        if args.action == "files":
            for name, size in sorted(stored_files(client, top).items()):
                print(f"{name}\t{size}")
        else:
            push(client, top) if args.action == "push" else pull(client, top, args.into)


if __name__ == "__main__":
    main()
