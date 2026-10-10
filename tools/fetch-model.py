"""Download one pinned Hugging Face snapshot into a model cache after checking that it fits on the disk.

run-sharar.sh and sharar-tp run it from the Python environment (which has huggingface_hub) for a model or draft the
target needs and the cache lacks:  python fetch-model.py REPO REVISION CACHE_HUB CACHE
Files already in the snapshot are not fetched again, so an interrupted download resumes.
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HEADROOM = 10 * 2**30  # left free after the download: logs, JIT caches, sessions


def main() -> int:
    repo, revision, cache, host_cache = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    missing = sum(s.size or 0 for s in info.siblings if not (snap / s.rfilename).is_file())
    free = shutil.disk_usage(cache).free
    if missing + HEADROOM > free:
        sys.exit(f"{repo}@{revision[:7]} needs {missing / 1e9:.1f} GB more, and {host_cache} has {free / 1e9:.1f} GB free "
                 f"(keeping {HEADROOM / 1e9:.0f} GB spare); free space, or point SHARAR_HF_CACHE/SHARAR_FLASHNEXT_CACHE at a larger disk")
    print(f"downloading {repo}@{revision[:7]} into {host_cache}: {missing / 1e9:.1f} GB", flush=True)
    snapshot_download(repo, revision=revision, cache_dir=cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
