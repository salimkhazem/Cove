"""Download the public verification datasets into the Hugging Face cache."""
import sys
from huggingface_hub import snapshot_download

REPOS = {
    "sop": "nyris/stanford-online-products-v1",
    "cub": "Donghyun99/CUB-200-2011",
    "inshop": "Marqo/deepfashion-inshop",
    "gldv2mini": "zguo0525/google-landmarks-v2-mini",
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(REPOS)
    for n in names:
        path = snapshot_download(REPOS[n], repo_type="dataset")
        print(n, path, flush=True)
