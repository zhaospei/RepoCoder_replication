import os
import json
import shutil
from tqdm import tqdm

def calculate_rust_percentage(repo_path):
    total_files = 0
    rust_files = 0

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Exclude directories that start with a dot
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        for filename in filenames:
            total_files += 1
            if filename.endswith('.rs'):
                rust_files += 1

    # Calculate the percentage
    if total_files == 0:
        return 0, 0, 0  # Avoid division by zero
    rust_percentage = (rust_files / total_files) * 100
    return rust_percentage, rust_files, total_files

if __name__ == "__main__":
    storage_url = 'rust_repositories/repo_meta_data.json'
    repo_storage_url = 'rust_repositories'
    rust_benchmark_path = 'repositories/rust_line_level'
    with open(storage_url, "r") as f:
        repo_metadata = json.loads(f.read())
    repo_urls = [repo["html_url"] for repo in repo_metadata]
    cnt = 0
    repo_name_list = []
    for repo_url in repo_urls:
        owner, repo = repo_url.split("/")[-2:]
        repo_path = os.path.join(repo_storage_url, f"{owner}_{repo}")
        percentage, rust_files, total_files = calculate_rust_percentage(repo_path)
        if percentage >= 70 and total_files >= 50:
            cnt += 1
            print(f"{owner}_{repo}: {percentage:.2f}% {rust_files} {total_files}")
            repo_name_list.append(f"{owner}_{repo}")
            Path to copy the repository to
            dest_path = os.path.join(rust_benchmark_path, f"{owner}_{repo}")
            try:
                shutil.copytree(repo_path, dest_path)
                print(f"Copied {owner}_{repo} to {rust_benchmark_path}")
            except Exception as e:
                print(f"Error copying {repo_path} to {rust_benchmark_path}: {e}")

    print(repo_name_list)
    print(cnt)