"""
Download and Extract LibriSpeech Dataset
==============================================
Downloads dev-clean-2.tar.gz or dev-clean.tar.gz from OpenSLR.
"""

import os
import tarfile
import urllib.request
import argparse
from pathlib import Path

DATA_DIR = Path("data")

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        print(f"\rDownloading: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB of {total_size / (1024*1024):.1f} MB)", end="")
    else:
        print(f"\rDownloading: {read_so_far / (1024*1024):.1f} MB", end="")

def main():
    parser = argparse.ArgumentParser(description="Download and Extract LibriSpeech Dataset")
    parser.add_argument(
        "--dataset", type=str, choices=["dev-clean-2", "dev-clean"], default="dev-clean-2",
        help="Which LibriSpeech split to download: 'dev-clean-2' (~15MB, 26 speakers) or 'dev-clean' (~337MB, 40 speakers)",
    )
    args = parser.parse_args()
    
    if args.dataset == "dev-clean-2":
        url = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
        tar_name = "dev-clean-2.tar.gz"
        extracted_name = "dev-clean-2"
        target_name = "dev_clean_2"
    else:
        url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        tar_name = "dev-clean.tar.gz"
        extracted_name = "dev-clean"
        target_name = "dev_clean"
        
    tar_path = DATA_DIR / tar_name
    target_path = DATA_DIR / target_name
    extract_path = DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if target_path.exists():
        print(f"Dataset already exists at: {target_path}")
        return
        
    # Download
    if not tar_path.exists():
        print(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, tar_path, download_progress)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nFailed to download dataset: {e}")
            return
    else:
        print(f"Archive already downloaded at: {tar_path}")
        
    # Extract
    print("Extracting archive...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract archive: {e}")
        return
        
    # Rename LibriSpeech/extracted_name to target_name
    extracted_folder = DATA_DIR / "LibriSpeech" / extracted_name
    if extracted_folder.exists():
        print(f"Moving {extracted_folder} to {target_path}...")
        import shutil
        import time
        
        if target_path.exists():
            try:
                shutil.rmtree(target_path)
            except Exception as e:
                print(f"Warning: Could not remove existing target path: {e}")
                
        # Retry loop for Windows file locks
        for attempt in range(5):
            try:
                shutil.move(str(extracted_folder), str(target_path))
                break
            except Exception as e:
                if attempt == 4:
                    raise e
                print(f"Move attempt {attempt+1} failed ({e}). Retrying in 1s...")
                time.sleep(1)
        
        # Clean up LibriSpeech parent folder if empty
        try:
            (DATA_DIR / "LibriSpeech").rmdir()
        except OSError:
            pass
            
    # Clean up archive
    if tar_path.exists():
        print("Cleaning up archive file...")
        tar_path.unlink()
        
    print("Dataset preparation complete.")
    print(f"Speakers folder path: {target_path.absolute()}")

if __name__ == "__main__":
    main()
