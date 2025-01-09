import os
import shutil
import subprocess
import sys
import time
import zipfile

# Function to install a package using pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])



try:
    import requests
except ImportError:
    print("Requests library not found. Installing...")
    install_package("requests")
    import requests

# Try to import the tqdm module and install it if not present
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Installing...")
    install_package("tqdm")
    from tqdm import tqdm

script_dir = os.getcwd()
install_dir = os.path.join(script_dir, "installer_files")
cache_dir = os.path.join(script_dir, "installer_cache")


def download_file(url, output_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} to download {url}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                    desc=os.path.basename(output_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

            # Check if download was successful
            if os.path.getsize(output_path) != total_size:
                raise RuntimeError("Download failed or incomplete.")

            return response.status_code

        except Exception as e:
            print(f"Error downloading: {str(e)}")
            if attempt < max_retries:
                print("Retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

    # Check if download was successful
    if os.path.getsize(output_path) != total_size:
        os.remove(output_path)
        raise RuntimeError("Download failed or incomplete.")

    return response.status_code

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_qdrant():

    qdrant_dir = os.path.join(install_dir, 'qdrant')
    snapshot_name = 'prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot'


    filename = 'qdrant-x86_64-pc-windows-msvc.zip'
    zip_path = os.path.join(cache_dir, filename)
    install_path = os.path.join(install_dir, filename)
    if not os.path.exists(zip_path):
        if not os.path.exists(os.path.join(install_dir, 'qdrant_loaded.txt')):
            print("Download Qdrant")
            url = 'https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-pc-windows-msvc.zip'
            status_code = download_file(url, install_path)
            if status_code != 200:
                print("\033[101;93m Error: Failed to download qdrant-x86_64-pc-windows-msvc.zip HTTP Status Code: {} \033[0m".format(status_code))
                input("Press Enter to exit...")
                exit(1)
    else:
        shutil.copy(zip_path, install_path)

    filename = 'dist-qdrant.zip'
    zip_path = os.path.join(cache_dir, filename)
    install_path = os.path.join(install_dir, filename)
    if not os.path.exists(zip_path):
        if not os.path.exists(os.path.join(install_dir, 'qdrant_loaded.txt')):
            print("Download Qdrant Web UI")
            url = 'https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.33/dist-qdrant.zip'
            status_code = download_file(url, install_path)
            if status_code != 200:
                print("\033[101;93m Error: Failed to download dist-qdrant.zip HTTP Status Code: {} \033[0m".format(status_code))
                input("Press Enter to exit...")
                exit(1)
    else:
        shutil.copy(zip_path, install_path)

    filename = 'data.zip'
    zip_path = os.path.join(cache_dir, filename)
    install_path = os.path.join(install_dir, filename)
    if not os.path.exists(zip_path):
        if not os.path.exists(os.path.join(install_dir, 'qdrant_loaded.txt')):
            print("Download Prompt Quill data")
            url = 'https://civitai.com/api/download/models/567736'
            status_code = download_file(url, install_path)
            if status_code != 200:
                print("\033[101;93m Error: Failed to download prompt quill data HTTP Status Code: {} \033[0m".format(status_code))
                input("Press Enter to exit...")
                exit(1)
    else:
        shutil.copy(zip_path, install_path)


    # Assuming the qdrant executable zip and data zip paths are known and set similarly
    qdrant_exe_zip = os.path.join(install_dir, 'qdrant-x86_64-pc-windows-msvc.zip')
    qdrand_dist_zip = os.path.join(install_dir, 'dist-qdrant.zip')
    data_zip = os.path.join(install_dir, 'data.zip')


    if not os.path.exists(os.path.join(qdrant_dir, 'qdrant.exe')):
        print("Extract Qdrant with unzip")
        extract_zip(qdrant_exe_zip, qdrant_dir)

    if not os.path.exists(os.path.join(qdrant_dir, 'dist')) and not os.path.exists(os.path.join(qdrant_dir, 'static')):
        print("Extract Qdrant web UI with unzip")
        extract_zip(qdrand_dist_zip, qdrant_dir)

    if not os.path.exists(os.path.join(qdrant_dir, 'static')):
        print("Rename the dist folder to static")
        os.rename(os.path.join(qdrant_dir, 'dist'), os.path.join(qdrant_dir, 'static'))

    if not os.path.exists(os.path.join(install_dir, 'delete_after_setup', snapshot_name)) and not os.path.exists(os.path.join(install_dir, 'qdrant_loaded.txt')):
        print("Extract data with unzip")
        extract_zip(data_zip, os.path.join(install_dir, 'delete_after_setup'))


download_qdrant()
