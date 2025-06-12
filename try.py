import os
import requests
import hashlib
import scipy.io

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return
    
    print(f"Downloading {url} to {save_path}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download file. Status code: {response.status_code}")
        print(response.text)
        return
    
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download completed.")

def calculate_md5(file_path, block_size=2**20):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

def check_md5(file_path, expected_md5):
    calculated_md5 = calculate_md5(file_path)
    if calculated_md5 == expected_md5:
        print(f"MD5 check passed: {calculated_md5}")
        return True
    else:
        print(f"MD5 check failed: expected {expected_md5}, got {calculated_md5}")
        return False

def load_and_check_mat_file(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        print("File loaded successfully.")
        print("Keys in the .mat file:", mat.keys())
        return mat
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None

def prepare_dataset():
    url = "https://code.aliyun.com/dataset/svhn/raw/master/train_32x32.mat"
    save_path = "/data_disk/dyy/python_projects/federated_diffusion/data/train_32x32.mat"
    expected_md5 = "e26dedcc434d2e4c54c9b2d4a06d8373"
    
    download_file(url, save_path)
    
    if os.path.exists(save_path):
        if check_md5(save_path, expected_md5):
            mat_data = load_and_check_mat_file(save_path)
            if mat_data is not None:
                # 进一步处理数据
                pass
        else:
            print("MD5 check failed. Please re-download the file.")
    else:
        print("File does not exist after download.")

if __name__ == "__main__":
    prepare_dataset()