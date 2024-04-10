import subprocess
import os
import json
import glob
from tqdm import tqdm
from pathlib import Path
import requests

def download_rvc_models(this_dir):
    folder = os.path.join(this_dir,'base_model')
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    files = {
        "hubert_base.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt",
        "rmvpe.onnx": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.onnx"
    }
    
    for filename, url in files.items():
        file_path = os.path.join(folder, filename)
    
        if not os.path.exists(file_path):
            print(f'File {filename} not found, start loading...')
    
            response = requests.get(url)
    
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'File {filename} successfully loaded.')
            else:
                print(f'f {filename}.')

def load_rmvpe(rmvpe_folder):
    os.makedirs(rmvpe_folder, exist_ok=True)
    rmvpe_file_path = os.path.join(rmvpe_folder, 'rmvpe.pt')

    if not os.path.isfile(rmvpe_file_path):
        print('rmvpe asset file not found. Downloading it from huggingface.')
        url = "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(rmvpe_file_path, 'wb') as file:
                file.write(response.content)
                print(f'File saved into {file.name}')
        else:
            raise f'Error {response.status_code} during download.'
    else:
        print('Loaded rmvpe asset file.')

def load_hubert_base(hubert_base_folder):
    os.makedirs(hubert_base_folder, exist_ok=True)
    hubert_base_file_path = os.path.join(hubert_base_folder, 'hubert_base.pt')

    if not os.path.isfile(hubert_base_file_path):
        print('hubert base asset file not found. Downloading it from huggingface.')
        url = "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(hubert_base_file_path, 'wb') as file:
                file.write(response.content)
                print(f'File saved into {file.name}')
        else:
            raise f'Error {response.status_code} during download.'
    else:
        print('Loaded hubert base asset file.')
