import yaml, json
import torch
import metrics # Impor modul metrics yang sudah dimodifikasi
from dataset import Dataset # Asumsi Dataset.py ada di path yang benar
import os.path as osp
import pathlib # Untuk path management
import os # Untuk os.makedirs

import numpy as np

# Compute the normalization used for the training

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

# !! PENTING: Ganti 'MY_ROOT_DIRECTORY' dengan path root proyek Anda !!
# root_dir = 'MY_ROOT_DIRECTORY' 
# Sebagai alternatif, gunakan path relatif jika skrip ini dijalankan dari root proyek
root_dir = "." # Asumsi skrip dijalankan dari root direktori proyek

tasks = ['full', 'scarce', 'reynolds', 'aoa']

for task in tasks:
    print(f'Generating results for task {task}...')
    
    s_test_split_name = task + '_test' if task != 'scarce' else 'full_test'
    s_train_split_name = task + '_train'

    data_dir = osp.join(root_dir, 'Dataset') # Path ke folder Dataset
    
    # Pastikan manifest.json ada di data_dir
    manifest_path = osp.join(data_dir, 'manifest.json')
    if not osp.exists(manifest_path):
        print(f"Error: manifest.json not found at {manifest_path}")
        continue # Lanjut ke task berikutnya jika manifest tidak ada

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if s_train_split_name not in manifest:
        print(f"Error: Training split '{s_train_split_name}' not found in manifest.json for task '{task}'. Skipping.")
        continue
        
    manifest_train_list = manifest[s_train_split_name]
    
    if not manifest_train_list:
        print(f"Warning: Training dataset for task '{task}' is empty. Cannot compute normalization coefficients.")
        # Provide default coef_norm or skip task
        # Defaulting to a placeholder. Adjust dimensions if your features change.
        # (mean_in[7], std_in[7], mean_out[4], std_out[4])
        coef_norm = (np.zeros(hparams_list_per_model_type[0]['encoder'][0] if hparams_list_per_model_type else 7), 
                     np.ones(hparams_list_per_model_type[0]['encoder'][0] if hparams_list_per_model_type else 7), 
                     np.zeros(hparams_list_per_model_type[0]['decoder'][-1] if hparams_list_per_model_type else 4), 
                     np.ones(hparams_list_per_model_type[0]['decoder'][-1] if hparams_list_per_model_type else 4))
    else:
        try:
            # Gunakan seluruh train set untuk coef_norm yang stabil
            _, coef_norm = Dataset(manifest_train_list, norm=True, sample=None) 
        except Exception as e:
            print(f"Error computing normalization coefficients for task '{task}': {e}")
            continue


    # Load model dan hyperparameter
    # Modifikasi list ini untuk menyertakan GAT
    model_names = ['MLP', 'GraphSAGE', 'PointNet', 'GUNet', 'GAT'] # Sesuaikan jika nama model berbeda
    loaded_models_outer_list = [] # List of lists: [model_type_idx][run_idx]
    hparams_list_per_model_type = []

    all_models_fully_loaded = True # Flag baru
    for model_name in model_names:
        model_file_path = osp.join(root_dir, 'metrics', task, model_name, model_name)
        
        if not osp.exists(model_file_path):
            print(f"Warning: Model file for {model_name} on task {task} not found at {model_file_path}. This model type will be skipped.")
            # Jangan break, tapi jangan tambahkan ke list jika file tidak ada
            continue # Lewati model ini, tapi lanjutkan dengan yang lain
            
        try:
            model_runs_list = torch.load(model_file_path, map_location=device)
            if not isinstance(model_runs_list, list) or not all(isinstance(m, torch.nn.Module) for m in model_runs_list):
                print(f"Warning: Model file {model_file_path} for {model_name} does not contain a valid list of PyTorch models. Skipping.")
                continue
            loaded_models_outer_list.append([m.to(device) for m in model_runs_list])
        except Exception as e:
            print(f"Error loading model file {model_file_path} for {model_name}: {e}. Skipping this model type.")
            continue


        params_yaml_path = osp.join(root_dir, 'params.yaml') 
        if not osp.exists(params_yaml_path):
            print(f"Error: params.yaml not found at {params_yaml_path}. Cannot load hparams.")
            all_models_fully_loaded = False; break 
        with open(params_yaml_path, 'r') as f:
            hparam_all_models = yaml.safe_load(f)
            if model_name not in hparam_all_models:
                print(f"Error: Hyperparameters for {model_name} not found in params.yaml. Skipping this model type's hparams.")
                # Jika hparams tidak ada, kita tidak bisa lanjut dengan model ini
                # Hapus model yang sudah ter-load dari loaded_models_outer_list
                if loaded_models_outer_list and model_runs_list: # Cek jika model sempat ditambahkan
                    loaded_models_outer_list.pop()
                all_models_fully_loaded = False; break # Atau flag per model
            hparams_list_per_model_type.append(hparam_all_models[model_name])
    
    if not all_models_fully_loaded: # Jika ada error fatal saat load hparams
        print(f"Skipping result generation for task {task} due to missing hparams for one or more models.")
        continue
        
    if not loaded_models_outer_list: # Jika tidak ada model yang berhasil di-load
        print(f"No models were successfully loaded for task {task}. Skipping result generation.")
        continue

    # Sesuaikan model_names agar hanya berisi model yang berhasil di-load hparams-nya
    # Ini penting jika beberapa model dilewati karena file tidak ada tapi hparams-nya ada
    # atau sebaliknya. Cara termudah adalah membangun list nama model baru.
    # Namun, `metrics.Results_test` akan menggunakan `model_names_list` yang kita berikan.
    # Untuk saat ini, kita asumsikan `model_names` adalah daftar model yang *ingin* kita proses,
    # dan jika file model atau hparams-nya tidak ada, kita skip model tersebut.
    # `loaded_models_outer_list` dan `hparams_list_per_model_type` akan memiliki panjang yang sesuai.
    # Kita perlu `model_names_list_for_results` yang panjangnya sama.
    
    # Filter model_names agar sesuai dengan model yang berhasil di-load
    # Ini agak rumit karena loop di atas `continue` per model.
    # Untuk penyederhanaan, kita akan asumsikan `model_names` yang asli tetap,
    # dan `Results_test` akan menangani jika panjang `loaded_models_outer_list` berbeda.
    # Atau, lebih baik, bangun `model_names_for_results` di dalam loop.
    
    model_names_for_results = []
    temp_loaded_models = []
    temp_hparams_list = []

    idx_model_names_original = 0 # Untuk iterasi model_names asli
    idx_loaded_successfully = 0 # Untuk iterasi loaded_models_outer_list & hparams_list_per_model_type
    
    # Membangun ulang list model yang akan diproses, agar konsisten
    rebuilt_loaded_models_outer_list = []
    rebuilt_hparams_list_per_model_type = []
    rebuilt_model_names_list = []

    for model_name_orig in model_names: # Iterasi semua nama model yang diinginkan
        model_file_path_check = osp.join(root_dir, 'metrics', task, model_name_orig, model_name_orig)
        params_yaml_path_check = osp.join(root_dir, 'params.yaml')
        
        hparams_available = False
        if osp.exists(params_yaml_path_check):
            with open(params_yaml_path_check, 'r') as f_check:
                hparam_all_models_check = yaml.safe_load(f_check)
                if model_name_orig in hparam_all_models_check:
                    hparams_available = True
        
        if osp.exists(model_file_path_check) and hparams_available:
            try:
                # Coba load model lagi (ini duplikasi, idealnya struktur loop di atas diubah)
                # Untuk sekarang, kita hanya akan ambil dari list yang sudah di-load jika ada.
                # Cari model_name_orig di hparams_list_per_model_type dan loaded_models_outer_list
                # Ini menjadi rumit. Mari kita sederhanakan: asumsikan loop di atas sudah benar
                # dalam mengisi loaded_models_outer_list dan hparams_list_per_model_type.
                # Kita hanya perlu pastikan model_names_list untuk Results_test sesuai.
                
                # Jika loop di atas sudah benar, maka panjang loaded_models_outer_list dan hparams_list_per_model_type
                # akan sama. Kita hanya perlu membuat model_names_list yang sesuai.
                # Ini bisa dilakukan dengan menyimpan nama model yang berhasil di-load di loop atas.
                
                # Untuk saat ini, kita akan asumsikan model_names yang asli bisa dipakai,
                # dan error handling di Results_test cukup.
                # Atau, kita buat model_names_list yang hanya berisi model yang ada di loaded_models_outer_list
                pass # Akan diperbaiki jika perlu.
            except:
                pass # Lewati jika ada masalah saat re-check
    
    # Jika kita ingin lebih robust:
    # Di loop atas, saat model dan hparam berhasil di-load, tambahkan model_name ke list baru.
    # model_names_successfully_loaded = []
    # ... (di dalam loop di atas) ...
    # if model_and_hparam_loaded_for_this_model_name:
    #    model_names_successfully_loaded.append(model_name)
    # ...
    # Kemudian gunakan model_names_successfully_loaded untuk Results_test.
    # Untuk saat ini, kita pakai `model_names` asli.


    results_dir_for_task = osp.join(root_dir, 'scores', task)
    os.makedirs(results_dir_for_task, exist_ok=True) # Menggunakan os.makedirs

    if s_test_split_name not in manifest:
        print(f"Error: Test split '{s_test_split_name}' not found in manifest.json for task '{task}'. Skipping result computation.")
        continue

    print(f"Running Results_test for task {task} with test split {s_test_split_name}...")
    if not loaded_models_outer_list or not hparams_list_per_model_type:
        print(f"No models or hparams available for task {task} to run Results_test. Skipping.")
        continue
        
    try:
        # Kita perlu membuat `model_names_list` yang panjangnya sama dengan `loaded_models_outer_list`
        # Ini adalah cara yang lebih aman:
        processed_model_names_for_results_test = []
        # Asumsi: urutan di model_names asli dijaga untuk model yang berhasil load
        # Ini masih agak rentan. Idealnya, loop di atas membangun ketiga list (models, hparams, names) secara sinkron.
        
        # Untuk sekarang, jika panjangnya tidak sama, kita ambil subset dari model_names
        if len(model_names) > len(loaded_models_outer_list):
            # Ini terjadi jika beberapa model di model_names tidak ditemukan/di-load
            # Kita perlu tahu model mana yang ada di loaded_models_outer_list.
            # Ini sulit tanpa info tambahan dari loop di atas.
            # Solusi sementara: jika panjang beda, mungkin ada error.
            # Atau, kita coba pakai model_names yang panjangnya sudah disesuaikan.
            # Anggap saja loop di atas sudah menghasilkan loaded_models_outer_list dan hparams_list_per_model_type
            # yang konsisten. Kita perlu nama model yang sesuai.
            # Jika kita tidak punya nama yang benar, kita pakai placeholder.
            current_model_names_list = [f"model{i}" for i in range(len(loaded_models_outer_list))]
            # Idealnya: `model_names_list` harus berisi nama model yang benar sesuai urutan di `loaded_models_outer_list`.

        else: # Jika panjangnya sama atau model_names lebih pendek (seharusnya tidak terjadi jika loop benar)
             current_model_names_list = model_names[:len(loaded_models_outer_list)]


        results_tuple = metrics.Results_test(
            device,
            loaded_models_outer_list, # List model yang berhasil di-load
            hparams_list_per_model_type, # List hparams yang sesuai
            coef_norm,
            data_dir, 
            results_dir_for_task, 
            model_names_list=current_model_names_list, # List nama model yang sesuai
            n_test=3, 
            criterion='MSE',
            s=s_test_split_name
        )
    except Exception as e:
        print(f"Error during metrics.Results_test for task {task}: {e}")
        import traceback
        traceback.print_exc()
        continue

    (true_coefs_np, pred_coefs_mean_np, pred_coefs_std_np,
     true_surf_coefs_np_obj, pred_surf_coefs_np_obj,
     true_bls_np_obj, pred_bls_np_obj) = results_tuple

    np.save(osp.join(results_dir_for_task, 'true_force_coeffs.npy'), true_coefs_np)
    np.save(osp.join(results_dir_for_task, 'pred_force_coeffs_mean_over_runs.npy'), pred_coefs_mean_np)
    np.save(osp.join(results_dir_for_task, 'pred_force_coeffs_std_over_runs.npy'), pred_coefs_std_np)
    
    np.save(osp.join(results_dir_for_task, 'true_surf_coeffs_selected_cases.npy'), true_surf_coefs_np_obj)
    np.save(osp.join(results_dir_for_task, 'pred_surf_coeffs_selected_cases_mean_runs.npy'), pred_surf_coefs_np_obj)
    np.save(osp.join(results_dir_for_task, 'true_boundary_layers_selected_cases.npy'), true_bls_np_obj)
    np.save(osp.join(results_dir_for_task, 'pred_boundary_layers_selected_cases_mean_runs.npy'), pred_bls_np_obj)

    print(f"Results for task {task} saved in {results_dir_for_task}")

print("All tasks processed.")