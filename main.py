import argparse, yaml, os, json, glob
import torch
import train, metrics
from dataset import Dataset
import os.path as osp

import numpy as np

parser = argparse.ArgumentParser()
# Modifikasi help string untuk menyertakan GAT
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet, GAT', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
args = parser.parse_args()

# Pastikan path ke manifest.json benar, relatif terhadap lokasi eksekusi main.py
# Jika main.py ada di root proyek, maka 'Dataset/manifest.json' sudah benar.
# Jika tidak, Anda mungkin perlu osp.join(osp.dirname(osp.abspath(__file__)), 'Dataset/manifest.json') atau path absolut.
# Untuk konsistensi dengan get_results.py, mari asumsikan 'Dataset' ada di root atau path yang dapat diakses.
manifest_file = osp.join('Dataset', 'manifest.json')
if not osp.exists(manifest_file):
    print(f"Error: {manifest_file} not found. Please ensure the path is correct.")
    exit()

with open(manifest_file, 'r') as f:
    manifest = json.load(f)

if args.task + '_train' not in manifest or (args.task + '_test' not in manifest and args.task != 'scarce') or ('full_test' not in manifest and args.task == 'scarce'):
    print(f"Error: Task '{args.task}' splits not found in {manifest_file}. Available keys: {list(manifest.keys())}")
    exit()
    
manifest_train = manifest[args.task + '_train']
test_dataset_names = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test'] # Ganti nama variabel
n = int(.1*len(manifest_train))
train_dataset_names = manifest_train[:-n] # Ganti nama variabel
val_dataset_names = manifest_train[-n:]   # Ganti nama variabel


# Periksa apakah dataset kosong sebelum memanggil Dataset()
if not train_dataset_names:
    print(f"Error: Training dataset for task '{args.task}' is empty.")
    exit()
if not val_dataset_names:
    print(f"Warning: Validation dataset for task '{args.task}' is empty. Validation might not be meaningful.")
    # Anda bisa memilih untuk exit() atau melanjutkan tanpa validasi yang valid
    # exit() 

train_dataset_list, coef_norm = Dataset(train_dataset_names, norm = True, sample = None) # Ganti nama variabel

# Handle jika val_dataset_names kosong
if val_dataset_names:
    val_dataset_list = Dataset(val_dataset_names, sample = None, coef_norm = coef_norm) # Ganti nama variabel
else:
    val_dataset_list = [] # Atau handle ini di fungsi train


# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

# Pastikan params.yaml ada di root atau path yang dapat diakses
params_file = 'params.yaml'
if not osp.exists(params_file):
    print(f"Error: {params_file} not found. Please ensure the path is correct.")
    exit()
with open(params_file, 'r') as f: # hyperparameters of the model
    hparams_all = yaml.safe_load(f)
    if args.model not in hparams_all:
        print(f"Error: Hyperparameters for model '{args.model}' not found in {params_file}.")
        exit()
    hparams = hparams_all[args.model]


# Asumsi semua model ada di subdirektori 'models'
from models.MLP import MLP # MLP digunakan sebagai encoder/decoder
models_list = [] # Ganti nama variabel
for i in range(args.nmodel):
    # Pastikan hparams['encoder'] dan hparams['decoder'] ada untuk model yang dipilih
    if 'encoder' not in hparams or 'decoder' not in hparams:
        print(f"Error: Encoder/Decoder hyperparameters missing for model '{args.model}' in {params_file}.")
        exit()
        
    encoder = MLP(hparams['encoder'], batch_norm = False) # Umumnya encoder/decoder tidak pakai BN di sini
    decoder = MLP(hparams['decoder'], batch_norm = False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'MLP':
        from models.NN import NN # NN menggunakan MLP sebagai arsitektur utama
        model = NN(hparams, encoder, decoder)

    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)
        
    elif args.model == 'GAT': # Ditambahkan
        from models.GAT import GAT # Pastikan path import benar
        model = GAT(hparams, encoder, decoder)   

    else: # Tambahkan penanganan jika model tidak dikenal
        print(f"Error: Model '{args.model}' is not recognized. Choose from MLP, GraphSAGE, PointNet, GUNet, GAT.")
        exit()

    
    log_path = osp.join('metrics', args.task, args.model) # path where you want to save log and figures    
    # Pastikan train_dataset_list dan val_dataset_list diteruskan
    model = train.main(device, train_dataset_list, val_dataset_list, model, hparams, log_path, 
                criterion = 'MSE_weighted', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True)
    models_list.append(model)

# Buat direktori jika belum ada sebelum menyimpan model
model_save_dir = osp.join('metrics', args.task, args.model)
os.makedirs(model_save_dir, exist_ok=True)
torch.save(models_list, osp.join(model_save_dir, args.model)) # Simpan list of models

if bool(args.score):
    # Pastikan direktori skor ada
    score_task_dir = osp.join('scores', args.task)
    os.makedirs(score_task_dir, exist_ok=True)
    
    s_test_split = args.task + '_test' if args.task != 'scarce' else 'full_test' # Ganti nama variabel
    
    # metrics.Results_test mengharapkan list of lists model, dan list hparams
    # models_list sudah merupakan list model dari beberapa run (jika nmodel > 1)
    # Kita perlu membungkusnya lagi dalam list untuk Results_test
    # dan hparams juga perlu dalam list
    
    # Periksa apakah test_dataset_names ada di manifest
    if s_test_split not in manifest:
        print(f"Error: Test split '{s_test_split}' for task '{args.task}' not found in manifest.json for scoring.")
    else:
        print(f"Running scoring for task {args.task} on test split {s_test_split}...")
        try:
            # Sesuai dengan perubahan di get_results.py, metrics.Results_test
            # mengharapkan loaded_models_outer_list: [model_type_idx][run_idx]
            # Di sini, kita hanya punya satu tipe model (args.model) dengan beberapa run (models_list)
            # Jadi, kita bungkus models_list dalam list.
            # hparams juga perlu jadi list of hparams.
            coefs = metrics.Results_test(
                device, 
                [models_list], # [[run1_model, run2_model, ...]]
                [hparams],     # [hparams_for_this_model_type]
                coef_norm, 
                path_in='Dataset', # path ke dir Dataset
                path_out=osp.join('scores', args.task), # path ke dir skor untuk task ini
                model_names_list=[args.model], # Nama model untuk penamaan file
                n_test=3, 
                criterion='MSE', 
                s=s_test_split
            )
            
            # Unpack hasil dan simpan
            (true_coefs_np, pred_coefs_mean_np, pred_coefs_std_np,
             true_surf_coefs_np_obj, pred_surf_coefs_np_obj,
             true_bls_np_obj, pred_bls_np_obj) = coefs

            np.save(osp.join(score_task_dir, 'true_force_coeffs.npy'), true_coefs_np)
            np.save(osp.join(score_task_dir, 'pred_force_coeffs_mean_over_runs.npy'), pred_coefs_mean_np)
            np.save(osp.join(score_task_dir, 'pred_force_coeffs_std_over_runs.npy'), pred_coefs_std_np)
            
            np.save(osp.join(score_task_dir, 'true_surf_coeffs_selected_cases.npy'), true_surf_coefs_np_obj)
            np.save(osp.join(score_task_dir, 'pred_surf_coeffs_selected_cases_mean_runs.npy'), pred_surf_coefs_np_obj)
            np.save(osp.join(score_task_dir, 'true_boundary_layers_selected_cases.npy'), true_bls_np_obj)
            np.save(osp.join(score_task_dir, 'pred_boundary_layers_selected_cases_mean_runs.npy'), pred_bls_np_obj)
            print(f"Scoring results saved in {score_task_dir}")

        except Exception as e:
            print(f"Error during scoring for task {args.task}: {e}")
            import traceback
            traceback.print_exc()