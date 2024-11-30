import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from dataloader import ToTensor_trace, Custom_Dataset
from net import create_hyperparameter_space, MLP, CNN
from trainer import trainer
from utils import perform_attacks, NTGE_fn, perform_attacks_ensemble


if __name__ == "__main__":
    train_models = False
    dataset = "ASCON"
    leakage = "ID"
    byte = 4
    num_epochs = 50
    total_num_models = 50  # Total number of models (25 MLP + 25 CNN)
    num_top_k_model = 10
    nb_traces_attacks = 1000


    if not os.path.exists('./Dataset/'):
        os.mkdir('./Dataset/')
    if not os.path.exists('./Result/'):
        os.mkdir('./Result/')

    
    root = "./Result/"
    save_root = root + f"{dataset}_MLP_CNN_ensemble_byte{byte}_{leakage}/"
    model_root = save_root+"models/"
    print("root:", root)
    print("save_time_path:", save_root)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    nb_attacks = 100
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256


    dataloadertrain = Custom_Dataset(root='./', dataset=dataset, leakage=leakage,
                                                transform=transforms.Compose([ToTensor_trace()]), byte = byte)
    dataloadertrain.choose_phase("train")
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")

    correct_key = dataloadertrain.correct_key
    X_attack = dataloadertrain.X_attack
    Y_attack = dataloadertrain.Y_attack
    plt_attack = dataloadertrain.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #Random Search
    print(f"train_models: {train_models}")
    print(f"nb_traces_attacks: {nb_traces_attacks}")

    ensemble_predictions = []
    total_attack_time = 0
    script_start_time = time.time()
    all_ind_GE = []

    for num_models in range(total_num_models):
        model_type = "cnn" if num_models < 25 else "mlp"
        print(f"Processing model {num_models + 1}/{total_num_models} - Type: {model_type.upper()}")

        if train_models:
            config = create_hyperparameter_space(model_type)
            np.save(model_root + f"model_configuration_{num_models}_{model_type}.npy", config)
            batch_size = config["batch_size"]
            num_workers = 2
            dataloaders = {
                "train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            }
            dataset_sizes = {"train": len(dataloadertrain), "val": len(dataloaderval)}

            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device)
            torch.save(model.state_dict(), model_root + f"model_{num_models}_{model_type}_byte{byte}.pth")
        else:
            config = np.load(model_root + f"model_configuration_{num_models}_{model_type}.npy", allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            model.load_state_dict(torch.load(model_root + f"model_{num_models}_{model_type}_byte{byte}.pth"))

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
            predictions_wo_softmax = model(attack_traces)
            predictions = F.softmax(predictions_wo_softmax, dim=1)
            predictions = predictions.cpu().detach().numpy()

        print(f"Predictions shape for model {num_models}: {predictions.shape}")
        ensemble_predictions.append(predictions)

        save_individual_ge = False
        if save_individual_ge:
            individual_GE, individual_key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, dataset=dataset,
                                            nb_attacks=nb_attacks, shuffle=True, leakage=leakage, byte=byte)
            individual_NTGE = NTGE_fn(individual_GE)
            np.save(model_root + f"/result_{num_models}_{model_type}_byte{byte}", {"GE": individual_GE, "NTGE": individual_NTGE})
        else:
            individual_result = np.load(model_root + f"/result_{num_models}_{model_type}_byte{byte}.npy", allow_pickle=True).item()
            individual_GE = individual_result["GE"]
            individual_NTGE = individual_result["NTGE"]
        all_ind_GE.append(individual_GE[-1])

    print(f"Number of models in ensemble: {len(ensemble_predictions)}")
    if len(ensemble_predictions) == 0:
        raise ValueError("No predictions were added to the ensemble. Check if models are being properly loaded and evaluated.")

    top_k_model_index = np.argsort(np.array(all_ind_GE))
    ensemble_predictions = np.array(ensemble_predictions)
    ensemble_predictions = ensemble_predictions[top_k_model_index[:num_top_k_model]]

    print(f"Shape of ensemble predictions: {ensemble_predictions.shape}")

    start_time = time.time()
    ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, ensemble_predictions, plt_attack, correct_key, dataset=dataset,
                                                     nb_attacks=nb_attacks, shuffle=True, leakage=leakage)
    end_time = time.time()
    ensemble_attack_time = end_time - start_time

    total_attack_time += ensemble_attack_time

    ensemble_NTGE = NTGE_fn(ensemble_GE)

    print("ensemble_GE", ensemble_GE)
    print("ensemble_NTGE", ensemble_NTGE)
    print(f"Time taken for perform_attacks_ensemble: {ensemble_attack_time:.4f} seconds")
    print(f"Total attack time: {total_attack_time:.2f} seconds")

    np.save(model_root + f"/result_ensemble_byte{byte}_NumModel_{num_top_k_model}", {"GE": ensemble_GE, "NTGE": ensemble_NTGE})

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(f"Total script runtime: {total_script_time:.2f} seconds")