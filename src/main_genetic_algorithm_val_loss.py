import argparse
import os
import random
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from dataloader import ToTensor_trace, Custom_Dataset
from net import create_hyperparameter_space, MLP, CNN
from trainer import trainer
from utils import perform_attacks, NTGE_fn, perform_attacks_ensemble
from tqdm import tqdm

def evaluate_loss(dataloadertest, model, device, verbose=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloadertest):
            if verbose and batch_idx == 0:
                print(f"First batch shape: data = {data.shape}, target = {target.shape}")
            if len(data) == 0 or len(target) == 0:
                if verbose:
                    print(f"Skipping empty batch {batch_idx}")
                continue
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            total_batches += 1
            data.detach()
            target.detach()
    if total_batches == 0:
        if verbose:
            print("No valid batches found!")
        return float('inf')  # Return a large loss if no valid batches
    loss /= total_batches
    return loss.detach().cpu().numpy()

def evaluate_ensemble(dataloaderval, models, device, ensemble, verbose=False, byte= 2):
    total_loss = 0
    for i in tqdm(ensemble):
        model_loss = evaluate_loss(dataloaderval, models[i], device, verbose)
        total_loss += model_loss
    avg_loss = total_loss / len(ensemble)
    print("avg_loss:", avg_loss)
    return avg_loss

def tournament_selection(population, scores, tournament_size=3):
    selected_indices = random.sample(range(len(population)), tournament_size)
    winner_index = min(selected_indices, key=lambda i: scores[i])
    return population[winner_index]

def crossover(parent1, parent2, total_num_models, num_top_k_model):
    crossover_point = random.randint(1, num_top_k_model - 1)
    child1 = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [x for x in parent1 if x not in parent2[:crossover_point]]
    
    # Ensure children have the correct length
    child1 = child1[:num_top_k_model]
    child2 = child2[:num_top_k_model]
    
    # If children are too short, add random models
    while len(child1) < num_top_k_model:
        new_model = random.randint(0, total_num_models - 1)
        if new_model not in child1:
            child1.append(new_model)
    while len(child2) < num_top_k_model:
        new_model = random.randint(0, total_num_models - 1)
        if new_model not in child2:
            child2.append(new_model)
    
    return child1, child2

def mutation(ensemble, total_num_models, mutation_rate):
    mutated = ensemble.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            new_model = random.randint(0, total_num_models - 1)
            while new_model in mutated:
                new_model = random.randint(0, total_num_models - 1)
            mutated[i] = new_model
    return mutated

def evolutionary_algorithm(dataloaderval, models, device, 
                           total_num_models=50, generations=5, population_size=30, num_top_k_model=10, 
                           mutation_rate=0.1, crossover_rate=0.9, byte = 2):

    print("\nGenetic Algorithm Hyperparameters:")
    print(f"Total number of models: {total_num_models}")
    print(f"Number of generations: {generations}")
    print(f"Population size: {population_size}")
    print(f"Number of models in each ensemble (num_top_k_model): {num_top_k_model}")
    print(f"Initial mutation rate: {mutation_rate}")
    print(f"Crossover rate: {crossover_rate}")
    print("\n" + "="*50 + "\n")

    # Initialize population
    population = []
    for _ in range(population_size):
        ensemble = random.sample(range(total_num_models), num_top_k_model)
        population.append(ensemble)

    best_ensemble_overall = None
    best_loss_overall = float('inf')

    for generation in range(generations):
        scores = []
        for ensemble in population:
            loss = evaluate_ensemble(dataloaderval, models, device, ensemble, byte)
            scores.append(loss)

        current_best_index = np.argmin(scores)
        current_best_ensemble = population[current_best_index]
        current_best_loss = scores[current_best_index]

        if current_best_loss < best_loss_overall:
            best_loss_overall = current_best_loss
            best_ensemble_overall = current_best_ensemble.copy()

        print(f"Generation {generation + 1}:")
        print(f"  Best Validation loss: {current_best_loss:.4f}")
        print(f"  Population Diversity: {len(set(tuple(ind) for ind in population))}/{population_size}")
        print(f"  Score Distribution: Min = {min(scores):.4f}, Max = {max(scores):.4f}, Avg = {np.mean(scores):.4f}")
        print(f"  Best Ensemble of This Generation: {current_best_ensemble}")

        # Create new population
        new_population = [current_best_ensemble]  # Elitism: keep the best ensemble of this generation

        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                parent1 = tournament_selection(population, scores)
                parent2 = tournament_selection(population, scores)
                child1, child2 = crossover(parent1, parent2, total_num_models, num_top_k_model)
                new_population.extend([child1, child2])
            else:
                new_population.append(tournament_selection(population, scores))

        # Trim population if it's too large
        new_population = new_population[:population_size]

        # Apply mutation
        for i in range(1, len(new_population)):  # Skip the first (best) ensemble
            if random.random() < mutation_rate:
                new_population[i] = mutation(new_population[i], total_num_models, mutation_rate)

        population = new_population

        diversity = len(set(tuple(ind) for ind in population)) / population_size
        adaptive_mutation_rate = mutation_rate * (1 + (1 - diversity))
        adaptive_crossover_rate = crossover_rate * diversity

        print(f"  Adaptive Mutation Rate: {adaptive_mutation_rate:.4f}")
        print(f"  Adaptive Crossover Rate: {adaptive_crossover_rate:.4f}")
        print("\n" + "-"*30 + "\n")

    print("\nEvolution completed!")
    print(f"Best overall ensemble based on Validation loss: {best_ensemble_overall}")
    print(f"Best overall Validation Loss: {best_loss_overall:.4f}")

    return best_ensemble_overall, best_loss_overall

if __name__ == "__main__":
    train_models = False
    dataset = "ASCON"
    model_type = "mlp" #mlp, cnn
    leakage = "ID" #ID, HW
   # ge_objective_fn = "val_loss"
    byte = 2
    num_epochs = 50
    total_num_models = 50
    num_top_k_model = 10
    nb_traces_attacks = 2000
    if not os.path.exists('./Dataset/'):
        os.mkdir('./Dataset/')
    if not os.path.exists('./Result/'):
        os.mkdir('./Result/')

    root = "./Result/"
    save_root = root+dataset+"_"+model_type+ "_byte"+str(byte)+"_"+leakage+"/"
    model_root = save_root+"models/"
    print("root:", root)
    print("save_time_path:", save_root)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_root):
        os.mkdir(model_root)

    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nb_attacks = 100
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256

    # Data loading
    dataloadertrain = Custom_Dataset(root='./', dataset=dataset, leakage=leakage,
                                    transform=transforms.Compose([ToTensor_trace()]), byte=byte)
    dataloadertrain.choose_phase("train")
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")

    # Add these lines for validation dataset checks
    print(f"Validation dataset size: {len(dataloaderval)}")
    if len(dataloaderval) == 0:
        raise ValueError("Validation dataset is empty!")

    # Determine batch size
    batch_size = min(32, len(dataloaderval))  # Ensure batch_size is not larger than dataset

    # Create DataLoader with the determined batch size
    dataloaderval = torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size, 
                                                shuffle=True, num_workers=2)

    print(f"Number of batches in validation loader: {len(dataloaderval)}")

    correct_key = dataloadertrain.correct_key
    X_attack = dataloadertrain.X_attack
    Y_attack = dataloadertrain.Y_attack
    plt_attack = dataloadertrain.plt_attack
    num_sample_pts = X_attack.shape[-1]
    ensemble_predictions = []
    models = []
    for num_models in range(total_num_models):
        if train_models:
            config = create_hyperparameter_space(model_type)
            np.save(model_root + "model_configuration_"+str(num_models)+".npy", config)
            batch_size = config["batch_size"]
            num_workers = 2
            dataloaders = {
                "train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            }
            dataset_sizes = {"train": len(dataloadertrain), "test": len(dataloadertest), "val": len(dataloaderval)}

            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device)
            torch.save(model.state_dict(), model_root + "model_"+str(num_models)+"_byte"+str(byte)+".pth")
        else:
            config = np.load(model_root + "model_configuration_"+str(num_models)+".npy", allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            model.load_state_dict(torch.load(model_root + "model_"+str(num_models)+"_byte"+str(byte)+".pth"))

        attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
        predictions_wo_softmax = model(attack_traces)
        predictions = F.softmax(predictions_wo_softmax, dim=1)
        predictions = predictions.cpu().detach().numpy()
        ensemble_predictions.append(predictions)
        models.append(model)

    start_time = time.time()

    best_ensemble, best_score = evolutionary_algorithm(dataloaderval, models, device, 
                                                       total_num_models, 
                                                       generations=5, population_size=30, num_top_k_model=num_top_k_model,
                                                       mutation_rate=0.1, crossover_rate=0.9, byte = byte)
    print(f"\nEvolutionary Algorithm Results:")
    print(f"Best ensemble: {best_ensemble}")
    print(f"Best Validation Loss: {best_score:.4f}")

    # Use the best ensemble for final prediction
    selected_predictions = [ensemble_predictions[i] for i in best_ensemble]
    ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, selected_predictions, plt_attack, correct_key, dataset=dataset,
                                                     nb_attacks=nb_attacks, shuffle=True, leakage=leakage, byte = byte)
    ensemble_NTGE = NTGE_fn(ensemble_GE)

    print("\nFinal Prediction Results:")
    print("Ensemble GE:", ensemble_GE)
    print(f"Ensemble NTGE: {ensemble_NTGE:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")

    # Save the final results with the new filename
    np.save(model_root + "gv." + leakage + ".npy",
            {
                "GE": ensemble_GE,
                "NTGE": ensemble_NTGE,
                "best_ensemble": best_ensemble,
                "best_score": best_score,
                "objective_function": "val_loss"
            })
    print("\nFinal results saved successfully.")

    # Load and print the saved results for verification
    saved_results = np.load(model_root + "gv." + leakage + ".npy", allow_pickle=True).item()
    print("\nVerification of Saved Results:")
    print(f"GE: {saved_results['GE']}")
    print(f"NTGE: {saved_results['NTGE']:.4f}")
    print(f"Best Ensemble: {saved_results['best_ensemble']}")
    print(f"Best Validation Loss: {saved_results['best_score']:.4f}")