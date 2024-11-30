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

def ge_ntge_fn(ensemble_GE, ensemble_NTGE, nb_traces_attacks):
    if ensemble_GE[-1] == 0:
        ge_ntge = ensemble_NTGE
    else:
        ge_ntge = ensemble_GE[-1] + nb_traces_attacks + 100
    return ge_ntge

def evaluate_loss(dataloadertest, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    # correct = 0
    with torch.no_grad():
        for data, target in dataloadertest:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            data.detach()
            target.detach()
    loss /= len(dataloadertest)
    # correct /= len(dataloadertest)
    return loss
def evaluate_ensemble(dataloaderval, models, device, ensemble, ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn):
    print(f"Ensemble: {ensemble}")
    print(f"Length of ensemble_predictions: {len(ensemble_predictions)}")
    
    if gen_objective_fn == "ge_ntge":
        try:
            selected_predictions = [ensemble_predictions[i] for i in ensemble if i < len(ensemble_predictions)]
            if not selected_predictions:
                print("Warning: No valid predictions selected. Returning worst possible score.")
                return float('inf'), None, None
            
            ensemble_GE, _ = perform_attacks_ensemble(nb_traces_attacks, selected_predictions, plt_attack, correct_key, dataset=dataset,
                                                    nb_attacks=nb_attacks, shuffle=True, leakage=leakage)
            ensemble_NTGE = NTGE_fn(ensemble_GE)
            score = ge_ntge_fn(ensemble_GE, ensemble_NTGE, nb_traces_attacks) 
            return score, ensemble_GE, ensemble_NTGE
        except IndexError as e:
            print(f"IndexError occurred. Ensemble: {ensemble}, len(ensemble_predictions): {len(ensemble_predictions)}")
            print("Returning worst possible score.")
            return float('inf'), None, None
    elif gen_objective_fn == "val_loss":
        score = 0
        valid_models = 0
        for i in ensemble:
            if i < len(models):
                score += evaluate_loss(dataloaderval, models[i], device)
                valid_models += 1
            else:
                print(f"Warning: Model index {i} is out of range. Max index: {len(models) - 1}")
        if valid_models == 0:
            print("Warning: No valid models in ensemble. Returning worst possible score.")
            return float('inf'), None, None
        score = score / valid_models
        return score, None, None
    else:
        raise ValueError(f"Unknown gen_objective_fn: {gen_objective_fn}")

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

def evolutionary_algorithm(dataloaderval, models, device, ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, 
                           total_num_models=50, generations=50, population_size=30, num_top_k_model=10, 
                           mutation_rate=0.1, crossover_rate=0.9, gen_objective_fn = "ge_ntge"):
    print("\nGenetic Algorithm Hyperparameters:")
    print(f"Total number of models: {total_num_models}")
    print(f"Number of generations: {generations}")
    print(f"Population size: {population_size}")
    print(f"Number of models in each ensemble (num_top_k_model): {num_top_k_model}")
    print(f"Initial mutation rate: {mutation_rate}")
    print(f"Initial crossover rate: {crossover_rate}")
    print(f"Dataset: {dataset}")
    print(f"Number of traces for attacks: {nb_traces_attacks}")
    print(f"Number of attacks: {nb_attacks}")
    print(f"Leakage model: {leakage}")
    print("\n" + "="*50 + "\n")

    print(f"Length of ensemble_predictions: {len(ensemble_predictions)}")
    print(f"Length of models: {len(models)}")

    if not models or not ensemble_predictions:
        raise ValueError("No valid models or predictions. Cannot proceed with the genetic algorithm.")

    # Adjust total_num_models if necessary
    total_num_models = min(total_num_models, len(models), len(ensemble_predictions))
    print(f"Adjusted total number of models: {total_num_models}")

    if total_num_models == 0:
        raise ValueError("No valid models available. Cannot proceed with the genetic algorithm.")

    # Evaluate individual model performances
    individual_performances = []
    for i in range(total_num_models):
        try:
            performance = evaluate_ensemble(dataloaderval, models, device, [i], ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn)
            individual_performances.append(performance)
        except Exception as e:
            print(f"Error evaluating model {i}: {str(e)}")

    # Sort models by performance (ascending order, assuming lower score is better)
    sorted_models = sorted(range(len(individual_performances)), key=lambda i: individual_performances[i][0])

    # Adjust num_top_k_model if necessary
    num_top_k_model = min(num_top_k_model, len(sorted_models))
    print(f"Adjusted num_top_k_model: {num_top_k_model}")

    # Initialize population randomly
    initial_population = []
    for _ in range(population_size):
        # Randomly select models for each ensemble
        ensemble = random.sample(range(total_num_models), num_top_k_model)
        initial_population.append(ensemble)

    population = initial_population.copy()
    best_ensemble = None
    best_score = float('inf')
    
    for generation in range(generations):
        # Evaluate fitness
        scores = []
        ensemble_GEs = []
        ensemble_NTGEs = []
        for ensemble in population:
            ind_score, ensemble_GE, ensemble_NTGE = evaluate_ensemble(dataloaderval, models, device, ensemble, ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn)
            scores.append(ind_score)
            ensemble_GEs.append(ensemble_GE)
            ensemble_NTGEs.append(ensemble_NTGE)

        # Update best ensemble
        current_best_index = np.argmin(scores)
        current_best_ensemble = population[current_best_index]
        current_best_score = scores[current_best_index]

        if current_best_score < best_score:
            best_score = current_best_score
            best_ensemble = current_best_ensemble.copy()
            if gen_objective_fn == "val_loss":
                np.save(model_root + f"/result_best_ensemble_byte{byte}_gen{generations}_popsize{population_size}_num_model{num_top_k_model}", 
                        {
                            "val_loss": best_score,
                            "best_ensemble": best_ensemble
                        })
            elif gen_objective_fn == "ge_ntge":
                ensemble_GE = ensemble_GEs[current_best_index]
                ensemble_NTGE = ensemble_NTGEs[current_best_index]
                if ensemble_GE is not None and ensemble_NTGE is not None:
                    np.save(model_root + f"/result_best_ensemble_byte{byte}_gen{generations}_popsize{population_size}_num_model{num_top_k_model}", 
                            {
                                "GE": ensemble_GE,
                                "NTGE": ensemble_NTGE,
                                "best_score": best_score,
                                "best_ensemble": best_ensemble
                            })
                else:
                    print("Warning: Best ensemble has null GE or NTGE. Not saving results.")

        
        print(f"Generation {generation + 1}:")
        print(f"  Best Score: {current_best_score:.4f}")
        print(f"  Population Diversity: {len(set(tuple(ind) for ind in population))}/{population_size}")
        print(f"  Score Distribution: Min = {min(scores):.4f}, Max = {max(scores):.4f}, Avg = {np.mean(scores):.4f}")
        print(f"  Best Ensemble: {current_best_ensemble}")
        
        # Create new population
        new_population = [best_ensemble]  # Elitism: keep the best ensemble ever found
        
        # Adaptive rates
        diversity = len(set(tuple(ind) for ind in population)) / population_size
        adaptive_mutation_rate = mutation_rate * (1 + (1 - diversity))
        adaptive_crossover_rate = crossover_rate * diversity
        
        while len(new_population) < population_size:
            if random.random() < adaptive_crossover_rate:
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
            if random.random() < adaptive_mutation_rate:
                new_population[i] = mutation(new_population[i], total_num_models, adaptive_mutation_rate)
        
        population = new_population
        
        print(f"  Adaptive Mutation Rate: {adaptive_mutation_rate:.4f}")
        print(f"  Adaptive Crossover Rate: {adaptive_crossover_rate:.4f}")
        print("\n" + "-"*30 + "\n")
    
    print("\nEvolution completed!")
    print(f"Best ensemble: {best_ensemble}")
    print(f"Best score: {best_score:.4f}")
    
    return best_ensemble, best_score

if __name__ == "__main__":
    train_models = False  # Set this to False since models are already trained
    dataset = "AES_HD_ext"
    leakage = "ID"
    gen_objective_fn = "ge_ntge"
    byte = 2
    num_epochs = 50  # This won't be used for loading, but keep it for consistency
    total_num_models = 50
    num_top_k_model = 10
    nb_traces_attacks = 1000

    # Setup directories
    root = "./Result/"
    save_root = root + f"{dataset}_MLP_CNN_ensemble_byte{byte}_{leakage}/"
    model_root = save_root + "models/"
    print("root:", root)
    print("save_root:", save_root)

    # Ensure directories exist
    for dir_path in [root, save_root, model_root]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Set random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up dataset parameters
    nb_attacks = 100
    classes = 9 if leakage == 'HW' else 256

    # Load datasets
    dataloadertrain = Custom_Dataset(root='./', dataset=dataset, leakage=leakage,
                                     transform=transforms.Compose([ToTensor_trace()]), byte=byte)
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

    ensemble_predictions = []
    models = []

    # Load models and generate predictions
    for num_models in range(total_num_models):
        model_type = "cnn" if num_models < 25 else "mlp"
        try:
            config_path = model_root + f"model_configuration_{num_models}_{model_type}.npy"
            model_path = model_root + f"model_{num_models}_{model_type}_byte{byte}.pth"
            
            if not os.path.exists(config_path):
                print(f"Configuration file not found: {config_path}")
                continue
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue
            
            config = np.load(config_path, allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()  # Set the model to evaluation mode

            attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
            with torch.no_grad():
                predictions_wo_softmax = model(attack_traces)
                predictions = F.softmax(predictions_wo_softmax, dim=1)
            predictions = predictions.cpu().numpy()
            ensemble_predictions.append(predictions)
            models.append(model)
            print(f"Loaded and processed model {num_models} ({model_type})")
        except Exception as e:
            print(f"Error loading model {num_models} ({model_type}): {str(e)}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"Loaded {len(models)} models in total")

    if not models or not ensemble_predictions:
        raise ValueError("No models were successfully loaded. Cannot proceed with the genetic algorithm.")

        # Run evolutionary algorithm
    start_time = time.time()
    best_ensemble, best_score = evolutionary_algorithm(dataloaderval, models, device, ensemble_predictions, plt_attack, correct_key, dataset, 
                                                        nb_traces_attacks, nb_attacks, leakage, total_num_models, 
                                                        generations=50, population_size=30, num_top_k_model=num_top_k_model,
                                                        mutation_rate=0.1, gen_objective_fn=gen_objective_fn)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")

    print(f"Best ensemble: {best_ensemble}")
    print(f"Best score: {best_score}")

    # Save the best ensemble results
    np.save(model_root + f"/ggfinal_best_ensemble_byte{byte}", {
        "best_ensemble": best_ensemble,
        "best_score": best_score,
        "runtime": total_time
    })


    # # Use the best ensemble for final prediction
    # selected_predictions = [ensemble_predictions[i] for i in best_ensemble]

    # ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, selected_predictions, plt_attack, correct_key, dataset=dataset,
    #                                     nb_attacks=nb_attacks, shuffle=True, leakage=leakage)

    # ensemble_NTGE = NTGE_fn(ensemble_GE)

    # print("ensemble_GE", ensemble_GE)
    # print("ensemble_NTGE", ensemble_NTGE)
    

    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"\nTotal running time: {total_time:.2f} seconds")