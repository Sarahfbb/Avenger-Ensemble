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

def evaluate_ensemble(dataloaderval, models, device, ensemble, ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn, byte):
    if gen_objective_fn == "ge_ntge":
        selected_predictions = [ensemble_predictions[i] for i in ensemble]
        ensemble_GE, _ = perform_attacks_ensemble(nb_traces_attacks, selected_predictions, plt_attack, correct_key, dataset=dataset,
                                                nb_attacks=nb_attacks, shuffle=True, leakage=leakage, byte = byte)
        ensemble_NTGE = NTGE_fn(ensemble_GE)
        score = ge_ntge_fn(ensemble_GE, ensemble_NTGE, nb_traces_attacks) 
        return score, ensemble_GE, ensemble_NTGE
    elif gen_objective_fn == "val_loss":
        for i in ensemble:
            score += evaluate_loss(dataloaderval, models[i], device)
        score = score/len(ensemble)
        return score, None, None


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
                           total_num_models=50, generations=10, population_size=30, num_top_k_model=10, 
                           mutation_rate=0.1, crossover_rate=0.9, gen_objective_fn = "ge_ntge",  byte = 2):
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

    # Evaluate individual model performances
    individual_performances = [evaluate_ensemble(dataloaderval, models, device, [i], ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn, byte =byte)
                               for i in range(total_num_models)]

    # Sort models by performance (ascending order, assuming lower score is better)
    # Ensures that we're sorting based on the first element of each tuple (the score)

    sorted_models = sorted(range(len(individual_performances)), key=lambda i: individual_performances[i][0])

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
            ind_score, ensemble_GE, ensemble_NTGE = evaluate_ensemble(dataloaderval, models, device, ensemble, ensemble_predictions, plt_attack, correct_key, dataset, nb_traces_attacks, nb_attacks, leakage, gen_objective_fn, byte = byte)
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
                np.save(model_root + f"/gvresult_best_ensemble_byte{byte}_gen{generations}_popsize{population_size}_num_model{num_top_k_model}", 
                        {
                            "val_loss": best_score,
                            "best_ensemble": best_ensemble
                        })
            elif gen_objective_fn == "ge_ntge":
                ensemble_GE = ensemble_GEs[current_best_index]
                ensemble_NTGE = ensemble_NTGEs[current_best_index]
                np.save(model_root + f"/ggresult_best_ensemble_byte{byte}_gen{generations}_popsize{population_size}_num_model{num_top_k_model}", 
                        {
                            "GE": ensemble_GE,
                            "NTGE": ensemble_NTGE,
                            "best_score": best_score,
                            "best_ensemble": best_ensemble
                        })

        
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
    train_models = False
    dataset = "ASCON"
    model_type = "mlp" #mlp, cnn
    leakage = "ID" #ID, HW
    gen_objective_fn = "ge_ntge"
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

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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

    correct_key = dataloadertrain.correct_key
    X_attack = dataloadertrain.X_attack
    Y_attack = dataloadertrain.Y_attack
    plt_attack = dataloadertrain.plt_attack
    num_sample_pts = X_attack.shape[-1]

    ensemble_predictions = []
    all_ind_GE = []
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

        #individual_GE, individual_key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, dataset=dataset,
        #                                     nb_attacks=nb_attacks, shuffle=True, leakage=leakage)
        # individual_NTGE = NTGE_fn(individual_GE)
        # all_ind_GE.append(individual_GE[-1])
        # print(f"Model {num_models} - GE: {individual_GE[-1]:.4f}, NTGE: {individual_NTGE}")

    # Evolutionary algorithm for ensemble selection
    start_time = time.time()
    best_ensemble, best_score = evolutionary_algorithm(dataloaderval, models, device, ensemble_predictions, plt_attack, correct_key, dataset, 
                                                   nb_traces_attacks, nb_attacks, leakage, total_num_models, 
                                                   generations=10, population_size=30, num_top_k_model=num_top_k_model,
                                                   mutation_rate=0.1, gen_objective_fn = gen_objective_fn,  byte =byte)
    

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")

    print(f"Best ensemble: {best_ensemble}")
    print(f"Best score: {best_score}")

    # # Use the best ensemble for final prediction
    # selected_predictions = [ensemble_predictions[i] for i in best_ensemble]

    # ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, selected_predictions, plt_attack, correct_key, dataset=dataset,
    #                                     nb_attacks=nb_attacks, shuffle=True, leakage=leakage,  byte =byte)

    # ensemble_NTGE = NTGE_fn(ensemble_GE)

    # print("ensemble_GE", ensemble_GE)
    # print("ensemble_NTGE", ensemble_NTGE)
    

    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"\nTotal running time: {total_time:.2f} seconds")
