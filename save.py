import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from net import CNN  # Assuming the CNN class is in net.py
from dataloader import ToTensor_trace, Custom_Dataset
from utils import perform_attacks_ensemble, NTGE_fn

# Configuration
dataset = "ASCON"
byte = 2
leakage = "ID"
nb_traces_attacks = 1000
nb_attacks = 100
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Best ensemble configuration
best_ensemble = [9, 34, 25, 16, 38, 13, 48, 43, 31, 45]
generations = 10
population_size = 30
num_top_k_model = 10

# Path setup
root = "./Result/"
save_root = root + dataset + "_cnn_byte" + str(byte) + "_" + leakage + "/"
model_root = save_root + "models/"

# Data loading
dataloader = Custom_Dataset(root='./', dataset=dataset, leakage=leakage,
                          transform=transforms.Compose([ToTensor_trace()]), byte=byte)
dataloader.choose_phase("train")  # We need this to get the attack data and correct key

correct_key = dataloader.correct_key
X_attack = dataloader.X_attack
plt_attack = dataloader.plt_attack
num_sample_pts = X_attack.shape[-1]

# Load models and get predictions
ensemble_predictions = []
models = []

for model_idx in best_ensemble:
    # Load model configuration
    config = np.load(model_root + f"model_configuration_{model_idx}.npy", allow_pickle=True).item()
    
    # Create and load model
    model = CNN(config, num_sample_pts, 256).to(device)  # 256 classes for ID leakage
    model.load_state_dict(torch.load(model_root + f"model_{model_idx}_byte{byte}.pth"))
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
        predictions_wo_softmax = model(attack_traces)
        predictions = F.softmax(predictions_wo_softmax, dim=1)
        predictions = predictions.cpu().detach().numpy()
        ensemble_predictions.append(predictions)

# Perform attacks with ensemble
ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, ensemble_predictions, plt_attack, 
                                               correct_key, dataset=dataset, nb_attacks=nb_attacks, 
                                               shuffle=True, leakage=leakage, byte=byte)

# Calculate NTGE
ensemble_NTGE = NTGE_fn(ensemble_GE)

# Calculate best score (same as in original code)
if ensemble_GE[-1] == 0:
    best_score = ensemble_NTGE
else:
    best_score = ensemble_GE[-1] + nb_traces_attacks + 100

# Save results
result_dict = {
    "GE": ensemble_GE,
    "NTGE": ensemble_NTGE,
    "best_score": best_score,
    "best_ensemble": best_ensemble
}

save_path = model_root + f"/ggresult_best_ensemble_byte{byte}_gen{generations}_popsize{population_size}_num_model{num_top_k_model}.npy"
np.save(save_path, result_dict)

print(f"Results saved to: {save_path}")
print(f"Ensemble GE: {ensemble_GE}")
print(f"Ensemble NTGE: {ensemble_NTGE}")
print(f"Best Score: {best_score}")
print(f"Best Ensemble: {best_ensemble}")