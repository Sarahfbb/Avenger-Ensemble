# Avenger-Ensemble

This repository developed for the paper [Genetic Algorithm-Driven Ensemble Selection for Deep Learning-based Side-Channel Analysis](https://sprint.iacr.org/2024/1949)

We proposed a new genetic algorithm-driven ensemble selection algorithm called Evolutionary Avenger Initiative (EAI) to generate a best-performing ensemble. To the best of our knowledge, this is the first work to investigate ensemble selection within the context of SCA.

## dataloader.py
The Custom_Dataset class is the heart of the data loading system, inheriting from PyTorch's Dataset class to handle various side-channel analysis datasets. When initialized, it takes parameters like the dataset type (Chipwhisperer, ASCAD, etc.), leakage model, and which byte to target. It's capable of handling multiple dataset formats and automatically applies the right preprocessing steps for each one.
Inside the class, apply_MinMaxScaler() gives us the option to normalize our data to a [0,1] range, which can be helpful for training stability. The choose_phase() function lets us switch between training, validation, and test data, making it easier to manage different stages of the machine learning pipeline.
The class also implements the standard PyTorch dataset requirements with __len__ and __getitem__, making it compatible with PyTorch's DataLoader for efficient batch processing.
For data conversion, there's a separate ToTensor_trace class that handles the conversion of our numpy arrays into PyTorch tensors, which is essential for working with PyTorch models.

## net.py
This file defines our neural network architectures. The MLP class creates a multi-layer perceptron network that's configurable through a search space dictionary. It can handle different numbers of layers, neurons, and activation functions based on the provided configuration.
The CNN class is more complex, implementing a convolutional neural network with both convolutional and fully connected layers. It includes batch normalization and pooling layers, with the architecture being highly configurable through the search space parameters.
Several helper functions support these classes. create_cnn_hp generates the CNN hyperparameters like kernel sizes and filter counts. weight_init handles different weight initialization strategies, and create_hyperparameter_space sets up the search spaces for both MLP and CNN architectures.

## utils.py
This file contains a wide array of utility functions for side-channel analysis. The attack-related functions are particularly important - rank_compute calculates how the key ranking evolves during an attack, while perform_attacks orchestrates the actual attack process using the model predictions.
For data processing, we have functions like calculate_HW for Hamming Weight calculations and generate_traces for creating simulated traces with different masking orders. There are also several dataset-specific loading functions (load_ascad, load_chipwhisperer, load_ascon_2, etc.) that handle the particularities of each dataset format.
The NTGE_fn function calculates a key metric in side-channel analysis - the Number of Traces needed to achieve a Guessing Entropy of zero, which helps us evaluate attack efficiency.

## trainer.py
The trainer module provides a complete training pipeline for our neural networks. Its main trainer function handles the entire training process, including setting up optimizers, managing the training loop, and evaluating performance. It supports both MLP and CNN models and includes features like learning rate scheduling and model checkpointing.
The training process alternates between training and validation phases, keeping track of losses and accuracies. It's designed to be flexible, allowing for different optimizers (Adam or RMSprop) and various training configurations.

## save.py
This is an example implementation file that shows how to put all the pieces together. It demonstrates setting up the dataset, configuring and training models, and running ensemble attacks. The script includes best practices for reproducibility (setting random seeds) and shows how to handle both CPU and GPU computation.
The script particularly focuses on ensemble methods, showing how to load multiple models, get their predictions, and combine them for more effective attacks. It also includes proper result saving and metric calculation, making it a good template for implementing similar systems.


## main
The genetic algorithm implementation in genetic_algorithm.py represents the core of the Evolutionary Avenger Initiative (EAI), which optimizes ensemble selection through evolutionary principles. The main function evolutionary_algorithm manages the entire process, starting with a random population of model ensembles and evolving them over multiple generations. It uses tournament selection (tournament_selection) to choose parent ensembles, combines them through crossover operations (crossover), and introduces variations through mutation (mutation). The algorithm features adaptive rates that adjust based on population diversity, with mutation rate increasing when diversity is low and crossover rate increasing when diversity is high. Two fitness functions are available: "ge_ntge" (combining Guessing Entropy and Number of Traces to achieve Guessing Entropy of 0) and "val_loss" (using validation loss). The code maintains the best-performing ensemble throughout evolution and includes elitism by preserving the best solution. Configuration parameters like population size, number of generations, mutation rate, and number of models per ensemble can be adjusted. The algorithm automatically saves progress and results, making it easy to track experiments and ensure reproducibility.

## Citation
If you find this code useful in your research, please consider citing:

```bibtex
@misc{cryptoeprint:2024/1949,
      author = {Zhao Minghui and Trevor Yap},
      title = {Avenger Ensemble: Genetic Algorithm-Driven Ensemble Selection for Deep Learning-based Side-Channel Analysis},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1949},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1949}
}
