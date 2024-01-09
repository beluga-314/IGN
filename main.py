import argparse
from autoencoder import Autoencoder
from SPNautoencoder import SPNAutoencoder
from utilities import initialize_weights, sampleimages, load_data, improveimages, calculatefid
from train import train_model
import torch
from torch import optim
import torch.nn as nn
import yaml
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Autoencoder Training and Evaluation")

    # YAML configuration file path
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML configuration file")

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load parameters from YAML file
    config = load_config(args.config_path)
    device = config['device']
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['train']:
        if config['fft']:
            os.makedirs(config['fft_model_save_dir'], exist_ok=True)
            os.makedirs(config['fft_generated_dir'], exist_ok=True)
        else:
            os.makedirs(config['model_save_dir'], exist_ok=True)
            os.makedirs(config['generated_dir'], exist_ok=True)

        data_loader = load_data(config['celeba'], config['batch_size'])
        # Set up model and optimizer
        if config['spectralnorm']:
            model = SPNAutoencoder().to(device)
        else:
            model = Autoencoder().to(device)
        model.apply(initialize_weights)
        model = nn.DataParallel(model)

        if config['spectralnorm']:
            model_copy = SPNAutoencoder().to(device)
        else:
            model_copy = Autoencoder().to(device)
        model_copy.apply(initialize_weights)
        model_copy = nn.DataParallel(model_copy)
        opt = optim.Adam(model.parameters(), lr=config['learningrate'], betas=config['betas'])

        # Train the autoencoder
        train_model(
            model,
            model_copy,
            opt,
            data_loader,
            config['n_epochs'],
            config['l_r'],
            config['l_i'],
            config['l_t'],
            config['l_gp'],
            config['alpha'],
            config['fft_generated_dir'],
            config['fft_model_save_dir'],
            config['generated_dir'],
            config['model_save_dir'],
            device,
            config['fft']
        )

    if config['sample']:
        os.makedirs(config['sampledir'], exist_ok=True)
        if config['spectralnorm']:
            model = SPNAutoencoder().to(device)
        else:
            model = Autoencoder().to(device)
        weights_path = config['checkpoint']
        checkpoint = torch.load(weights_path)
        if 'module' in list(checkpoint['state_dict'].keys())[0]:
            # Remove the 'module.' prefix from the keys
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        test_loader = load_data(config['celeba'], batch_size=config['no_of_samples'])
        sampleimages(model, test_loader, config['sampledir'], config['no_of_samples'], device, config['fft'])
        if config['degrade']:
            os.makedirs(config['degradedir'], exist_ok=True)
            improveimages(model, test_loader, config['degradedir'], device)

    if config['fid']:
        if config['spectralnorm']:
            model = SPNAutoencoder().to(device)
        else:
            model = Autoencoder().to(device)
        weights_path = config['checkpoint']
        checkpoint = torch.load(weights_path)
        if 'module' in list(checkpoint['state_dict'].keys())[0]:
            # Remove the 'module.' prefix from the keys
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        real_loader = load_data(config['celeba'], batch_size=100)
        calculatefid(model, config['gen_dir'], config['real_dir'], real_loader, config['samples_for_fid'], device, config['fft'])

if __name__ == "__main__":
    main()