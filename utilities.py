import torch
from torch.nn import init
import torch.fft as fft
from torch import nn
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
# from cleanfid import fid
from pytorch_fid import fid_score

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, mean=0, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)

def sample_from_fft(real_data):
    batch, channels, height, width = real_data.shape

    # Calculate FFT of real data
    f = fft.fft2(real_data)  # assuming height and width are the last two dimensions

    # Calculate mean and variance for real and imaginary parts
    mean_real = f.real.mean(dim=0)
    mean_imag = f.imag.mean(dim=0)
    std_real = f.real.std(dim=0)
    std_imag = f.imag.std(dim=0)

    # generate noise from mean and std of real and imaginary part
    freq_real = [torch.normal(mean_real, std_real) for _ in range(batch)]
    freq_real = torch.stack(freq_real, dim=0)
    freq_imag = [torch.normal(mean_imag, std_imag) for _ in range(batch)]
    freq_imag = torch.stack(freq_imag, dim=0)
    freq = torch.complex(freq_real, freq_imag)
    noise = fft.ifft2(freq)
    return noise.real

def add_noise(x, std_dev=0.15):
    noise = torch.randn_like(x) * std_dev
    return x + noise

def grayscale(x):
    return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

def generate_sketch(x):
    blurred = gaussian_blur(grayscale(x+1), kernel_size=21)
    return grayscale(x+1) / (blurred + 1e-10) - 1

def load_data(dir, batch_size):
    transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])
    # data_path = os.path.join(os.path.expanduser("~"), "../../mntnas/CelebA")
    celeba_dataset = ImageFolder(root=dir, transform=transform)
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

def sampleimages(model, test_loader, dir, num, device, fft=True):
    with torch.no_grad():
        if fft:
            x, _ = next(iter(test_loader))
            fixed_z = sample_from_fft(x).to(device)
        else:
            fixed_z = torch.randn(num, 3, 64, 64).to(device)
        f_fixed_z = model(fixed_z)
        f_f_fixed_z = model(f_fixed_z)

        # Inverse transformations before saving the grid
        inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        ])
        images_to_save = torch.cat([
            inverse_transform(fixed_z.cpu()),
            inverse_transform(f_fixed_z.cpu()),
            inverse_transform(f_f_fixed_z.cpu())
        ], dim=0)
        save_image(images_to_save, os.path.join(dir, f'grid.png'), nrow=num)

def calculatefid(model, gen_dir, real_dir, real_loader, num_samples, device, fft=True):
    batch_size = 100
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        total_real_images = min(len(real_loader.dataset), num_samples)
        num_batches_real = (total_real_images + batch_size - 1) // batch_size
        for i, real_batch in enumerate(tqdm(real_loader, desc="Saving Real Images")):
            if i == num_batches_real:
                break

            real_images = real_batch[0].cpu()
            inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            ])
            for j in range(min(batch_size, total_real_images - i * batch_size)):
                real_image_to_save = inverse_transform(real_images[j])
                save_image(real_image_to_save, os.path.join(real_dir, f'real_image_{i * batch_size + j + 1}.png'))

    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
        with torch.no_grad():
            for i in tqdm(range(num_samples // batch_size), desc="Generating and Saving Images"):
                if fft:
                    x, _ = next(iter(real_loader))
                    z = sample_from_fft(x).to(device)
                else:
                    z = torch.randn(batch_size, 3, 64, 64).to(device)
                generated_images = model(z)
                for j in range(batch_size):
                    inverse_transform = transforms.Compose([
                        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                    ])
                    image_to_save = inverse_transform(generated_images[j].cpu())
                    save_image(image_to_save, os.path.join(gen_dir, f'generated_image_{i * batch_size + j + 1}.png'))

    score = fid_score.calculate_fid_given_paths(['/home/rameshbabu/IGN/formal/realforfid', '/home/rameshbabu/IGN/formal/generatedforfid'], batch_size=10, device = 'cuda', dims=2048)
    print(f"FID Score: {score}")


            

def improveimages(model, test_loader, dir, device):
    # Load a batch of 5 images from the test dataset
    for i, (input_data, _) in enumerate(test_loader):
        if i >= 1:  # Process only one batch for simplicity
            break

        # Apply transformations
        noisy_input = add_noise(input_data)
        grayscale_input = grayscale(input_data)
        sketch_input = generate_sketch(input_data)

        # Move data to device if needed (e.g., GPU)
        input_data = input_data.to(device)
        noisy_input = noisy_input.to(device)
        grayscale_input = grayscale_input.to(device)
        sketch_input = sketch_input.to(device)

        # Forward pass through the model
        output = model(input_data)
        output_noisy = model(noisy_input)
        output_grayscale = model(grayscale_input)
        output_sketch = model(sketch_input)

        # Forward pass throught the model ..second time
        output_twice = model(output)
        output_noisy_twice = model(output_noisy)
        output_grayscale_twice = model(output_grayscale)
        output_sketch_twice = model(output_sketch)

        # Create a grid for visualization
        grid_input_noisy_output = make_grid(torch.cat([input_data, noisy_input, output_noisy, output_noisy_twice], dim=0),
                                            nrow=5, normalize=True, scale_each=True)
        grid_grayscale = make_grid(torch.cat([input_data, grayscale_input, output_grayscale, output_grayscale_twice], dim=0),
                                nrow=5, normalize=True, scale_each=True)
        grid_sketch = make_grid(torch.cat([input_data, sketch_input, output_sketch, output_sketch_twice], dim=0),
                                nrow=5, normalize=True, scale_each=True)

        save_image(grid_input_noisy_output, os.path.join(dir, f'grid_noisy.png'))
        save_image(grid_grayscale, os.path.join(dir, f'grid_grayscale.png'))
        save_image(grid_sketch, os.path.join(dir, f'grid_sketch.png'))