import os
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from utilities import sample_from_fft
import torchvision.transforms as transforms

def train_model(f, f_copy, opt, data_loader, n_epochs, l_r, l_i, l_t, l_gp, alpha, fft_generated, fft_modelsave, generated, modelsave, device, fft=True):
    fixed_z = torch.randn(5, 3, 64, 64).to(device)
    modelsavedir = fft_modelsave if fft else modelsave
    generateddir = fft_generated if fft else generated
    for epoch in range(1, n_epochs+1):
        print(f"Current GPU: {torch.cuda.current_device()}, Total GPUs: {torch.cuda.device_count()}")
        acc_loss = 0
        for x, _ in tqdm(data_loader, desc=f'Epoch {epoch}/{n_epochs}', leave=False):
            if fft:
                z = sample_from_fft(x).to(device)
            else:
                z = torch.randn_like(x).to(device)
            x = x.to(device)
            # apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)

            # calculate losses
            loss_rec = (fx - x).pow(2).mean()
            loss_idem = (f_fz - fz).pow(2).mean()
            loss_tight = -(ff_z - f_z).pow(2).mean()

            # calculate gradient penalty term
            loss_gp = 0
            if l_gp != 0:
                alpha_gp = torch.rand(x.size(0), 1, 1, 1).to(device)
                interpolates = (alpha_gp * x + (1 - alpha_gp) * f_z).requires_grad_(True)
                d_interpolates = f(interpolates)
                gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                loss_gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            # clamping
            loss_tight = torch.tanh(loss_tight / (alpha * loss_rec)) * alpha * loss_rec

            # optimize for losses
            loss = l_r * loss_rec + l_i * loss_idem + l_t * loss_tight + l_gp * loss_gp
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(f.parameters(), 1)  # Gradient clipping
 
            # for p in f.parameters():
            #     p.data.clamp_(-1, 1)

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{n_epochs}, Loss: {acc_loss},')
        
        # Save model every 100 epochs
        if epoch % 200 == 0:
            model_path = os.path.join(modelsavedir, f'model_epoch_{epoch}.pt')  
            torch.save({
            'epoch': epoch,
            'state_dict': f.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, model_path)
            print(f'Model saved at {model_path}')

        # Save images every 20 epochs
        if epoch % 20 == 0:
            with torch.no_grad():
                f_fixed_z = f(fixed_z)
                f_f_fixed_z = f(f_fixed_z)

                # Inverse transformations before saving the grid
                inverse_transform = transforms.Compose([
                    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                ])
                images_to_save = torch.cat([
                    inverse_transform(fixed_z.cpu()),
                    inverse_transform(f_fixed_z.cpu()),
                    inverse_transform(f_f_fixed_z.cpu())
                ], dim=0)

                save_image(images_to_save, os.path.join(generateddir, f'grid_epoch_{epoch}.png'), nrow=5)