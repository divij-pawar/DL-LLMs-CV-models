import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
latent_size = 100  # Size of noise vector input to generator
image_size = 64  # Images will be resized for the network
num_epochs = 20
lr = 0.0002

# Dataset / Dataloader
dataset = datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z.view(z.size(0), -1))

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 32, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),  
            nn.Linear(256 * 8 * 8, 1),  # Corrected linear layer input size
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers and loss function
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        real_preds = discriminator(real_images)
        real_loss = criterion(real_preds, real_labels)
        
        noise = torch.randn(batch_size, latent_size, device=device)
        fake_images = generator(noise)
        fake_preds = discriminator(fake_images.detach())
        fake_loss = criterion(fake_preds, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        generator.zero_grad()
        fake_preds = discriminator(fake_images)
        g_loss = criterion(fake_preds, real_labels)  # Trick discriminator into thinking fake is real
        g_loss.backward()
        optimizer_g.step()
        
        # Print losses occasionally
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}')
    
    # Save generated samples at the end of each epoch
    with torch.no_grad():
        sample_noise = torch.randn(16, latent_size, device=device)
        generated_images = generator(sample_noise).cpu()
        plt.figure(figsize=(8, 8))
        for j in range(16):
            plt.subplot(4, 4, j+1)
            plt.imshow(generated_images[j][0], cmap='gray')
            plt.axis('off')
        plt.savefig(f'generated_epoch_{epoch+1}.png')
        plt.show()

# Inference Function
def generate_images(num_images=16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_size, device=device)
        generated_images = generator(noise).cpu()
        plt.figure(figsize=(8, 8))
        for j in range(num_images):
            plt.subplot(4, 4, j+1)
            plt.imshow(generated_images[j][0], cmap='gray')
            plt.axis('off')
        plt.savefig('generated_inference.png')
        plt.show()

generate_images()  # Generate and display images after training
