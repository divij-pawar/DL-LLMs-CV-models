import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.utils as vutils  # For saving images
import os

# Create output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Optimized Hyperparameters
num_epochs = 50
batch_size = 128
timesteps = 1000
image_size = 32
learning_rate = 1e-4
hidden_layer_size = 512  # Increased network capacity

# Load dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Noise schedule function (cosine schedule for improved stability)
def noise_schedule(timesteps):
    steps = torch.linspace(0, timesteps, timesteps + 1)
    return torch.cos(steps / timesteps * (torch.pi / 2)) ** 2  # Cosine schedule

# Function to apply forward diffusion
def forward_diffusion(x_0, t, alphas):
    alpha_t = alphas[t].view(-1, 1, 1, 1)  # Ensure correct shape
    eps = torch.randn_like(x_0)
    noisy_image = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * eps
    return noisy_image, eps

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, image_size * image_size)
        )
        self.timesteps = timesteps

    def forward(self, x, t):
        x = x.view(x.size(0), -1)  # Flatten input
        predicted_noise = self.model(x)
        return predicted_noise.view(x.size(0), 1, image_size, image_size)  # Reshape back

# Initialize model and optimizer
model = DiffusionModel(timesteps).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
alphas = noise_schedule(timesteps).to(device)

# Training loop
for epoch in range(num_epochs):
    for images, _ in dataloader:
        images = images.to(device)

        # Sample random diffusion steps
        t = torch.randint(0, timesteps, (images.size(0),), device=device)

        # Apply noise
        noisy_images, eps = forward_diffusion(images, t, alphas)

        # Predict noise
        predicted_eps = model(noisy_images, t)

        # Compute loss (MSE between predicted and actual noise)
        loss = ((eps - predicted_eps) ** 2).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")

# ---------------------------
# Inference (Reverse Diffusion)
# ---------------------------
@torch.no_grad()
def reverse_diffusion(model, alphas, shape=(1, 1, image_size, image_size)):
    """
    Generates images by reversing the diffusion process from pure noise.
    """
    model.eval()
    x_t = torch.randn(shape).to(device)  # Start from Gaussian noise

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device)  # Current timestep tensor
        predicted_eps = model(x_t, t_tensor)  # Predict noise

        # Reverse diffusion formula
        alpha_t = alphas[t]
        x_t = (x_t - torch.sqrt(1 - alpha_t) * predicted_eps) / torch.sqrt(alpha_t)

    return x_t

# Generate images
generated_images = reverse_diffusion(model, alphas, shape=(1, 1, image_size, image_size))

# Save generated image
image_path = os.path.join(output_dir, "generated_image.png")
vutils.save_image(generated_images, image_path, normalize=True)
print(f"Image saved to {image_path}")
