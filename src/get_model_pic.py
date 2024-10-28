import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary







class SimpleANN(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Create an instance of the model
model = SimpleANN(output_dim=10)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a dummy input tensor and move it to the same device
dummy_input = torch.randn(1, 1, 28, 28).to(device)  # Batch size of 1, 1 channel, 28x28 image

# Get the output
with torch.no_grad():  # Disable gradient calculation
    output = model(dummy_input)

# Visualize the model
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("simple_ann_architecture", format="png")  # Save as PNG

summary(model, input_size=(1, 28, 28))  # Adjust input size as needed
