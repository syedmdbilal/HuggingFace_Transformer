import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    def _init_(self):
        super(Net, self)._init_()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Modify fc1 to match the size in the saved checkpoint
        self.fc1 = nn.Linear(400, 120)
        # Modify fc2 to match the size in the saved checkpoint
        self.fc2 = nn.Linear(120, 84)
        # Modify fc3 to match the size in the saved checkpoint
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        print(x.shape, "x-shape")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# Load the trained model
model = Net()
model.load_state_dict(torch.load("cifar.pth"))
model.eval()

# Define the transformation to be applied to input images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a function to make predictions on input images
def classify_image(image):
    img_tensor = preprocess(image)
    print(img_tensor.shape)
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    print(output)
    _, predicted = torch.max(output, dim=1)
    print(predicted)
    return classes[predicted[0]]  # Return as a list

# Create input and output interfaces
image_input = gr.inputs.Image()
label_output = gr.outputs.Label()

# Create a Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
    title="CIFAR-10 Image Classification",
    description="Upload an image from the CIFAR-10 dataset and let the model classify it."
)

# Launch the interface
iface.launch()
