import torch
from PIL import Image
import torchvision.transforms as transforms
from model import UNet

def remove_background(image_path, model_path, output_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate mask
    with torch.no_grad():
        mask = model(input_tensor)
    
    # Post-process mask
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype('uint8') * 255
    mask = Image.fromarray(mask).resize(original_size)
    
    # Apply mask to original image
    image = image.convert('RGBA')
    image.putalpha(mask)
    
    # Save result
    image.save(output_path, 'PNG')

if __name__ == '__main__':
    remove_background(
        image_path='path/to/input/image.jpg',
        model_path='background_removal_model.pth',
        output_path='output.png'
    ) 