from transformers import AutoImageProcessor, ViTModel
from PIL import Image
import torch

image = Image.open("complete_dataset/naevus/008.jpg")

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

list(last_hidden_states.shape)

