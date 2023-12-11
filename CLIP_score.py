from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_path = "result/video2190_ddim/keys/"

imgs = sorted(os.listdir(img_path))

total_similarity = 0
img_num = 0

for img in imgs:
    img_num += 1
    image = Image.open(img_path + img)
    inputs = processor(text=["a handsome man in Chinese ink wash painting, black and white"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    
    total_similarity += logits_per_image[0][0]

print(total_similarity/img_num)