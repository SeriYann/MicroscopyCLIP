# Microscopy-CLIP

This repository provides the official weights and usage instructions for  
**MicroscopyCLIP: A Domain-Specific Vision-Language Model for Optical Microscopy**.

> A CLIP model (`clip-vit-base-patch32` base) fine-tuned on microscopy image–text pairs, released for downstream microscopy image classification and retrieval tasks.


### Environment

All dependencies are provided in `environment.yml`. You can create the environment with:

```bash
conda env create -f environment.yml
```
After installation, activate the environment:

```bash
conda activate msclip
```

## Usage


### Model Weights

The pretrained weights are available on Hugging Face:

**[MicroscopyCLIP](https://huggingface.co/SeriYann/MicroscopyCLIP)**

 
### Load the model
 
```python
from transformers import CLIPModel, CLIPProcessor
 
model_path = "path/to/checkpoint"
 
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```
 
### Zero-shot image–text similarity
 
```python
import cv2
import numpy as np
from PIL import Image
 
class_names = ["class name 1", "class name 2", "..."]
text_prompts = [f"a microscopy image of {name.lower()}" for name in class_names]
 
text_inputs = processor.tokenizer(
    text=text_prompts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
 
image = cv2.imread("example.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image_rgb = np.stack([image, image, image], axis=2)
image_pil = Image.fromarray(image_rgb.astype("uint8"))
 
image_inputs = processor.image_processor(images=[image_pil], return_tensors="pt")
 
outputs = model(
    input_ids=text_inputs["input_ids"],
    attention_mask=text_inputs["attention_mask"],
    pixel_values=image_inputs["pixel_values"],
)
 
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
```
 
### Using it as a feature extractor / classifier backbone
 
```python
import torch.nn as nn
from transformers import CLIPModel
 
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model_path, num_classes, freeze_backbone=True):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        feature_dim = self.clip_model.config.vision_config.hidden_size
 
        if freeze_backbone:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
 
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
 
    def forward(self, images):
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        pooled_output = vision_outputs.pooler_output
        return self.classifier(pooled_output)
```
 



## Acknowledgements

We would like to thank the authors of the original CLIP implementation, which this work builds upon:

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [🤗 Transformers](https://github.com/huggingface/transformers)

Their open-source contributions provided the foundation for this work.



