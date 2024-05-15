from transformers import ASTFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch

dataset = load_dataset("superb", "ks")
dataset = dataset.sort("file")["validation"]
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = ASTFeatureExtractor(num_mel_filters=64, num_frequency_bins=512)
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
breakpoint()

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)

# compute loss - target_label is e.g. "down"
target_label = model.config.id2label[0]
inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
loss = model(**inputs).loss
round(loss.item(), 2)
