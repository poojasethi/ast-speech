# This is the code to fine-tune the AST model on emotion classification using the IEMOCAP dataset

pip install transformers datasets torchaudio scikit-learn
pip install accelerate -U
pip install transformers[torch]

from transformers import AutoFeatureExtractor, ASTForAudioClassification, TrainingArguments, Trainer, get_scheduler, EarlyStoppingCallback
from datasets import load_dataset, concatenate_datasets
import torch
from sklearn.metrics import accuracy_score
import torchaudio
from collections import Counter
import numpy as np

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the IEMOCAP_Audio dataset
dataset = load_dataset("Zahra99/IEMOCAP_Audio")

# Inspect the dataset to find the correct label key
print(dataset["session1"].features)

# Correct label key based on inspection
label_key = "label"

# Merge all sessions into one dataset
all_sessions = [dataset[session] for session in ["session1", "session2", "session3", "session4", "session5"]]
full_dataset = concatenate_datasets(all_sessions)

# Split the dataset into train and test sets
train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Function to resample audio to 16kHz
def resample(audio_array, original_sr, target_sr=16000):
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        audio_array = resampler(torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return audio_array

# Preprocess function to extract features and handle labels
def preprocess_function(examples):
    audio = examples["audio"]["array"]
    sampling_rate = examples["audio"]["sampling_rate"]
    audio = resample(audio, sampling_rate)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension

    # Convert label to numerical ID
    label_str = examples[label_key]
    inputs["labels"] = torch.tensor(label2id[label_str], dtype=torch.long)

    return inputs

# Load feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Move model to the appropriate device
model.to(device)

# Extract unique labels from the dataset
unique_labels = list(set(full_dataset[label_key]))
label2id = {name: i for i, name in enumerate(unique_labels)}
id2label = {i: name for name, i in label2id.items()}

# NEW
unique_labels = full_dataset.unique(label_key)
print("Unique labels in the dataset:", unique_labels)
# NEW

model.config.label2id = label2id
model.config.id2label = id2label

print("Label2id mapping:", label2id)

# Check the distribution of labels before preprocessing
labels = [example[label_key] for example in train_dataset]
label_counts = Counter(labels)
print("Label distribution before preprocessing:", label_counts)

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio"])
test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio"])

# Check a few samples to ensure correct processing
for sample in train_dataset.select(range(10)):
    print(f"Processed label: {id2label[sample['labels']]}")  # Debugging line

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5, #can also run for 20
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # gradient_accumulation_steps=2,  # Gradient accumulation
    # warmup_steps=int(0.1 * len(train_dataset) * 50)  # Warmup steps as 10% of total steps
)

# Initialize the learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * len(train_dataset) * 50), num_training_steps=len(train_dataset) * 50
)

# Compute metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc}

# Initialize Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(eval_dataset=test_dataset)
print(results)

# Calculate accuracy on the test set
true_labels = []
predicted_labels = []

# Iterate over the preprocessed test dataset
for sample in test_dataset:
    inputs = {
        "input_values": torch.tensor(sample["input_values"]).unsqueeze(0).to(device)
    }

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = torch.argmax(logits, dim=-1).item()
    true_label_id = sample["labels"]

    true_labels.append(true_label_id)
    predicted_labels.append(predicted_class_id)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print sample predictions for verification
for i in range(10):  # Print first 10 sample predictions
    true_label_name = id2label[true_labels[i]]
    predicted_label_name = id2label[predicted_labels[i]]
    print(f"Sample {i}: Predicted Label: {predicted_label_name}, True Label: {true_label_name}")
