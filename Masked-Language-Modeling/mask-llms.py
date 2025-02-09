import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Function to mask tokens randomly
def mask_tokens(inputs, mask_prob=0.15):
    labels = inputs.clone()
    
    # Probability mask
    prob = torch.rand(inputs.shape)
    mask = (prob < mask_prob) & (inputs != tokenizer.pad_token_id) & \
           (inputs != tokenizer.cls_token_id) & (inputs != tokenizer.sep_token_id)

    inputs[mask] = tokenizer.mask_token_id  # Replace tokens with [MASK]
    return inputs, labels

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Define a small sample corpus
unlabeled_corpus = [
    "Machine learning is transforming the world.",
    "BERT is a powerful transformer model.",
    "Masked language modeling helps BERT understand context."
]

# Training loop
model.train()
for epoch in range(3):  # Example: 3 epochs
    total_loss = 0
    for text in unlabeled_corpus:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        masked_inputs, labels = mask_tokens(inputs['input_ids'])

        outputs = model(masked_inputs, labels=labels)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(unlabeled_corpus):.4f}")

print("Training complete.")

# Inference function to predict masked words
def predict_masked_sentence(sentence):
    model.eval()  # Set model to evaluation mode
    
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = outputs.logits
    masked_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    for idx in masked_index:
        predicted_token_id = torch.argmax(predictions[0, idx]).item()
        predicted_word = tokenizer.decode(predicted_token_id)
        sentence = sentence.replace("[MASK]", predicted_word, 1)

    return sentence

# Testing with examples
test_sentences = [
  "My favorite sport to watch is [MASK].",
    "She added some [MASK] to make the soup taste better.",
    "The baby was crying because it was [MASK].",
    "He couldn’t find his [MASK], so he had to walk in the rain.",
    "Before going to bed, I always brush my [MASK].",
    "I forgot my [MASK] at home, so I couldn't pay for my coffee.",
    "They decided to take a [MASK] to the mountains for the weekend.",
    "My phone ran out of [MASK] just when I needed it most."
]

for sentence in test_sentences:
    print(f"Input: {sentence}")
    print(f"Output: {predict_masked_sentence(sentence)}\n")
