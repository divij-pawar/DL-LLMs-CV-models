# BERT Masked Language Modeling

This repository demonstrates a simple implementation of **Masked Language Modeling** (MLM) using **BERT** from the `transformers` library by Hugging Face. The code covers the basics of training a BERT model on a small corpus by masking tokens and training it to predict the masked tokens.

## Requirements
Create a python virtual environment to avoid conflicts.

```bash
$ python -m venv env
```

Activate the environment to install the dependencies inside it.
```bash
$ source env/bin/activate
```
To install the required dependencies, you can use the following command:

```bash
(env) $ pip install -r requirements.txt
```
## Run
Run the file inside the python virtual environment.
```bash
(env) $ python mask-llms.py
```
## Output Example
```bash
Epoch 1, Loss: 4.4672
Epoch 2, Loss: 3.3216
Epoch 3, Loss: 3.4203
Training complete.
Input: My favorite sport to watch is [MASK].
Output: My favorite sport to watch is basketball.

Input: She added some [MASK] to make the soup taste better.
Output: She added some salt to make the soup taste better.

Input: The baby was crying because it was [MASK].
Output: The baby was crying because it was crying.

Input: He couldn’t find his [MASK], so he had to walk in the rain.
Output: He couldn’t find his car, so he had to walk in the rain.

Input: Before going to bed, I always brush my [MASK].
Output: Before going to bed, I always brush my teeth.

Input: I forgot my [MASK] at home, so I couldn't pay for my coffee.
Output: I forgot my coffee at home, so I couldn't pay for my coffee.

Input: They decided to take a [MASK] to the mountains for the weekend.
Output: They decided to take a trip to the mountains for the weekend.

Input: My phone ran out of [MASK] just when I needed it most.
Output: My phone ran out of battery just when I needed it most.
```
The model can yield better results with more epochs and finetuning.

## Training Process
The training process for fine-tuning the BERT model using Masked Language Modeling (MLM) involves the following steps:

1. **Tokenization**  
   The input text is tokenized using the BERT tokenizer. This converts the raw text into a sequence of token IDs that the model can process.

2. **Masking**  
   Tokens in the input text are randomly masked with a probability of 15%. The masking process ensures that the model learns to predict the masked tokens based on the surrounding context.

3. **Model Training**  
   The model is trained to predict the masked tokens using the `CrossEntropyLoss` function. The loss is computed by comparing the model's predictions with the original (unmasked) tokens.

4. **Optimization**  
   The AdamW optimizer is used to update the model parameters. This optimizer is a variant of Adam that includes weight decay for better regularization.

This process is repeated for multiple epochs to fine-tune the model on the provided corpus.


## License  
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

