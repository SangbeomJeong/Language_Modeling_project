# Language_Modeling_project

This project will work on deep learning networks for character-level language modeling. We compare performance by implementing vanilla RNN and LSTM using the Shakespeare dataset. Afterwards, samples are generated using a model with high performance.

## Code Configuration

- `dataset.py`: Defines a PyTorch dataset class that reads Shakespeare's works from a given text file and creates a character-by-character dataset for language modeling.
- `model.py`: Defines the architectures of vanilla RNN and vanilla LSTM.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.

## How to run it 
```
### code download
git clone https://github.com/SangbeomJeong/

### Train run
CUDA_VISIBLE_DEVICES=0 python main.py

### Generate run 
CUDA_VISIBLE_DEVICES=0 python generate.py
```

# Model Implementations
* For both models, the **hidden size was used as 128, num_layer was set to 4, and the output size was made the same as the input size**.
## Vanilla_RNN:

Layer1 : embedding layer(input_size, hidden_size)

Layer2 : RNN(hidden_size, hidden_size, num_layers, batch_first=True)

Layer3 : FC layer(hidden_size, output_size)

## Vanilla_LSTM:

Layer1 : embedding layer(input_size, hidden_size)

Layer2 : LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

Layer3 : FC layer(hidden_size, output_size)


# Performance Comparison(수정필요)

## Model Performance Summary
### Vanilla_RNN

Training Accuracy: 56.91%
Testing Accuracy: 53.55%
Training Loss: 1.4140
Testing Loss: 1.5644

![Uploading training_testing_metrics.png…](figures/LSTM_loss.png)

### Vanilla_LSTM

Training Accuracy: 52.82%
Testing Accuracy: 51.38%
Training Loss: 1.5952
Testing Loss: 1.6645

![Uploading training_testing_metrics.png…](figures/RNN_loss.png)

## Analysis and Discussion



