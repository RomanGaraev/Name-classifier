# Name-classifier

A project with comparison of simple LSTM and fully-connected networks in a task of gender classification by a person's name

## Repository Structure

```
├── requirements.txt         <- The requirements file for reproducing the experiments. Most of them are standart and maybe already installed
├── data                     <- Data files and scripts directory
│   └── data                 <- Loaded names in .csv format
│       └── test_eng.csv      
│       └── test_rus.csv     
│       └── train_eng.csv     
│       └── train_eng.csv    
│   └── data_loader.py       <- Loading and data normalizing script
├── models                   <- Trained models
│   ├── BaselineLSTM.h5        
│   └── MyLSTM.h5            
│   └── Perceptron.h5       
├── src                      <- All the logic related to the training the models
│   ├── train.py             <- train script
│   └── test.py              <- test script
│   └── my_models.py         <- the architecture of models
```
## Dependences installing 
All the libraries can be pip installed using `pip install -r requirements.txt`

Python >=3.8 is required

## Train & test
Go to `src\` folder and run `python train.py` to get hyper-tuned models: simple LSTM, dense network and combined model. 

If you would like to test them, run `python test.py`. Expected output is confusion matrices for each model.

