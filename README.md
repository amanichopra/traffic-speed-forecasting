# DS Capstone

## Data Loading
1. Create a new environment and install the dependencies in requirements.txt.
2. Open the utils directory and read the README to set up credentials to scrape from PEMS.
3. Run ```./utils/data_loader.py``` to gather the datasets from PEMS.
4. Run the cells in ```./utils/station_meta_extractor.ipynb``` and ```./utils/station_data_extractor.ipynb``` to lightly process and load data into pickle files used in EDA.

## Data Processing & EDA
1. Run the notebooks in ```./processing``` to process the data for model building. The output pickle files will be in ```./data/processed```. (If you don't want to run the notebooks to process the data as this can be a time consuming step, unzip [this](https://drive.google.com/file/d/1bmIvoG4kBYyH5cQILljLiI7vFuvY1dqA/view?usp=sharing) folder and replace ```./data/processed``` with it.
2. Run EDA.ipynb to get an overview of the data.

## Modeling
There are 6 models currently in place.

1. **Baseline:** Our naive model forecasts using mean value based on time of day.
2. **Prophet:** Facebook's additive regression model with four main components — a piecewise linear logistic growth curve trend; a yearly seasonal component modelled using Fourier series; a weekly seasonal component created using dummy variables; and a user-provided list of important holidays.
3. **ARIMA**
4. **CNN**
5. **LSTM**
6. **STGCN:** Spatio-temporal graph convolutional neural network designed by researchers at Peking University. 


These models can be loaded by running the notebooks in ```./models``` or by importing from [here](https://drive.google.com/file/d/1dEe7mns24EDFOV5ny4iQqrhZSjIX_bgq/view?usp=sharing) (unzip the folder and replace ```./models/trained``` with it). 

Tuning is performed in each model's notebook in ```./models```. We searched the hyperparameter space using an exhaustive grid search to minimize RMSE. The parameter grid for each model is listed below:

1. **Baseline:** No hyperparameters, so no tuning performed.
2. **Prophet:** ```{  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'yearly_seasonality': [True, False]
}```
3. **ARIMA:** Used pmdarima.auto_arima to select best parameters while minimizing AIC.
4. **CNN:** ```{  
    'num_lags': [1, 3, 5, 10]
}```
5. **LSTM:** ```{  
    'num_lags': [1, 3, 5, 10]
}```
6. **STGCN:** ```{  
    'k': [2, 4, 6, 8, 10, 12],
    'epochs': [5, 10, 15, 20, 25, 30],
    'opt': ['RMSProp', 'Adam'],
    'ks': [1, 3, 5],
    'num_lags': [4, 8, 12],
    'lr': [1e-3, 1e-2, 1e-1, 1, 10]
}``` 


## Evaluation
Run ```evaluation.ipynb``` to measure and compare the models and view the plots in ```./plots``` to visualize forecast results.




