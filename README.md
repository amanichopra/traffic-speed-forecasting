# DS Capstone

## Data Loading
1. Create a new environment and install the dependencies in requirements.txt.
2. Open the utils directory and read the README to set up credentials to scrape from PEMS.
3. Run data_loader.py to gather the datasets.
4. Run the cells in station_meta_extractor.ipynb and station_data_extractor.ipynb to process and load data into pickle files used in EDA.

## Data Processing & EDA
1. Run the notebooks in ./processing to process the data for model building. The output pickle files will be in ./data/processed. (If you don't want to run the notebooks to process the data as this can be a time consuming step, unzip [this](https://drive.google.com/file/d/1sh5PCJXXAM-qBnQtIAKqVSOAY440i8Xy/view?usp=sharing) folder and replace ./data/processed with it.
2. Run EDA.ipynb to get an overview of the data.

## Modeling


