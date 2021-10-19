# FIFA19 CSE 163 Project - Winter 2021, Hunter Schafer

###  Authors: Neel Jog, Mark Ryman, Davin Seju

  
 
  

## Environment / System requirements

This project is designed to be run in a python environment, version 3.7 or later. 
Required libraries include installation of stated or later versions of:

 - Pandas 1.0.5
 - GeoPandas 0.8.2
 - Matplotlib 3.2.2
 - Seaborn 0.10.1
 - Numpy 1.18.5
 - Sklearn 0.23.1
 - Keras Tensorflow 2.3.0.

  ### Source Files

The source files required to be in the same directory for running the project are:

1. fifa_lib.py : most methods for loading, cleaning and filtering data and producing output to answer all research questions not pertaining specifically to the machine learning model, as well as some data visualizations

2. ml_visualization.py : produces data visualizations for output from CNN machine learning model

3. ml_training.py : handles pre-processing of data and trains CNN machine learning model

4. fifa19_main.py : calls and executes all methods from fifa_lib.py and ml_visualization.py



### Data Files  

The three required data files below should also be in the same directory


1. FIFA2019.csv is the primary dataset

download link: https://www.kaggle.com/karangadiya/fifa19

2. world_cup_2018_stats.csv

download link: https://gitlab.com/djh_or/2018-world-cup-stats/blob/master/world_cup_2018_stats.csv

3. The third file can be downloaded directly from the collection at this site. It is the 2018-19 season data for the Premier League in the England section.
File name: england-premier-league-2018-to-2019.csv 

download link: https://sports-statistics.com/sports-data/soccer-datasets/



## Instructions to run code

This project requires the four different .py files and the three data files to reside in the same directory.

The machine learning aspect of this project is somewhat computing intensive.


To avoid the expense of the model training,  run the training from within the fifa19_main module and then comment out line : train_save_both_models for any subsequent runs.

All of the remaining code will run with terminal command:

 _python fifa19_main.py_

All output from running this code has been incorporated into our full report in PDF format and print statements have been removed for clarity.  Visualizations are saved as png files in local directory.