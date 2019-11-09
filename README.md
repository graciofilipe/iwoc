This file covers how to execute code to reproduce and obtain all the results. 
For comments and answers to the questions look at the Comment.md and  Q&A.md files respectively. 


## To be able ro tun: 

(1) Install requirements in your virtual environment: 

```
$pip install -r requirements.txt
```

(2) Add data in appropriate folder! The executables `quick_exploration.py` and `analysis_for_questions.py`, assume the files called `calls.csv` , `leads.csv`, and `singups.csv`, to be in a folder called `input_data/`. 
To run them, make sure you have such files and folder. 
Alternatively you can change these assumptions in the code in the "mains" of each file. 

## Running data exploration

(1) Run explore_input_data.py
```.env
$python quick_exploration.py
```

This will produce info about each file save log files in the `exploratory_analysis` folder. The results of the exploration include the counting of categorical variables, verification of missing variables and the dataframe summaries. 

## Running the Analysis for the Q&A

(1) Run analysis_for_questions.py

```.env
$python analysis_for_questions.py
```
This will 