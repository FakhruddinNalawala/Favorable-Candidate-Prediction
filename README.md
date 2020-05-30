# Election
This is the working directory for our project. 

explaining all files ---

1. Images 
    output of result_visualization.py . Contains all images as output for visualizations ran on labelled files

2. bingnews.py
    script to scrape news articles . (COMMENTS NEED TO BE ADDED)

3. BJP_Labelled.csv and INC_Labelled.csv 
    output of vaderSentiment.py . Contains the BJP and INC dataset labelled POS,NEG,NEU by the program.

4. Logistic Regression , SVM , RandomForest , XGBoost,Naive bayes are the programs for the respective ML models. 
    comments have been added to Logistic Regression file for understanding. other files have almost similar codes. 

5. preprocess.py
    preprocessing and cleaning of data for the ML models . comments added

6. procBJPtweets.csv and procINCtweets.csv  are the datasets for BJP and INC tweets . 
    (PS - More INC tweets needed.. not enough )

7. result_visualizatiojn.py
    visualtions done on BJP_Labelled.csv and INC_Labelled.csv . preprocessing included . Comments added. 

8. train.csv 
    60k tweets for training all ML models . small part of the 16 million tweets dataset uploaded in Google drive in the TRAINING DATASET folder

9. tweets.py   
    script to fetch twitter data

10. vaderSentiment.py
    classifies tweets in dataset as POS,NEG,NEU using VADER  lexicon based approach

11. visual.py
    contains visualization done on train.csv . ##TEST VISUALIZATIONS##
 
12. data.csv
    dataset required for naive_bayes.py contains 40.5k tweets( equal ratio of pos and neg
    
13. pred.py
    It is consolidated code with various graphical representations. Uses LS_2.0.csv as dataset.
    
14. LS_2.0.csv
    Dataset for kaggle.py
    
15. accuracy.py
    compares the accuracy of predicted winners with the actual winners.

16. Winners.csv
    Consolidated dataset with all winners of karnataka 2019 elections.


