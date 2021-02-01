# Predicting_Diabetes_Patients_Readmission
# Reducing Readmission Risk using Predictive Analytics

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Problem Definition
### What is a Readmission?
Admission to a hospital within 30 days of a discharge from the same or another hospital. The numbers are disturbing and disappointing. According to the federal government, one in five elderly patients winds up back in the hospital within 30 days of leaving. The cost is troubling, too. The readmission of Medicare patients alone costs $26 billion annually, $17 billion of which is spent on return trips that wouldn’t need to happen if patients received proper care during their first visit.

```
Domain      : Machine Learning
Techniques  : Classification techinque (XGBoost)
Application : Healthcare Management
```
## Dataset Details
Original Dataset		: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
```
Dataset Name		  : Diabetes 130-US hospitals for years 1999-2008 Data Set
Number of Class		  : 2
Number of Instances	  : 100000
Number of Attributes  : 55
```
![Data Sample](/images/sample.png)

## Demographic Information
![Demograhic_Info](/images/demographic_info.png)

#### Summary-I
* Caucasian population is the most predominant group in the dataset
* There is almost an equal representation of the male and female population
* Majority of the patients are 40 - 90 years old, 70-80 years age group being the predominant group 

## EHR Dataset Levels
![EHR](/images/EHR_dataset_levels.png)

### Diabetes 130-US hospitals dataset is at the encounter level
The following code is used to select only the first encounter of each patient
        
       :::python3
       def select_first_encounter(df):
       '''
       df: pandas dataframe, dataframe with all encounters
       returns:
           - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
       '''
                first_encounter = df.sort_values(["encounter_id", "patient_nbr"], ascending=[True, True]). \
                               groupby("patient_nbr").head(1).reset_index(drop=True)
                return first_encounter

## Prevalance
       
       :::python3
       def compute_prevelance(df, col):
            """
            Count number of occurrences of each value in array.
            Args:
                df - DataFrame
                col - array
            Returns
                prevelance
            """
            neg, pos = np.bincount(df[col])
            prevelance = 100 * pos / (neg + pos)
            return prevelance
            
![Prevelance](/images/prevelance.png)    

![Workflow](/images/workflow.png)

## Numeric Data
### Normalization Techniques at a Glance

**CLIPPING** - caps all feature values above (or below) a certain value
        * 'num_medications'
        * 'num_lab_procedures'
        * 'number_diagnoses'

**LOG_SCALING** - compress a wide range to a narrow range
        * 'number_inpatient'
        * 'number_outpatient'
        * 'number_emergency'

**Standard Scaling**
        * All features will be standardized using the Z-score

![Normalized_Data](/images/Normalized_data.png)

## Categorical Data
### High Cardinality of Diagnostic Codes(ICD-9)

        :::python3
        def reduce_diag_dimensionality(dataframe, col):
                """
                reduce the dimensionality of the diagnostic codes by grouping them to one level up
                df - dataframe
                col - diagnostic code column
                returns:
                - grouped col with codes grouped at one level high    
                """
                # 1. pad all codes with zeros at the begining with zfill() upto 3
                dataframe[col] = dataframe[col].str.zfill(3)
                # 2. extract the first two characters and store them into a different column
                new_colname = 'grouped_'+ col
                dataframe[new_colname] = dataframe[col].map(lambda x: x[:2] if type(x) is str else x)
                dataframe.drop(col, axis=1, inplace=True)
                return dataframe

![Diagnostic_Codes](/images/Diagnostic_Codes.png)

## Splitting Data into Train/Validation/Test data

        :::python3
        def split_dataset_patient_level(df, key, test_percentage):
                """ Splits data into Train and Test without data leakage"""
                df = df.iloc[np.random.permutation(len(df))]                    # random shuffle the data
                unique_values = df[key].unique()                                # unique patients
                total_values = len(unique_values)                               # number of unique patients   
                sample_size = round(total_values * (1-test_percentage))         # sample size
                train = df[df[key].isin(unique_values[:sample_size])].reset_index(drop=True) #subset train_df
                test = df[df[key].isin(unique_values[sample_size:])].reset_index(drop=True)  # subset test_df
                return train, test
 
![Prevalence](/images/train_validation_test_data.png)

## UpSampling Imbalance Data - SMOTE
![UpSampling](/images/SMOTE.png)

## Hyperparameters - Tuned
    :::python3
    param = {'max_depth': 11,
             'min_child_weight': 5,
             'eta': 0.01,
             'subsample': 1,
             'colsample_bytree': 0.8,
             'objective': 'binary:logistic',
             'eval_metric': 'auc'}

## Model performance on validation data
![Model Performance on validation data](/images/confusion_matrix_thresh_50.png)
-------------------------------------
#### Confusion Matrix (threshold=0.5)
-------------------------------------
The confusion matrix above is a 2-by-2 matrix laying out correct and incorrect predictions made in each label.
Class 0 - Patients **NOT readmitted**
Of the 3721 patients in this class **(top left and top right quadrants)** the model correctly predicted the outcome for 3418 patients, i.e; ~92% of them.
Class 1 - Patients **Readmitted**
Of the 366 patients in this class **(bottom left and bottom right quadrants)**  the model captured only 45 patients, i.e; ~12.2% of them. 

## Cumulative Gain Curve
The cumulative gains chart shows the percentage of the overall number of cases in the “readmitted” category (label=1) **gained** by targeting a percentage of the total number of cases. For example, the point 0.4 (40%) on the x-axis, if we score a dataset with the model and score all of the observations by predicted probability of readmitted, we would expect the top 40% of the patient population will contain ~52% of the readmitted patients. 
The diagonal line is the "baseline" curve; if you select 40% of the cases from the scored dataset at random, you would expect to "gain" approximately 40% of all of the cases that actually take the category **readmitted**. **The farther above the baseline a curve lies, the greater the gain**. The cumulative gains chart to help choose a classification cutoff by choosing a percentage that corresponds to a desirable gain, and then mapping that percentage to the appropriate cutoff value.
![Cumulative_Gain_Plot](/images/cumulative_gain.png)


## Precision-Recall Trade-Off
### Computing the cost of mistakes
As a healthcare provider who aims to improve patient care we want to monitor as many patients as possible to identify the high-risk patients. Our inability to find readmitted patients may generate a a large penalty and monetary burden. So we don't want to miss many high-risk patients --- we'd rather put up with false alarms about potentially readmitted patients instead of missing readmitted patients entirely. In other words, false negatives cost more than false positives.
We may simply prefer to reduce the percentage of false negatives. Let’s examine two metrics here:
```
Recall:
                [Recall] = [# positive data points with positive predictions] / [# all positive data points]
                                                        OR
                         = [# true positives] / [# true positives] + [# false negatives]
```

```
Precision: 
                [Precision] = [# positive data points with positive predictions] / [# all data points with positive predictions]
                                                        OR
                         = [# true positives] / [# true positives] + [# false positives]
```

In the graph below we will explore the trade-off between precision and recall. We will examine what happens when we use a different threshold value for making class predictions. False negatives are costly in our case, so we may want to be more liberal about making positive predictions. To achieve this, instead of thresholding class probabilities at 0.5, we can choose a lower threshold. We then explore a range of 100 different threshold values between 0-0.5 and plot the associated precision-recall curve.


![Precision-Recall curve](/images/PR_curve.png)

## Model performance on test data
![Model Performance on test data](/images/confusion_matrix_thresh_32_test.png)
-------------------------------------
#### Confusion Matrix (threshold=0.33)
-------------------------------------
The confusion matrix above is a 2-by-2 matrix laying out correct and incorrect predictions made in each label at a threshold value of 0.33.
Class 0 - Patients **NOT readmitted**
Of the 12813 patients in this class **(top left and top right quadrants)** the model incorrectly predicted the outcome for 10514 patients, i.e; ~822% of them.
Class 1 - Patients **Readmitted**
Of the 1274 readmitted patients (bottom left and bottom right quadrants) in the test dataset the model captured 88% of them i.e;  a66% increase in the number of true positives from ~12.2% identified with the base model.

# Top Features
![Best and Worst Features](/images/features_25.png)

## Summary
* The model built here should be used only to predict readmission of patients of the Caucasian race in the age group of 50-90 years old. Deploying this model on any other race or age group can result is significant errors.


* The model identified the following features as most important in predicting readmission: 
    
    *	Time a patient spent in the hospital
    
    *	Number of diagnosis
    
    *	Number of procedures a patient underwent
    
    *	Number of lab procedures a patient underwent
    
    *	The number of medications the patient is prescribed
A high value of these features indicate that the patient is probably very sick and needs continuous attention. 


* The model also identifies patients with diseases associated with the circulatory and the respiratory systems as a high-risk group. This is consistent with published research that people suffering heart diseases and infections such as pneumonia are more likely to be readmitted than others in the hospital.


* Age is another important factor in determining if a patient is readmitted or not. Older people between 60-90 years are more likely to be readmitted. It is also important to keep in mind that the majority of the patients in the dataset are in this age.


* The model performs about 1.2 times better than a random guess in identifying readmitted patients. To identify at least 50% of the readmitted patients, according to the model the healthcare provides should target 40% of the most likely patients.


## Tools/ Libraries
```
Languages	: Python
Tools/IDE	: Jupyter Notebook
Libraries	: Scikit Learn, Pandas, Numpy, Matplotlib
```



