# Predicting_Diabetes_Patients_Readmission

## Problem Definition
A hospital readmission is when a patient who is discharged from the hospital gets re-admitted again within a certain period of time. Readmission of patients is a huge financial burden on the insurance companies and other payment agencies. The hospitals also suffer because it reflects poorly on the quality of the service they provide.
In 2011, American hospitals spent over $41 billion on diabetic patients who got readmitted within 30 days of discharge. Determining factors that lead to higher readmission, and correspondingly being able to predict which patients will get readmitted can help hospitals save millions of dollars while improving quality of care.

In this project I will build a model predicting readmission for patients with diabetes. The steps I will go through are:

•	data exploration and data cleaning

•	building training/validation/test samples

•	model selection

•	model evaluation

## Conclusion
Through this project, I created a binary classifier to predict the probability that a patient with diabetes would be readmitted to the hospital within 30 days. On held out data, my best model was Logistic Regression with AUC of 0.64. Linear Classifiers performed better than the tree-based models.

•	My best model performs approximately 1.8 times better than randomly selecting patients.

•	The caregivers should check on 11 people to identify 1 patient who is likely to be readmitted for the same cause. My best will reduce this by half, the caregivers should contact only 5 people to identify a likely candidate.

Linear classifiers and Tree-based classifiers identified a number of diagnostic codes as top predictors for patient readmission.

•	Patients with a history of readmission are more likley to be readmitted again.

•	Age of the patients is another key feature. Older patients are more likely to be readmitted. Hence more attention should be paid to older patients.

•	Diabetes Medicines 'Rosiglitazone', 'Nateglinide', 'Glyburide', Glyburide-metformin' seem to control patients from readmission. Patients not given these drugs are more likely to be readmitted.
