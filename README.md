# fl_based_intrusion_detection_system

# Privacy-Preserving Federated Learning based Intrusion Detection System (IDS)
## Feature Selection Module

This module performs hybrid feature selection on the IDS dataset using:
- Variance Threshold
- Mutual Information
- Random Forest Feature Importance

It generates reduced datasets for further IDS model training and Federated Learning implementation.

------------------------------------------------------------
## PROJECT STRUCTURE
------------------------------------------------------------

/FL_BASED_INTRUSION_DETECTION_SYSTEM
│
├── feature_selection.py
├── Train_data.csv
├── Test_data.csv
├── requirements.txt
├── README.md


------------------------------------------------------------
## HOW TO RUN THIS PROJECT USING VENV
------------------------------------------------------------

STEP 1: Open terminal inside project folder

cd path_to_project_folder

Example:
cd C:\Users\Downloads\fl


STEP 2: Create Virtual Environment

python -m venv venv


STEP 3: Activate Virtual Environment

Windows:
venv\Scripts\activate

If successful, you will see:

(venv) C:\Users\LENOVO\Downloads\fl>


STEP 4: Install Required Libraries

pip install -r requirements.txt

This installs:
- pandas
- numpy
- scikit-learn
- scipy
- joblib
- threadpoolctl


STEP 5: Run Feature Selection Script

python feature_selection.py

After successful execution, the following files will be created:

Final_Selected_Train.csv
Final_Selected_Test.csv


------------------------------------------------------------
## HOW TO RUN IN VS CODE
------------------------------------------------------------

1. Open the project folder in VS Code.

2. Press:
   Ctrl + Shift + P

3. Type:
   Python: Select Interpreter

4. Select:
   venv\Scripts\python.exe

5. Open feature_selection.py and click Run
   OR use terminal:
   python feature_selection.py



