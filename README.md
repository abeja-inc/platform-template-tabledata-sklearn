# platform-template-tabledata-sklearn
## Environment options
### DataLake
env name|required|default|description
--------|--------|-------|-----------
DATALAKE_CHANNEL_ID|True|None|DataLake channel id
DATALAKE_TRAIN_ITEM_ID|True|None|DataLake item id
DATALAKE_TEST_ITEM_ID|False|None|DataLake item id

### Features
env name|required|default|description
--------|--------|-------|-----------
INPUT_FIELDS|False|None|Names of features. <br>e.g. var_1,var_2,var_3
TARGET_FIELD|True|None|Name of target column.


### Scikit-Learn
env name|required|default|description
--------|--------|-------|-----------
CLF|False|LinearRegression|Name of algorithm. You can choose LinearRegression, LogisticRegression, SVR and SVC.
PARAMS|False|dict|Parameters for model. <br>e.g. max_iter=1000,n_jobs=-1
NFOLD|False|5|To be passed to nfold.

