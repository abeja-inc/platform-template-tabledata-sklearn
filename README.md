# platform-template-tabledata-sklearn
## Environment options
### DataLake
env name|required|default|description
--------|--------|-------|-----------
DATALAKE_CHANNEL_ID|True|None|DataLake channel id
DATALAKE_TRAIN_FILE_ID|True|None|DataLake file id for training
DATALAKE_VAL_FILE_ID|False|None|DataLake file id for validating

### Features
env name|required|default|description
--------|--------|-------|-----------
INPUT_FIELDS|False|None|Names of features. <br>e.g. var_1,var_2,var_3
LABEL_FIELD|True|None|Name of label column.


### Scikit-Learn
For regression, use one of `LinearRegression`, `LinearSVR` or `SVR` with `IS_CLASSIFICATION=False`. For classification, use one of `LogisticRegression`, `LinearSVC` or `SVC`.

#### Common Parameters
env name|required|default|description
--------|--------|-------|-----------
CLASSIFIER|False|LinearRegression|Name of algorithm. You can choose LinearRegression, LogisticRegression, LinearSVR, LinearSVC, SVR and SVC.
IS_CLASSIFICATION|False|True|If `True`, classification, else regression.
STRATIFIED|False|True|Whether to perform stratified sampling.
NFOLD|False|5|Number of folds in CV. constraints: `NFOLD > 2`.
NUM_CLASS|False|2|Number of classes. Used only in `LogisticRegression` and `LinearSVC`. constraints: `NUM_CLASS > 2`. `2` means binary classification.

#### LinearRegression Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

env name|required|default|description
--------|--------|-------|-----------
FIT_INTERCEPT|False|True|Whether to calculate the intercept for this model.
NORMALIZE|False|False|This parameter is ignored when `FIT_INTERCEPT` is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
COPY_X|False|True|If True, X will be copied; else, it may be overwritten.
N_JOBS|False|None|The number of jobs to use for the computation.

#### LogisticRegression Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

env name|required|default|description
--------|--------|-------|-----------
PENALTY|False|l2|Used to specify the norm used in the penalization. Must be one of `[l1, l2]`
DUAL|False|False|Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
TOL|False|1e-4|Tolerance for stopping criteria.
C|False|1.0|Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
FIT_INTERCEPT|False|True|Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
INTERCEPT_SCALING|False|1.0|Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes `[x, self.intercept_scaling]`, i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes `intercept_scaling * synthetic_feature_weight`.
CLASS_WEIGHT|False|None|dict or ‘balanced’. Weights associated with classes in the form `{class_label: weight}`. If not given, all classes are supposed to have weight one.
RANDOM_STATE|False|42|The seed of the pseudo random number generator to use when shuffling the data.
SOLVER|False|lbfgs|Algorithm to use in the optimization problem. Must be one of `[newton-cg, lbfgs, liblinear, sag, saga]`.
MAX_ITER|False|100|Maximum number of iterations taken for the solvers to converge.
MULTI_CLASS|False|auto|If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’. Must be one of `[ovr, multinomial, auto]`.
VERBOSE|False|0|For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
WARM_START|False|False|When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
N_JOBS|False|None|The number of jobs to use for the computation.
L1_RATIO|False|None|The Elastic-Net mixing parameter, with `0 <= L1_RATIO <= 1`.

#### LinearSVR Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)

env name|required|default|description
--------|--------|-------|-----------
EPSILON|False|0.1|Epsilon parameter in the epsilon-insensitive loss function.
SUPPORT_VECTOR_TOL|False|1e-4|Tolerance for stopping criteria.
C|False|1.0|Penalty parameter C of the error term. The penalty is a squared l2 penalty. The bigger this parameter, the less regularization is used.
SVR_LOSS|False|epsilon_insensitive|Specifies the loss function. The epsilon-insensitive loss (standard SVR) is the L1 loss, while the squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss. Must be one of `[epsilon_insensitive, squared_epsilon_insensitive]`
FIT_INTERCEPT|False|True|Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations.
INTERCEPT_SCALING|False|1.0|When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes `intercept_scaling * synthetic feature weight`. Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
SUPPORT_VECTOR_DUAL|False|True|Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
VERBOSE|False|0|Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.
RANDOM_STATE|False|42|The seed of the pseudo random number generator to use when shuffling the data.
SUPPORT_VECTOR_MAX_ITER|False|1000|The maximum number of iterations to be run.

#### LinearSVC Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

env name|required|default|description
--------|--------|-------|-----------
PENALTY|False|l2|Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to `coef_` vectors that are sparse. Must be one of `[l1, l2]`.
SVC_LOSS|False|squared_hinge|Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss. Must be one of `[hinge, squared_hinge]`
SUPPORT_VECTOR_DUAL|False|True|Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
SUPPORT_VECTOR_TOL|False|1e-4|Tolerance for stopping criteria.
C|False|1.0|Penalty parameter C of the error term.
SVC_MULTI_CLASS|False|ovr|Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers, while `"crammer_singer"` optimizes a joint objective over all classes. While `crammer_singer` is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute. If `"crammer_singer"` is chosen, the options loss, penalty and dual will be ignored. Must be one of `[ovr, crammer_singer]`.
FIT_INTERCEPT|False|True|Whether to calculate the intercept for this model.
INTERCEPT_SCALING|False|1.0|When self.fit_intercept is True, instance vector x becomes `[x, self.intercept_scaling]`, i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector.
CLASS_WEIGHT|False|None|dict or ‘balanced’. Set the parameter C of class i to `class_weight[i]*C` for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`
VERBOSE|False|0|Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.
RANDOM_STATE|False|42|The seed of the pseudo random number generator to use when shuffling the data.
SUPPORT_VECTOR_MAX_ITER|False|1000|The maximum number of iterations to be run.

#### SVR Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

env name|required|default|description
--------|--------|-------|-----------
KERNEL|False|rbf|Specifies the kernel type to be used in the algorithm. Must be one of `[linear, poly, rbf, sigmoid, precomputed]`.
DEGREE|False|3|Degree of the polynomial kernel function (‘poly’).
GAMMA|False|auto|Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
COEF0|False|0.0|Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
SUPPORT_VECTOR_TOL|False|1e-3|Tolerance for stopping criterion.
C|False|1.0|Penalty parameter C of the error term.
EPSILON|False|0.1|Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
SHRINKING|False|True|Whether to use the shrinking heuristic.
CACHE_SIZE|False|200|Specify the size of the kernel cache (in MB).
SUPPORT_VECTOR_VERBOSE|False|False|Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
SUPPORT_VECTOR_MAX_ITER|False|1000|Hard limit on iterations within solver, or -1 for no limit.

#### SVC Parameters
Refer to [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

env name|required|default|description
--------|--------|-------|-----------
KERNEL|False|rbf|Specifies the kernel type to be used in the algorithm. Must be one of `[linear, poly, rbf, sigmoid, precomputed]`.
DEGREE|False|3|Degree of the polynomial kernel function (‘poly’).
GAMMA|False|auto|Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
COEF0|False|0.0|Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
SUPPORT_VECTOR_TOL|False|1e-3|Tolerance for stopping criterion.
C|False|1.0|Penalty parameter C of the error term.
PROBABILITY|False|False|Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
SHRINKING|False|True|Whether to use the shrinking heuristic.
CACHE_SIZE|False|200|Specify the size of the kernel cache (in MB).
SUPPORT_VECTOR_VERBOSE|False|False|Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
SUPPORT_VECTOR_MAX_ITER|False|1000|Hard limit on iterations within solver, or -1 for no limit.
CLASS_WEIGHT|False|None|dict or ‘balanced’. Set the parameter C of class i to `class_weight[i]*C` for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`
DECISION_FUNCTION_SHAPE|False|ovr|Must be one of `[ovo, ovr]`.
RANDOM_STATE|False|42|The seed of the pseudo random number generator used when shuffling the data for probability estimates.


## Run on local
Use `requirements-local.txt`.

```
$ pip install -r requirements-local.txt
```

Set environment variables.

| env | type | description |
| --- | --- | --- |
| ABEJA_ORGANIZATION_ID | str | Your organization ID. |
| ABEJA_PLATFORM_USER_ID | str | Your user ID. |
| ABEJA_PLATFORM_PERSONAL_ACCESS_TOKEN | str | Your Access Token. |
| DATASET_ID | str | Dataset ID. |
