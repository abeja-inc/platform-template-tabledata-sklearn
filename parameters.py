import os
import json


def get_env_var(key, converter, default=None):
    value = os.getenv(key)
    if value:
        return converter(value)
    return default


def get_env_var_bool(key, default: bool):
    value = os.getenv(key)
    if value:
        return value.lower() == 'true'
    return default


def get_env_var_csv(key, converter):
    value = os.getenv(key)
    if value:
        return list(set([converter(x.strip()) for x in value.split(',')]))
    return list()


def get_env_var_required(key, converter):
    value = get_env_var(key, converter)
    if value is None:
        raise Exception(f'"{key}"" is required.')
    return value


def get_env_var_validate(key, converter, default=None, min_=None, max_=None, list_=None):
    value = get_env_var(key, converter, default)
    if value:
        if min_ and value < min_:
            raise Exception(f'"{key}" must be "{min_} =< x =< {max_}"')
        if max_ and value > max_:
            raise Exception(f'"{key}" must be "{min_} =< x =< {max_}"')
        if list_ and value not in list_:
            raise Exception(f'"{key}" must be one of [{",".join(list_)}]')
    return value


def get_env_var_class_weight(key):
    value = os.getenv(key)
    if value:
        if value == "balanced":
            pass
        else:
            value = json.loads(value)
    return value


def get_env_var_gamma(key):
    value = os.getenv(key)
    if value:
        if value == "auto":
            pass
        else:
            value = float(value)
        return value
    return "auto"


class Parameters:
    """Parameter class
    parameter name must be consist of upper case characters.
    """
    """User Parameters"""
    _USER_PARAMETERS = {
        "DATALAKE_CHANNEL_ID", "DATALAKE_TRAIN_FILE_ID", "DATALAKE_TEST_FILE_ID", "INPUT_FIELDS", "LABEL_FIELD"
    }

    DATALAKE_CHANNEL_ID = get_env_var_required('DATALAKE_CHANNEL_ID', str)
    DATALAKE_TRAIN_FILE_ID = get_env_var_required('DATALAKE_TRAIN_FILE_ID', str)
    DATALAKE_TEST_FILE_ID = os.getenv('DATALAKE_TEST_FILE_ID')

    INPUT_FIELDS = get_env_var_csv('INPUT_FIELDS', str)
    LABEL_FIELD = get_env_var_required('LABEL_FIELD', str)

    """Core Parameters"""
    _CORE_PARAMETERS = {
        "CLASSIFIER", "NFOLD"
    }

    _CLASSIFIER_LIST = [
        "LinearRegression", "LogisticRegression", "SVR", "SVC"
    ]
    CLASSIFIER = get_env_var_validate('CLASSIFIER', str, "LinearRegression", list_=_CLASSIFIER_LIST)
    NFOLD = get_env_var_validate('NFOLD', int, default=5, min_=2, max_=None)

    """LinearRegression Parameters"""
    _LINEAR_REGRESSION_PARAMETERS = {
        "FIT_INTERCEPT", "NORMALIZE", "COPY_X", "N_JOBS"
    }

    FIT_INTERCEPT = get_env_var_bool('FIT_INTERCEPT', True)
    NORMALIZE = get_env_var_bool('NORMALIZE', False)
    COPY_X = get_env_var_bool('COPY_X', True)
    N_JOBS = get_env_var('N_JOBS', int)

    """LogisticRegression Parameters"""
    _LOGISTIC_REGRESSION_PARAMETERS = {
        "PENALTY", "DUAL", "TOL", "C", "FIT_INTERCEPT", "INTERCEPT_SCALING",
        "CLASS_WEIGHT", "RANDOM_STATE", "SOLVER", "MAX_ITER", "MULTI_CLASS",
        "VERBOSE", "WARM_START", "N_JOBS", "L1_RATIO"
    }

    _PENALTY_LIST = [
        "l1", "l2", "elasticnet", "none",
    ]
    PENALTY = get_env_var_validate('PENALTY', str, "l2", list_=_PENALTY_LIST)
    DUAL = get_env_var_bool('DUAL', False)
    TOL = get_env_var_validate('TOL', float, 1e-4, min_=1e-10)
    C = get_env_var_validate('C', float, 1.0, min_=1e-10)
    INTERCEPT_SCALING = get_env_var_validate('INTERCEPT_SCALING', float, 1.0, min_=1e-10)
    CLASS_WEIGHT = get_env_var_class_weight('CLASS_WEIGHT')
    RANDOM_STATE = get_env_var('RANDOM_STATE', int, 42)
    _SOLVER_LIST = [
        "newton-cg", "lbfgs", "liblinear", "sag", "saga",
    ]
    SOLVER = get_env_var_validate('SOLVER', str, "lbfgs", list_=_SOLVER_LIST)
    MAX_ITER = get_env_var_validate('MAX_ITER', int, 100, min_=0)
    _MULTI_CLASS_LIST = [
        "ovr", "multinomial", "auto",
    ]
    MULTI_CLASS = get_env_var_validate('MULTI_CLASS', str, "auto", list_=_MULTI_CLASS_LIST)
    VERBOSE = get_env_var_validate('VERBOSE', int, 0, min_=0)
    WARM_START = get_env_var_bool('WARM_START', False)
    L1_RATIO = get_env_var_validate('L1_RATIO', float, min_=0, max_=1)

    """SVR Parameters"""
    _SVR_PARAMETERS = {
        "KERNEL", "DEGREE", "GAMMA", "COEF0", "SUPPORT_VECTOR_TOL", "C",
        "EPSILON", "SHRINKING", "CACHE_SIZE", "SUPPORT_VECTOR_VERBOSE", "SUPPORT_VECTOR_MAX_ITER"
    }

    _KERNEL_LIST = [
        "linear", "poly", "rbf", "sigmoid", "precomputed",
    ]
    KERNEL = get_env_var_validate('KERNEL', str, "rbf", list_=_KERNEL_LIST)
    DEGREE = get_env_var('DEGREE', int, 3)
    GAMMA = get_env_var_gamma('GAMMA')
    COEF0 = get_env_var('COEF0', float, 0.0)
    EPSILON = get_env_var('COEF0', float, 0.1)
    SHRINKING = get_env_var_bool('SHRINKING', True)
    CACHE_SIZE = get_env_var('CACHE_SIZE', float, 200)
    SUPPORT_VECTOR_TOL = get_env_var_validate('SUPPORT_VECTOR_TOL', float, 1e-3, min_=1e-10)
    SUPPORT_VECTOR_VERBOSE = get_env_var_bool('SUPPORT_VECTOR_VERBOSE', False)
    SUPPORT_VECTOR_MAX_ITER = get_env_var('SUPPORT_VECTOR_MAX_ITER', int, -1)

    """SVC Parameters"""
    _SVC_PARAMETERS = {
        "KERNEL", "DEGREE", "GAMMA", "COEF0", "SUPPORT_VECTOR_TOL", "C",
        "PROBABILITY", "SHRINKING", "CACHE_SIZE", "SUPPORT_VECTOR_VERBOSE", "SUPPORT_VECTOR_MAX_ITER",
        "CLASS_WEIGHT", "DECISION_FUNCTION_SHAPE", "RANDOM_STATE"
    }

    PROBABILITY = get_env_var_bool('PROBABILITY', False)
    DECISION_FUNCTION_SHAPE = get_env_var_validate('DECISION_FUNCTION_SHAPE', str, "ovr", list_=["ovo", "ovr"])

    """ABEJA Platform environment variables"""
    _SYSTEM_PARAMETERS = {
        "ABEJA_STORAGE_DIR_PATH", "ABEJA_TRAINING_RESULT_DIR"
    }

    ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
    ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')

    @classmethod
    def as_dict(cls):
        params = {
            k: v for k, v in cls.__dict__.items()
            if k.isupper() and not k.startswith("_")
        }

        TARGET_LIST = {
            **cls._USER_PARAMETERS,
            **cls._CORE_PARAMETERS,
            **cls._SYSTEM_PARAMETERS
        }
        if params["CLASSIFIER"] == "LinearRegression":
            TARGET_LIST = {**cls._LINEAR_REGRESSION_PARAMETERS, **TARGET_LIST}
        elif params["CLASSIFIER"] == "LogisticRegression":
            TARGET_LIST = {**cls._LOGISTIC_REGRESSION_PARAMETERS, **TARGET_LIST}
        elif params["CLASSIFIER"] == "SVR":
            TARGET_LIST = {**cls._SVR_PARAMETERS, **TARGET_LIST}
        elif params["CLASSIFIER"] == "SVC":
            TARGET_LIST = {**cls._SVC_PARAMETERS, **TARGET_LIST}
        else:
            TARGET_LIST = {**cls._LINEAR_REGRESSION_PARAMETERS, **TARGET_LIST}

        rtn = {
            k: v for k, v in params.items()
            if k in TARGET_LIST
        }
        return rtn

    @classmethod
    def as_params(cls):
        params = cls.as_dict()
        is_support_vector_app = False
        if params["CLASSIFIER"] == "LinearRegression":
            TARGET_LIST = cls._LINEAR_REGRESSION_PARAMETERS
        elif params["CLASSIFIER"] == "LogisticRegression":
            TARGET_LIST = cls._LOGISTIC_REGRESSION_PARAMETERS
        elif params["CLASSIFIER"] == "SVR":
            TARGET_LIST = cls._SVR_PARAMETERS
            is_support_vector_app = True
        elif params["CLASSIFIER"] == "SVC":
            TARGET_LIST = cls._SVC_PARAMETERS
            is_support_vector_app = True
        else:
            TARGET_LIST = cls._LINEAR_REGRESSION_PARAMETERS

        rtn = {
            k.lower(): v for k, v in params.items()
            if k in TARGET_LIST
        }
        if is_support_vector_app:
            rtn["tol"] = rtn.pop("support_vector_tol")
            rtn["verbose"] = rtn.pop("support_vector_verbose")
            rtn["max_iter"] = rtn.pop("support_vector_max_iter")
        return rtn
