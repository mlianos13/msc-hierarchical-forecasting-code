# core_main.py

#%%
from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import Dict, Union, TYPE_CHECKING, Type, Set, Sequence
from scipy.optimize import minimize
import inspect
import copy
import os
import functools
import pickle
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
from warnings import warn
from scipy.spatial.distance import cdist
from river.tree import HoeffdingTreeRegressor
import pyswarms as ps
import warnings
import traceback
import logging
from scipy.linalg import LinAlgError
from scipy.linalg import solve_triangular
eps = np.finfo(float).eps
from scipy.linalg import qr, qr_update, solve_triangular
from scipy.spatial.distance import cdist


logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This forces reconfiguration even if logging was already initialized
)

logger = logging.getLogger(__name__)


# no loggings
logging.disable(logging.CRITICAL)

# TODO: consider removing scipy dependency and including optimization and externally. Confidence intervals may use different quantile.
# TODO: check for issues using "col", consider instead using iloc for indexing columns in dataframes to avoid issues with non unique column names.
# TODO: change variance estimation to be done (optionally) empirically in the ForecastModel, otherwise in the specific predictors as an extra output dataframe

# Get the directory of the current module
module_dir = os.path.dirname(__file__)

# Construct the path to the data file
data_folder = os.path.join(module_dir, 'data')

@pd.api.extensions.register_dataframe_accessor("fc")
class ForecastMatrix:
    
    def __init__(self, data: pd.DataFrame):
        self._obj = data
     
    def append(self, data: pd.DataFrame, allow_new_columns = False):
        # Append time data of same shape
        data = data.copy()
        if not allow_new_columns and not all(self._obj.columns == data.columns):
            raise ValueError("New data columns do not match own.")
        if self.n_t > 0:
            data.index = range(self._obj.index[-1] + 1, self._obj.index[-1] + 1 + len(data))        
        return pd.concat([self._obj, data], axis = 0)

    def join(self, data: pd.DataFrame, how = "left"):
        if not data.fc.check():
            data = data.copy()
            data.fc.convert()
        return self._obj.join(data, how = how)

    def drop_data(self, *variables):
        for v in variables:
            if v in self._obj.columns.get_level_values(0):
                self._obj.drop(columns=[v], level=0, inplace=True)
            else:
                raise ValueError(f"Variable '{v}' not found in data.")

    def _get_columns(self, *variables, kseq = None):
        if variables:
            v_cond = lambda v: v in variables
        else: 
            v_cond = lambda v: True
        if kseq:
            k_cond = lambda h: h in kseq
        else:
            k_cond = lambda h: True

        col_cond = lambda col: v_cond(col[0]) and k_cond(col[1])

        return [col_cond(col) for col in self._obj.columns]

    def subset(self, *variables, start_time=None, end_time=None, start_index=None, end_index=None, kseq=None) -> DataFrame:
        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        if end_time is not None:
            end_time = pd.to_datetime(end_time)
        row_mask = np.ones(len(self._obj), dtype=bool)
        if start_time is not None and 't' in self._obj.columns:
            row_mask &= (self._obj["t"] >= start_time).values.flatten()
        if end_time is not None and 't' in self._obj.columns:
            row_mask &= (self._obj["t"] < end_time).values.flatten()
        if start_index is not None:
            row_mask[:start_index] = False
        if end_index is not None:
            row_mask[end_index:] = False


        selected_cols = self._get_columns(*variables, kseq = kseq)
        subset_data = self._obj.iloc[:, selected_cols].copy()

        # Only apply row mask if any corresponding function arguments are not none
        if any([arg is not None for arg in [start_time, end_time, start_index, end_index]]):
            subset_data = subset_data.loc[row_mask]

        return subset_data

    def get_data(self, *variables, kseq=None, include_shape=False):
        result = {}
        if include_shape:
            result["n_t"] = self.n_t
    
        selected_cols = self._get_columns(*variables, kseq = kseq)

        subset_data = self._obj.iloc[:,selected_cols]

        for v in variables:
            result[v] = subset_data.fc[[v]]

    # Use slicing to retain upper-level column names
 #       for v in variables:
 #          result[v] = subset_data.loc[:, pd.IndexSlice[v, :]]

        return result

    def check(self):
        # Check if self._obj conforms to forecast structure
        if not isinstance(self._obj.columns, pd.MultiIndex):
            return False
        elif not self._obj.columns.names == ['Variable', 'Horizon']:
            return False
        # Check if all horizons are integers
        elif not all(isinstance(h, int) for h in self.kseq):
            return False
        
        return True

    def convert(self, separator = ".k"):
        data = self._obj

        if not isinstance(data.columns, pd.MultiIndex):
            input_name_pattern = re.compile(rf'^(.*?){re.escape(separator)}(\d+)$')
            new_columns = []
            for col in data.columns:
                if isinstance(col, str):
                    match = input_name_pattern.match(col)
                    if match:
                        name = match.group(1)
                        if not match.group(2).isdigit():
                            raise ValueError(f"Horizon must be an integer, got {match.group(2)}.")
                        horizon = int(match.group(2))
                        new_columns.append((name, horizon))
                    else:
                        new_columns.append((col, 'NA'))    
                else:
                    new_columns.append((col, 'NA'))

        else:
            # Rename horizons if they are not integers

            new_columns = list(data.columns)
            for i, col in enumerate(new_columns):
                if not isinstance(col[1], int):
                    if isinstance(col[1], str):
                        if col[1].isdigit():
                            new_columns[i] = (col[0], int(col[1]))
                        else:
                            match = col[1].rsplit(separator, 1)
                            if len(match) == 2 and match[1].isdigit():
                                new_columns[i] = (col[0], int(match[1]))
                            else:
                                new_columns[i] = (col[0], 'NA')

        data.columns = pd.MultiIndex.from_tuples(new_columns, names=('Variable', "Horizon"))

        if "t" in data.columns.get_level_values(0):
            # Check if dtype is correct
            if data["t"]["NA"].dtype != 'datetime64[ns]':
                data[("t", "NA")] = pd.to_datetime(data[("t", "NA")])


    def remove_old_data(self, n_keep: int = None):
        if n_keep is None:
            n_keep = max(self.kseq)
        # Keep only required data
        n_drop = self.n_t - n_keep
        if n_drop > 0:
            self._obj.drop(self._obj.index[:n_drop], inplace=True)

    @property
    def n_t(self):
        return len(self._obj)
    
    @property
    def variables(self):
        return self._obj.columns.get_level_values(0).unique().tolist()
    
    @property
    def kseq(self):
        res = self._obj.columns.get_level_values(1).unique().tolist()
        if 'NA' in res:
            res.remove('NA')
        return tuple(res)

    def get_horizon_tail(self, horizon: str | int, n_t):
        col_filter = self._obj.columns.get_level_values('Horizon').isin([horizon, 'NA'])
        data = self._obj.loc[:, col_filter].copy()
        return data[-(n_t+horizon):]


    def get_lagged_subset(self, *args, extend = False, reverse = False, **kwargs) -> DataFrame:
        """
        Retrieves a subset of the data lagged according to column name forecast horizons.
        """
        subset = self.subset(*args, **kwargs)
        if extend: # Add nan values to the end, to support lagged values beyond original data
            k = max(subset.fc.kseq)
            nan_frame = pd.DataFrame({col: [np.nan]*k for col in subset.columns})
            if reverse:
                nan_frame.index = range(subset.index[0] - len(nan_frame), subset.index[0])
                subset = nan_frame.fc.append(subset)
            else:
                subset = subset.fc.append(nan_frame)
            
        for i, col in enumerate(subset.columns):
            if col[1] not in ["NA", ""]:
                k = -col[1] if reverse else col[1]
                subset.iloc[:,  [i]] = subset.iloc[:, [i]].shift(k, fill_value=float("nan"))
        return subset

    def get_design(self, n, k):
        kseq = None if k is None else ("NA", k)        
        subset = self.subset(kseq = kseq)
        return subset.shift(k)


    def join_variable(self, var, data: pd.DataFrame, how = "right") -> DataFrame:
        data = data.copy()
        new_cols = pd.MultiIndex.from_product([[var], data.columns], names = self._obj.columns.names)
        data.columns = new_cols
        return self.join(data, how = how)

    def join_horizon(self, horizon, data: pd.DataFrame, how = "right") -> DataFrame:
        data = data.copy()
        new_cols = pd.MultiIndex.from_product([data.columns.get_level_values(0), [horizon]], names = self._obj.columns.names)
        data.columns = new_cols
        return self.join(data, how = how)

    def __getitem__(self, key):
        match = next((col for col in self._obj.columns if col[0] == key), None)
        if match and match[1] == 'NA':
            result = self._obj[key]["NA"]
            if isinstance(result, pd.Series):
                result.name = key
            return result
        else:
            return self._obj[key]

class DataFrame(pd.DataFrame):
    fc: ForecastMatrix

def new_fc(data = None, index = None, columns = None, dtype = None, copy = None, separator = '.k') -> DataFrame:
    result = pd.DataFrame(data, index, columns, dtype = dtype, copy = copy)
    if not result.fc.check():
        result.fc.convert(separator = separator)
    return result

def forecast_matrix_from_product(names: list | tuple, kseq: tuple) -> DataFrame:
    """
    Create a DataFrame with MultiIndex columns from the product of names and kseq.
    """
    cols = pd.MultiIndex.from_product([names, kseq], names = ['Variable', 'Horizon'])
    return new_fc(columns = cols)

#%%
@functools.wraps(pd.read_csv)
def read_forecast_csv(*args, multiindex = False, separator = ".k", **kwargs) -> DataFrame:
    if multiindex:
        df: DataFrame = pd.read_csv(*args, header = [0, 1], **kwargs)
    else:
        df: DataFrame = pd.read_csv(*args, **kwargs)
    if not df.fc.check():
        df.fc.convert(separator = separator)
    # TODO: add check that columns are correctly loaded
    return df

if os.path.exists(data_folder) and 'simulated_data.csv' in os.listdir(data_folder):
    sample_data = read_forecast_csv(data_folder + '/simulated_data.csv', separator=".k")


#%%
# Keyword for returning all raw data
class RawData:
    pass

class Transformation:

    params: Dict[str, Union[int, float, str]] = {}
    recursion_pars: list  = []
    output_mode = None # Either None, or "repeat". If "repeat", output is repeated along horizons, if None, output is not modified.
    preserve_names = False # Whether to disallow ForecastModels to change column names when combining outputs.

    # Superclass for transformations
    def __init__(self, **kwargs: Dict[str, Union[int, float, str]]):

            """
            kwargs: dict of input names or transforms (required) and parameter values (optional). (Parameter values may not be transforms).
            """

            self.inst_params = self.__class__.params.copy()
            self.inst_inputs = {key: None for key in self.__class__.inputs}

            # Check for unused params
            for key in kwargs:
                if key in self.inst_params:
                    if kwargs[key] is not None:
                        self.inst_params[key] = kwargs[key]
                elif key in self.inputs:
                    if (isinstance(kwargs[key], (str, Transformation)) or kwargs[key] == RawData):
                        self.inst_inputs[key] = kwargs[key]
                    else:
                        raise ValueError(f"Input {key} must be a string, Transformation or RawData.")
                else:
                    raise KeyError(f"{self} has no input or parameter: {key}.")

            # Check if all inputs are satisfied 
            if None in self.inst_inputs.values():
                raise ValueError(f"Inputs not fully specified: {self.inputs}")


    def __init_subclass__(cls):


        if hasattr(cls, "evaluate"):
            sig = inspect.signature(cls.evaluate)
            cls.evaluate_args = list(sig.parameters.keys())[1:]
            var_keyword_arg = next((param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)
            cls._accepts_any_args = var_keyword_arg is not None
            cls.inputs = [key for key in cls.evaluate_args if key not in cls.params and key not in cls.recursion_pars and key not in ["n_t", "kseq", var_keyword_arg]]
        else:
            raise ValueError("No evaluate method found.")

    @property
    def dependencies(self):
        return [v for v in (self.inst_params | self.inst_inputs).values() if isinstance(v, Transformation)]

    #TODO: add check for circular dependencies

    def update_params(self, **params:  Dict[str, Union[int, float, str]]):
         # Store input parameters as names of variables, or numerical parameter values
        for key, value in params.items():
            if key in self.inst_params:
                self.inst_params[key] = value
            else:
                raise KeyError(f"{self} has no parameter: {key}.")

    @property
    def evaluate_args(self):
        return list(inspect.signature(self.evaluate).parameters.keys())[1:]

    @property
    def _var_keyword_arg(self):
        return next((param.name for param in inspect.signature(self.evaluate).parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)

    @property
    def _accepts_any_args(self):
        return self._var_keyword_arg is not None

    @property
    def inputs(self):
        return [key for key in self.evaluate_args if key not in self.inst_params and key not in self.recursion_pars and key != self._var_keyword_arg]

    def apply(self, data: DataFrame = None, kseq = None, return_recursion_pars = True, evaluate_args = None):

        if evaluate_args is None:
            evaluate_args = {}

        # Setup for recursive application if required
        extra_recursion_pars = {}

        # Check inputs
        for name, val in self.inst_inputs.items():
            
            # Try to fetch data from raw input if not provided directly
            if name not in evaluate_args:
                if val is RawData:
                    evaluate_args[name] = data
                    
                elif val in data:
                    evaluate_args[name] = data.fc[[val]]

                # Attempt to fetch transformation dependencies if not provided directly.
                elif isinstance(val, Transformation):
                    val_res = val.apply(data, kseq = kseq, return_recursion_pars = True, evaluate_args = evaluate_args)
                    if isinstance(val_res, tuple):
                        t_val, t_rec_pars = val_res
                    else:
                        t_val = val_res
                        t_rec_pars = {}

                    evaluate_args[name] = t_val
                    extra_recursion_pars.update({val: t_rec_pars})
                else:
                    raise ValueError(f"Input {val} not found in data.")

        # Expand args targetting self, i.e. recursive parameters passed in case this transformation is a dependency and its value was not provided to parent.
        if self in evaluate_args:
            evaluate_args.update(evaluate_args[self])

        # Clean up evaluate arguments and evaluate
        eval_out = self.evaluate(**self.inst_params, **{k: v for k, v in evaluate_args.items() if k in self.evaluate_args})
        if isinstance(eval_out, tuple):
            result, recursion_pars = eval_out
        else:
            result = eval_out
            recursion_pars = {}
    
        # Clean up result if needed
        if not isinstance(result, pd.DataFrame):
            try:
                result = new_fc(result)
            except:
                raise ValueError(f"Could not convert output to DataFrame: {self}.")

        # Check if output horisontal dimension matches kseq, if not repeat along horizons

        if kseq is None:
            kseq = data.fc.kseq

        recursion_pars.update(extra_recursion_pars)

        # Check that result adheres to fc format
        if not result.fc.check():
            result.fc.convert()

        if return_recursion_pars:
            return result, recursion_pars
        else:
            return result

    def __add__(self, other):
        return SumTransformation(self, other)

    def __radd__(self, other):
        return SumTransformation(other, self)

    def __sub__(self, other):
        return SubTransformation(self, other)
    
    def __rsub__(self, other):
        return SubTransformation(other, self)
    
    def __mul__(self, other):     
        return MulTransformation(self, other)

    def __rmul__(self, other):
        return MulTransformation(self, other)

    def __truediv__(self, other):
        return DivTransformation(self, other)
    
    def __rtruediv__(self, other):
        return DivTransformation(other, self)

    def __pow__(self, other):
        return PowTransformation(self, other)

    def __rpow__(self, other):
        return PowTransformation(other, self)

    def evaluate(self, **data) -> tuple[DataFrame, dict] | DataFrame:
        # In: any inputs or parameters required for the transformation
        # Note: function signature can be freely set, but input params should be set on instantiation to map generic function arguments of evaluate to actual data names of input data list.
        # Out: result (DataFrame) or a tuple of result and a dict including recursion parameters (for e.g. rls).
        raise NotImplementedError("This method should be overridden by subclasses")

class One(Transformation):
    
    params = {}
    recursion_pars = []
    output_mode = "repeat"

    def __init__(self):
        super().__init__(data = RawData)

    def evaluate(self, data):
        n_t = data.fc.n_t
        return np.ones((n_t, 1))


class LowPass(Transformation):
    params = {"ta": 1}
    recursion_pars = ["prev_value"]

    def __init__(self, var, ta=None):
        super().__init__(data=var, ta=ta)

    def evaluate(self, data: pd.DataFrame | pd.Series, ta, prev_value=None):
        # Ensure data is a NumPy array for faster computation
        y = data.to_numpy(copy=True)
        n_t = len(y)

        # Initialize the output array
        result = np.empty_like(y)

        # Set the initial value
        if prev_value is None:
            prev_value = y[0]

        # Handle NaN values in the input
        valid_mask = ~np.isnan(y)  # Mask for valid (non-NaN) values

        # Initialize the first value
        result[0] = np.where(valid_mask[0], prev_value, np.nan)

        # Apply the low-pass filter using vectorized operations

        for i in range(1, n_t):

            result[i] = np.where(valid_mask[i], ta * result[i - 1] + (1 - ta) * y[i], result[i-1])
            result[i] = np.where(~np.isnan(result[i]), result[i], y[i])


        # Convert back to DataFrame or Series
        result_df = pd.DataFrame(result, index=data.index, columns=data.columns)
        return result_df, {"prev_value": result[-1]}


class FourierSeries(Transformation):

    params = {"nharmonics": 1}

    def __init__(self, data, nharmonics = 1):
        super().__init__(data = data, nharmonics = nharmonics)

    def evaluate(self, data, nharmonics):
        dfs = {}
        for i in range(1, nharmonics + 1):
            dfs[f"cos_{i}"] = np.cos(2*np.pi*i*data)
            dfs[f"sin_{i}"] = np.sin(2*np.pi*i*data)

        result = pd.concat(dfs, axis = 1)

        result.fc.convert()

        return result


class PrimitiveTransformation(Transformation):

    def __init__(self, a, b):

        if not isinstance(a, Transformation):
            a = Param(a)

        if not isinstance(b, Transformation):
            b = Param(b)

        self.a, self.b = a, b

        self.inst_inputs = {}

        self.inst_inputs["a"] = a

        self.inst_inputs["b"] = b

        super().__init__(a = a, b = b)

    @property
    def inst_params(self):
        result = {}
        result.update({"a_" + key: value for key, value in self.a.inst_params.items()})
        result.update({"b_" + key: value for key, value in self.b.inst_params.items()})

        
        return result

    @inst_params.setter
    def inst_params(self, value):
        if isinstance(self.a, Transformation):
            self.a.update_params(**{key[2:]: value for key, value in value.items() if key.startswith("a_")})
        elif "a" in value:
            self.a = value["a"]
        if isinstance(self.b, Transformation):
            self.b.update_params(**{key[2:]: value for key, value in value.items() if key.startswith("b_")})
        elif "b" in value:
            self.b = value["b"]

    def update_params(self, **params):
        self.inst_params = params

    def evaluate(self, **data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def fetch_values(self, a, b):
        if isinstance(a, pd.DataFrame):
            a = a.values
        if isinstance(b, pd.DataFrame):
            b = b.values
        return a, b

class SumTransformation(PrimitiveTransformation):

    def evaluate(self, a, b, **params):
        a, b = self.fetch_values(a, b)            
        return a+b
    
class MulTransformation(PrimitiveTransformation):

    def evaluate(self, a, b, **params):
        a, b = self.fetch_values(a, b)            
        return a*b 

class DivTransformation(PrimitiveTransformation):    

    def evaluate(self, a, b, **params):
        a, b = self.fetch_values(a, b)
        return a/b
        
class SubTransformation(PrimitiveTransformation):        


    def evaluate(self, a, b, **params):
        a, b = self.fetch_values(a, b)
        return a-b  

class PowTransformation(PrimitiveTransformation):

    def evaluate(self, a, b, **params):
        a, b = self.fetch_values(a, b)
        return a**b  

class TimeOfDay(Transformation):

    tile_output = True

    def __init__(self, t = "t"):
        super().__init__(t = t)

    def evaluate(self, t, kseq):
        delta = t - t.dt.floor("D")
        seconds = delta.dt.total_seconds()
        time_of_day_float = 2 * np.pi * seconds / 86400
        return time_of_day_float

class Identity(Transformation):

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
         return data

class Param(Transformation):
    
    params = {"value": None}

    def __init__(self, value):
        super().__init__(value = value)

    def apply(self, data = None, kseq=None, return_recursion_pars=True, evaluate_args=None):
        return self.evaluate(**self.inst_params)

    def evaluate(self, value):
        return value

class Map(Transformation):

    preserve_names = True

    def __init__(self, *vars, data = RawData):
        # TODO: consider renaming "data" to "source"
        super().__init__(data = data)
        self.vars = list(vars)
    
    def evaluate(self, data):
        return data[self.vars]

class IndexMap(Transformation):
    
    preserve_names = True

    def __init__(self, *indices, data = RawData):
        super().__init__(data = data)
        self.indices = list(indices)
    
    def evaluate(self, data):
        return data.iloc[:, self.indices]

class AR(Transformation):

    params = {"order": 1}
    recursion_pars = ["prev_values"]

    def __init__(self, data, order = 1):
        super().__init__(data = data, order = order)

    def evaluate(self, data, order, prev_values = None):
        y = data.copy()

        if not prev_values is None:
            y = pd.concat([prev_values, y])

        prev_values = y.iloc[-order:]

        y = y.shift(order, fill_value=float("nan"))

        result = y[-len(data):]

        return result, {"prev_values": prev_values}

class TransformComponent:

    def __init__(self, transform: Transformation, regressor = False, regressand = False):
        self.regressor = regressor
        self.regressand = regressand
        self.recursion_pars: dict = {}        
        self.transform: Transformation = transform

    def reset(self):
        self.recursion_pars = {}

    @property
    def params(self):
        return self.transform.inst_params.copy()

    def update_params(self, **params):
        self.transform.update_params(**params)

    def __repr__(self):
        return f"TransformComponent(transform={type(self.transform).__name__}, params={self.params})"

class ForecastTuple(tuple):
    def __new__(cls, *dataframes: Optional[DataFrame]):
        return super().__new__(cls, dataframes)

    def append(self, other: ForecastTuple, allow_new_columns = False):
        if len(self) == 0:
            return other
        elif len(self) != len(other):
            raise ValueError("Number of dataframes must match.")

        dataframes = []
        for d_self, d_other in zip(self, other):
            if d_self is None:
                df = d_other
            elif d_other is None:
                df = d_self
            else:
                df = d_self.fc.append(d_other, allow_new_columns = allow_new_columns)
            dataframes.append(df)

        return ForecastTuple(*dataframes)

    def join(self, other: ForecastTuple, how = "left"):
        if len(self) != len(other):
            raise ValueError("Number of dataframes must match.")
        
        dataframes = []
        for d_self, d_other in zip(self, other):
            if d_self is None:
                df = d_other
            elif d_other is None:
                df = d_self
            else:
                df = d_self.fc.join(d_other, how = how)
            dataframes.append(df)

        return ForecastTuple(*dataframes)

    def get_lagged_subset(self, *args, extend = False, **kwargs):
        dataframes = []
        for df in self:
            if df is None:
                dataframes.append(None)
            else:
                dataframes.append(df.fc.get_lagged_subset(*args, extend = extend, **kwargs))
        return ForecastTuple(*dataframes)
    
    def subset(self, *variables, start_time=None, end_time=None, start_index=None, end_index=None, kseq=None):
        dataframes = []
        for df in self:
            if df is None:
                dataframes.append(None)
            else:
                dataframes.append(df.fc.subset(*variables, start_time=start_time, end_time=end_time, start_index=start_index, end_index=end_index, kseq=kseq))
        return ForecastTuple(*dataframes)

    def remove_old_data(self, n_keep: int = None):
        for df in self:
            if df is not None:
                df.fc.remove_old_data(n_keep)

    def copy(self):
        return ForecastTuple(*[df.copy() if df is not None else None for df in self])

    def to_fc(self):
        return pd.concat([df for df in self if df is not None], axis = 1)

def concat_forecast_tuples(*forecast_tuples: ForecastTuple):
    dataframes = []
    for i in range(len(forecast_tuples[0])):
        dataframes.append(pd.concat([p[i] for p in forecast_tuples], axis = 1))
    return ForecastTuple(*dataframes)


#%%
class Predictor(ABC):

    # TODO: consider simplifying this class

    def __init__(self, x_columns: pd.MultiIndex = None, y_columns: pd.MultiIndex = None):
        super().__init__()
        self.x_columns = x_columns
        self.y_columns = y_columns

        # For variance estimation
        self.memory = None
        self.variance_estimate = None

    def __init_subclass__(cls):        
        # Inspect __init__ signature
        init_sig = inspect.signature(cls.__init__)

        # Ensure m_x and m_y are in signature, overwrite __init__ and issue warning if not present.
        if not "x_columns" in init_sig.parameters or not "y_columns" in init_sig.parameters:
            orig_init = cls.__init__
            @functools.wraps(orig_init)
            def new_init(self, *args, x_columns=None, y_columns=None, **kwargs):
                orig_init(self, *args, **kwargs)
                self.x_columns = x_columns
                self.y_columns = y_columns

            cls.__init__ = new_init

            # Warn that x_columns and y_columns were not found in signature and have been added.
            warn(f"x_columns and y_columns not found in signature of {cls.__name__}.__init__. They have been added to the signature.")        
            
        # Extract all parameters from update_fit signature
        update_fit_sig = inspect.signature(cls.update_fit)
        cls.params = [k for k, v in list(update_fit_sig.parameters.items())[1:] if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and k not in ["X", "Y", "X_pred"]]


    @abstractmethod
    def update_fit(self, X: DataFrame, Y: DataFrame, X_pred: DataFrame, **params) -> ForecastTuple:

        # Input: X: (lagged) regressor matrix, Y: regressand matrix, X_pred: (non-lagged) values for which to provide predictions, Y_hat: (old) predictions for Y ~ X.
        # Note, X, Y, X_pred and Y_hat should all be assumed to be lagged appropriately.
        # predict: if True, make predictions for X_pred. If False, only update fit.
        # params: any parameters required for updating fit, e.g. lambda for RLS.
        # output: a tuple of dataframes. The first dataframe should contain predictions for X_pred. Subsequent dataframes can contain any additional data, e.g. variance estimates. Dataframes should contain number of columns equal to number of horizons x number of outputs.
        pass

    # TODO: Remove this and move empirical estimation to ForecastModel. Specific variance estimators should be included in update_fit as an extra df output.
    def update_var_est(self, residuals: DataFrame, lam: float = 0.99):
        """
        Updates the weighted sample variance estimate, dynamically weighted according to memory:
            T_(n+1) = (memory*T_n + res**2)/(memory + 1)

        Where res is the residual and memory is the geometric series of lambda, approaching 1/(1-lambda).

        Note: this is intended primarily for use with RLS with forgetting factor lambda, but may be useful for other predictors as well.

        TODO: consider implementing a more generally applicable variance estimation method, that can reliably be used with any predictor.

        TODO: consider using fixed weighting and initialising variance estimate from burn-in period.
        """ 

        n = len(residuals)
        cols = residuals.columns
        n_vars = len(cols)
        residuals = residuals.to_numpy(dtype=float)

        # Initialise memory and variance estimate
        if self.variance_estimate is None:
            last_row = np.full(n_vars, float("nan"))
            memory = np.zeros(n_vars)
        else:
            last_row = self.variance_estimate #self.variance_estimate.iloc[[-1]].to_numpy(dtype=float)
            memory = self.memory #.iloc[[-1]].to_numpy(dtype=float)

        # Compute variance estimate
        variance_array = np.zeros((n, n_vars))

        for i in range(n):
            mod_last_row = np.where(np.isnan(last_row), 0, last_row) # Only use non nan values for computing new estimates
            tmp = lam*memory
            new_memory = tmp + (~np.isnan(residuals[i])).astype(int)
            new_row = (tmp*mod_last_row + residuals[i]**2)/new_memory
            new_row = np.where(np.isnan(new_row), last_row, new_row) # Use last estimate where new is nan
            variance_array[i] = new_row
            last_row = new_row 
            memory = new_memory

        # Update local variance estimate and memory
        self.variance_estimate = last_row
        self.memory = memory

        return pd.DataFrame(variance_array, columns=cols)

class RLS(Predictor):

    def __init__(self, theta = None, R = None, x_columns: pd.MultiIndex = None, y_columns: pd.MultiIndex = None, burn_in = 1, estimate_variance = False):
        super().__init__()
        if theta is None or R is None:
            if x_columns is None or y_columns is None:
                raise ValueError("x_columns and y_columns must be provided if theta or R are not.")
            else:
                n_x, n_y = len(x_columns), len(y_columns)
                theta, R = np.zeros((n_x, n_y)), 1/10000*np.eye(n_x)
        
        self.theta: np.ndarray = theta
        self.R: np.ndarray = R
        self.burn_in = burn_in
        self._n_updates = 0
        self.estimate_variance = estimate_variance

    def update_fit(self, X, Y, X_pred, predict = True, rls_lambda = 0.99):

        # Prepare result container
        Y_hat_new = new_fc(columns = Y.columns, index = X_pred.index)

        # Recursively update fit
        n_t = len(X)
        for i in range(n_t):

            # Set and check x, y
            x_i = X.iloc[i]
            y_i = Y.iloc[i]
            x_i_pred = X_pred.iloc[i]

            x_i_ready = not np.isnan(x_i).any()
            y_i_ready = not np.isnan(y_i).any()

            # Only update if data is valid
            if x_i_ready and y_i_ready:
                self.rls_update(x_i, y_i, rls_lambda)

            # Make prediction if x_i_pred is valid
            if not np.isnan(x_i_pred).any() and predict and self._n_updates >= self.burn_in:
                Y_hat_new.iloc[i] = self.predict(x_i_pred)
            else:
                Y_hat_new.iloc[i] = float("nan")

        return Y_hat_new

    def rls_update(self, x_i, y_i, rls_lambda):
        #TODO: handle errors in case of singular R
        self.R = rls_lambda * self.R + np.outer(x_i, x_i)
        self.theta = self.theta + np.outer(np.linalg.solve(self.R, x_i), y_i - x_i.T @ self.theta)
        self._n_updates += 1

    def predict(self, x: np.ndarray):
        return np.dot(x.T, self.theta)
   
    def update_var_est(self, residuals, rls_lambda = 0.99):
        return super().update_var_est(residuals, lam = rls_lambda)

def call_with_kwargs(func, *args, **kwargs):
    # Function to blindly call a function with any args and kwargs, passing only the kwargs that are in the function signature.
    sig = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return func(**kwargs)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **filtered_kwargs)

def get_func_kwargs(func, **kwargs):
    # Function to get only the kwargs that are in the function signature.
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
















##----------------------------------------------------------------------------------------------------------------------------------------------------##
##---------------------------------------------Autoregressive Moving Average with eXogenous Inputs----------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##

class ARMAX(Predictor):
    """
    ARMAX model using Recursive Prediction Error Method (RPEM) as described in the textbook.
    Model form: φ(B)Y_t = ω(B)u_t + θ(B)ε_t
    
    This implementation follows the recursive prediction error method (RPEM) from equations 11.62a-c,
    which requires filtering the regressor vector through the inverse MA polynomial.
    """

    params = ['ar_order', 'ma_order', 'exog_order', 'lambda_val']

    def __init__(self, order=(1, 0), exog_order=1, lambda_val=0.99,
                 exog_columns=None, x_columns=None, y_columns=None,
                 target_model_horizon=None):
        super().__init__(x_columns=x_columns, y_columns=y_columns)
        logger.info(f"Initializing ARMAX model with order={order}, exog_order={exog_order}, lambda_val={lambda_val}, target_model_horizon={target_model_horizon}")
        self.target_model_horizon = target_model_horizon

        # Parse model orders
        if isinstance(order, tuple) and len(order) >= 2:
            self.ar_order = int(order[0])
            self.ma_order = int(order[1])
        else:
            self.ar_order = 1
            self.ma_order = 0
        logger.debug(f"AR order: {self.ar_order}, MA order: {self.ma_order}")

        self.exog_order = max(0, int(exog_order))
        self.lambda_val = float(lambda_val)
        logger.debug(f"Exogenous order: {self.exog_order}, Lambda: {self.lambda_val}")

        if exog_columns is None:
            self.exog_columns = []
        elif isinstance(exog_columns, str):
            self.exog_columns = [exog_columns]
        else:
            self.exog_columns = list(exog_columns)
        logger.debug(f"Exogenous columns: {self.exog_columns}")

        self.n_exog_vars = len(self.exog_columns)
        self.total_exog_params = self.n_exog_vars * self.exog_order

        if y_columns is None:
            self.y_var_names = []
            warn("ARMAX initialized without y_columns. Target variable names may not be correctly set until the first fit if ForecastModel doesn't pass them.", UserWarning)
            logger.warning("ARMAX initialized without y_columns.")
        else:
            if isinstance(y_columns, pd.MultiIndex):
                self.y_var_names = list(y_columns.get_level_values(0).unique())
            elif isinstance(y_columns, (list, pd.Index)):
                if all(isinstance(item, str) for item in y_columns):
                    self.y_var_names = list(y_columns)
                else:
                    try:
                        self.y_var_names = list(pd.MultiIndex.from_tuples(y_columns).get_level_values(0).unique())
                    except Exception as e:
                        raise ValueError(f"y_columns format as list of tuples not recognized for MultiIndex conversion: {e}")
            elif isinstance(y_columns, str):
                self.y_var_names = [y_columns]
            else:
                raise ValueError("y_columns format not recognized. Should be string, list of strings, or MultiIndex.")
        logger.debug(f"Target y_var_names: {self.y_var_names}")

        # Initialize history for each target variable
        self.history_y = {name: pd.Series(dtype=float) for name in self.y_var_names}
        self.history_exog = {name: pd.DataFrame(columns=self.exog_columns, dtype=float) for name in self.y_var_names}
        self.history_eps = {name: pd.Series(dtype=float) for name in self.y_var_names}
        
        # NEW: History for gradient vectors (filtered regressors) - critical for RPEM
        self.history_psi = {name: [] for name in self.y_var_names}

        # Parameter vectors and information matrices
        self.param_vectors = {}
        self.R_matrices = {}

        for name in self.y_var_names:
            self._init_parameters(name)













    def _init_parameters(self, y_var_name):
        """Initializes the parameter vector (theta) and the inverse covariance matrix (P)."""
        n_params = self.ar_order + self.total_exog_params + self.ma_order
        logger.info(f"Initializing parameters for target '{y_var_name}'. Number of parameters: {n_params}")

        # Ensure the P_matrices dictionary exists on the instance
        if not hasattr(self, 'P_matrices'):
            self.P_matrices = {}

        if n_params == 0:
            warn(f"ARMAX model for {y_var_name} has no parameters (all orders are zero).", UserWarning)
            logger.warning(f"ARMAX model for {y_var_name} has no parameters.")
            self.param_vectors[y_var_name] = np.array([], dtype=float)
            self.P_matrices[y_var_name] = np.array([[]], dtype=float)
            return

        # Initialize parameter vector (theta) to zeros
        params = np.zeros(n_params)
        self.param_vectors[y_var_name] = params

        # Initialize P0 = R0^-1 with a large diagonal value.
        # This represents a large initial uncertainty (uninformative prior).
        P0 = np.eye(n_params) * 1e4
        
        # Store the initialized inverse covariance matrix P.
        self.P_matrices[y_var_name] = P0

        logger.debug(f"Initialized theta for '{y_var_name}': {params}")
        logger.debug(f"Initialized P matrix (inverse covariance) for '{y_var_name}':\n{P0}")














    def _prepare_exog(self, X_df: pd.DataFrame, horizon=None) -> pd.DataFrame | None:
        # This function remains mostly the same - it extracts exogenous variables from input data
        if not self.exog_columns or X_df is None or X_df.empty:
            logger.debug("_prepare_exog: No exogenous columns specified or X_df is empty. Returning None.")
            return None

        final_exog_data_series = {}
        for exog_name_to_find in self.exog_columns:
            selected_series_for_exog = None
            # Prioritize the model's own target horizon if available
            model_specific_horizon = self.target_model_horizon

            # Check if X_df columns are MultiIndex before proceeding
            if not isinstance(X_df.columns, pd.MultiIndex):
                if exog_name_to_find in X_df.columns:
                    selected_series_for_exog = X_df[exog_name_to_find]
            else:  # It is a MultiIndex
                target_col_tuple = (exog_name_to_find, model_specific_horizon)
                na_col_tuple = (exog_name_to_find, 'NA')

                if model_specific_horizon is not None and target_col_tuple in X_df.columns:
                    selected_series_for_exog = X_df[target_col_tuple]
                elif na_col_tuple in X_df.columns:  # Fallback to 'NA'
                    selected_series_for_exog = X_df[na_col_tuple]
                else:  # Further fallback: check if any numeric horizon exists for the base name
                    cols_for_var = [col for col in X_df.columns if col[0] == exog_name_to_find and isinstance(col[1], (int, float))]
                    if cols_for_var:
                        selected_series_for_exog = X_df[cols_for_var[0]]
                    else:  # Final fallback: if only one column exists with base name, regardless of horizon string
                        cols_for_var_any_horizon = [col for col in X_df.columns if col[0] == exog_name_to_find]
                        if len(cols_for_var_any_horizon) == 1:
                            selected_series_for_exog = X_df[cols_for_var_any_horizon[0]]
                        elif cols_for_var_any_horizon:  # Multiple matches, but none specific. Pick first.
                            selected_series_for_exog = X_df[cols_for_var_any_horizon[0]]

            if selected_series_for_exog is not None:
                if isinstance(selected_series_for_exog, pd.DataFrame):
                    # Ensure we take only the first column if DataFrame is returned
                    selected_series_for_exog = selected_series_for_exog.iloc[:, 0]
                final_exog_data_series[exog_name_to_find] = selected_series_for_exog.astype(float, errors='ignore')
        
        result = pd.DataFrame(final_exog_data_series, index=X_df.index) if final_exog_data_series else None
        if result is not None:
            logger.debug(f"_prepare_exog: Prepared exogenous data with shape {result.shape}:\n{result.head()}")
        else:
            logger.debug("_prepare_exog: No exogenous data prepared.")
        return result




















    def _construct_regressor(self, y_var_name: str) -> np.ndarray:
        """
        Constructs the regressor vector X_t for the ARMAX model.
        This vector includes past values of Y, exogenous inputs U, and past prediction errors ε.
        It corresponds to the textbook's pseudo-linear form X_t(θ). 
        """
        y_history = self.history_y.get(y_var_name, pd.Series(dtype=float))
        exog_hist_df = self.history_exog.get(y_var_name)
        if exog_hist_df is None:
            exog_hist_df = pd.DataFrame(columns=self.exog_columns, dtype=float)

        regressor_components = []

        # AR terms
        if self.ar_order > 0:
            for lag in range(1, self.ar_order + 1):
                try:
                    ar_value = -y_history.iloc[-lag] if lag <= len(y_history) else 0.0
                    regressor_components.append(ar_value)
                except IndexError:
                    regressor_components.append(0.0)

        # Exogenous terms
        if self.exog_order > 0 and self.n_exog_vars > 0:
            for exog_col_name in self.exog_columns:
                if exog_col_name in exog_hist_df.columns:
                    specific_exog_history = exog_hist_df[exog_col_name]
                    for lag in range(1, self.exog_order + 1):
                        try:
                            exog_value = specific_exog_history.iloc[-lag] if lag <= len(specific_exog_history) else 0.0
                            regressor_components.append(exog_value)
                        except IndexError:
                            regressor_components.append(0.0)
                else:
                    regressor_components.extend([0.0] * self.exog_order)

        # MA terms (Corrected based on instructions)
        if self.ma_order > 0:
            eps_history = self.history_eps.get(y_var_name, pd.Series(dtype=float))
            for lag in range(1, self.ma_order + 1):
                try:
                    ma_val = eps_history.iloc[-lag]
                except (IndexError, KeyError):
                    ma_val = 0.0
                regressor_components.append(ma_val)

        X_t_array = np.array(regressor_components, dtype=float)
        X_t_array = np.nan_to_num(X_t_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Adjust expected_len to include ma_order (Corrected based on instructions)
        expected_len = self.ar_order + self.total_exog_params + self.ma_order
        
        if len(X_t_array) != expected_len:
            actual_len = len(X_t_array)
            if actual_len < expected_len:
                X_t_array = np.pad(X_t_array, (0, expected_len - actual_len), 'constant', constant_values=0.0)
            else:
                X_t_array = X_t_array[:expected_len]
            warn(f"Regressor length mismatch for {y_var_name}. Expected {expected_len}, got {actual_len}. Adjusted.", UserWarning)
            logger.warning(f"_construct_regressor for '{y_var_name}': Regressor length mismatch. Expected {expected_len}, got {actual_len}. Adjusted.")

        if expected_len == 0 and not X_t_array.size == 0:
            X_t_array = np.array([], dtype=float)
        
        logger.debug(f"_construct_regressor for '{y_var_name}': X_t = {X_t_array}")
        return X_t_array










    def _filter_regressor(self, y_var_name: str, X_t: np.ndarray) -> np.ndarray:
        """
        Filters the regressor vector X_t to compute the gradient vector ψ_t.
        This is a direct implementation of the textbook's Equation (11.70),
        C(B)ψ_t(θ) = X_t(θ), which is solved recursively.

        The recursive formula is:
            ψ_t = X_t - c₁ψ_{t-1} - c₂ψ_{t-2} - ... - c_qψ_{t-q}

        Args:
            y_var_name: The name of the target variable.
            X_t: The UNFILTERED regressor vector [AR, Exog, MA terms] for the current time step.

        Returns:
            The filtered gradient vector ψ_t.
        """
        logger.debug(f"_filter_regressor for '{y_var_name}': Filtering X_t = {X_t}")
        
        # Retrieve the current MA coefficients (c₁ to c_q)
        theta = self.param_vectors[y_var_name]
        ma_start_idx = self.ar_order + self.total_exog_params
        ma_coeffs = theta[ma_start_idx : ma_start_idx + self.ma_order]
        logger.debug(f"  _filter_regressor: Using MA coeffs c = {ma_coeffs}")

        # Initialize psi_t with the current unfiltered regressor vector X_t
        psi_t = X_t.copy()

        # Recursively apply the filter 1/C(B)
        if self.ma_order > 0:
            psi_history = self.history_psi.get(y_var_name, [])
            for j in range(self.ma_order):
                if j < len(psi_history):
                    # Get the past gradient vector ψ_{t-(j+1)}
                    past_psi = psi_history[-(j + 1)]
                    psi_t -= ma_coeffs[j] * past_psi
                # If history is not available (at the beginning), the term is zero,
                # so no operation is needed.
        
        logger.debug(f"  _filter_regressor: Computed gradient vector ψ_t = {psi_t}")
        return psi_t












    def _one_step_prediction(self, y_var_name: str, X_t: np.ndarray) -> float:
        """
        Makes a one-step-ahead prediction using the UNFILTERED regressor vector (X_t).
        This correctly follows the pseudo-linear form described in the textbook's
        Equation (11.50): Ŷ_{t|t-1}(θ) = X_t^T(θ)θ.

        The filtered gradient (ψ_t) is used for updates, not for prediction.

        Args:
            y_var_name: The name of the target variable.
            X_t: The UNFILTERED regressor vector for the current time step.

        Returns:
            The one-step-ahead prediction.
        """
        if y_var_name not in self.param_vectors:
            warn(f"Parameter vector for {y_var_name} not found. Cannot predict.", UserWarning)
            logger.warning(f"_one_step_prediction for '{y_var_name}': Parameter vector not found.")
            return np.nan

        theta = self.param_vectors[y_var_name]

        # The prediction is the dot product of the parameters and the UNFILTERED regressor.
        if len(X_t) != len(theta):
            warn(f"Dimension mismatch for prediction: X_t ({len(X_t)}) vs theta ({len(theta)}) for {y_var_name}.", UserWarning)
            logger.error(f"_one_step_prediction for '{y_var_name}': Dimension mismatch X_t ({len(X_t)}) vs theta ({len(theta)}).")
            return np.nan

        try:
            y_pred = float(np.dot(X_t, theta))
            logger.debug(f"_one_step_prediction for '{y_var_name}':")
            logger.debug(f"  X_t (unfiltered) = {X_t}")
            logger.debug(f"  θ current        = {theta}")
            logger.debug(f"  y_pred (Ŷ_t|t-1) = {y_pred:.4f}")
            
            return y_pred if np.isfinite(y_pred) else np.nan
        except ValueError as e:
            warn(f"Error during dot product for prediction: {e}", UserWarning)
            logger.error(f"_one_step_prediction for '{y_var_name}': Error in dot product: {e}")
            return np.nan
        










    def _update_parameters_rpem(self, y_var_name: str, X_t: np.ndarray, prediction_error: float):
        """
        Updates parameters using the stable Recursive Prediction Error Method (RPEM).
        This implementation follows the textbook's equations (11.62a-c). It uses
        the filtered gradient vector (ψ_t) for the update to ensure stability.

        Args:
            y_var_name: The name of the target variable.
            X_t: The UNFILTERED regressor vector for the current time step.
            prediction_error: The calculated error (ε_t) for the current step.
        """
        eps_t = prediction_error
        logger.debug(f"_update_parameters_rpem for '{y_var_name}': Received X_t={X_t}, ε_t={eps_t:.4f}")

        # Initialization check
        if y_var_name not in self.param_vectors or y_var_name not in self.P_matrices:
            # This block should ideally not be hit if initialization is correct
            self._init_parameters(y_var_name)
            warn(f"Parameters for {y_var_name} were not initialized prior to update. Initializing now.", UserWarning)

        # Skip if no parameters to update or invalid inputs
        if len(self.param_vectors[y_var_name]) == 0 or not np.isfinite(eps_t):
            logger.debug("  _update_parameters_rpem: Skipping update (no parameters or invalid error).")
            return

        try:
            # 1. Retrieve previous state
            P_prev = self.P_matrices[y_var_name]
            theta_prev = self.param_vectors[y_var_name]

            # 2. Compute the filtered gradient vector ψ_t
            psi_t = self._filter_regressor(y_var_name, X_t)

            # 3. Compute the denominator for the gain vector K_t
            lam = self.lambda_val
            # Use ψ_t for the update, not X_t
            denom = lam + psi_t.T @ P_prev @ psi_t
            logger.debug(f"  _update_parameters_rpem: Denominator for K_t = {denom:.4e} (using ψ_t)")

            if abs(denom) < np.finfo(float).eps:
                warn(f"Denominator in RPEM update for {y_var_name} is near zero ({denom}). Skipping update.", UserWarning)
                return

            # 4. Compute the gain vector K_t
            K_t = (P_prev @ psi_t) / denom
            logger.debug(f"  _update_parameters_rpem: Gain vector K_t computed.")

            # 5. Update the parameter vector θ
            theta_new = theta_prev + K_t * eps_t
            logger.debug(f"  _update_parameters_rpem: Updated theta_new = {theta_new}")

            # 6. Update the inverse covariance matrix P
            P_new = (P_prev - np.outer(K_t, psi_t.T @ P_prev)) / lam
            
            # 7. Store the new state if it's numerically stable
            if np.isfinite(theta_new).all() and np.isfinite(P_new).all():
                self.param_vectors[y_var_name] = theta_new
                self.P_matrices[y_var_name] = P_new
                logger.debug("  _update_parameters_rpem: Stored new theta and P matrix.")
            else:
                warn(f"Parameter update for {y_var_name} resulted in NaN/inf. Not updating theta/P.", UserWarning)
                logger.warning(f"  _update_parameters_rpem for '{y_var_name}': NaN/inf detected. State not updated.")
            
            # 8. Append the calculated ψ_t to history for the next iteration's filter.
            self.history_psi[y_var_name].append(psi_t)
            
            # Limit history size to prevent memory issues
            max_history = max(self.ma_order, 50)
            if len(self.history_psi[y_var_name]) > max_history:
                self.history_psi[y_var_name].pop(0)

        except np.linalg.LinAlgError as e:
            warn(f"Linear algebra error in RPEM update for {y_var_name}: {e}. Skipping update.", UserWarning)
            logger.error(f"_update_parameters_rpem for '{y_var_name}': LinAlgError during update: {e}")














    def update_fit(self, X_arg: pd.DataFrame, Y_arg: pd.DataFrame, X_pred_arg: pd.DataFrame | None,
                  predict: bool = True, lambda_val: float = None, **kwargs) -> ForecastTuple:
        """
        Updates the ARMAX model with new data and returns the 1-step-ahead predictions
        generated *during* the fitting process.
        """
        logger.info(f"ARMAX update_fit called. Lambda from arg: {lambda_val}, from kwargs: {kwargs.get('lambda_val')}")

        # Process lambda_val from explicit parameter first, then from kwargs as fallback
        if lambda_val is not None:
            try:
                # Handle potential array/list input for lambda_val
                current_lambda = lambda_val[0] if isinstance(lambda_val, (np.ndarray, list, tuple)) and len(lambda_val) > 0 else lambda_val
                self.lambda_val = float(current_lambda)
            except (TypeError, ValueError) as e:
                warn(f"Invalid lambda_val provided: {lambda_val}. Using existing {self.lambda_val}. Error: {e}", UserWarning)
                logger.warning(f"Invalid lambda_val: {lambda_val}. Using {self.lambda_val}. Error: {e}")
        # Fallback to kwargs for backward compatibility or other parameters
        elif 'lambda_val' in kwargs:
             new_lambda = kwargs['lambda_val']
             try:
                 current_lambda_kw = new_lambda[0] if isinstance(new_lambda, (np.ndarray, list, tuple)) and len(new_lambda) > 0 else new_lambda
                 self.lambda_val = float(current_lambda_kw)
             except (TypeError, ValueError) as e:
                 warn(f"Invalid lambda_val provided in kwargs: {new_lambda}. Using existing {self.lambda_val}. Error: {e}", UserWarning)
                 logger.warning(f"Invalid lambda_val in kwargs: {new_lambda}. Using {self.lambda_val}. Error: {e}")
        logger.info(f"ARMAX update_fit: Effective lambda_val = {self.lambda_val}")


        # Determine output variable names if not set already
        pred_output_cols = []
        if not self.y_var_names and Y_arg is not None and not Y_arg.empty:
            if isinstance(Y_arg.columns, pd.MultiIndex):
                self.y_var_names = list(Y_arg.columns.get_level_values(0).unique())
            elif isinstance(Y_arg.columns, pd.Index):
                self.y_var_names = list(Y_arg.columns)
            warn(f"ARMAX y_var_names was empty, derived as {self.y_var_names} from Y_arg.", UserWarning)
            logger.info(f"ARMAX y_var_names derived: {self.y_var_names}")
            # Initialize history buffers and parameters for newly derived variables
            self.history_y = {name: pd.Series(dtype=float) for name in self.y_var_names}
            self.history_exog = {name: pd.DataFrame(columns=self.exog_columns, dtype=float) for name in self.y_var_names}
            self.history_eps = {name: pd.Series(dtype=float) for name in self.y_var_names}
            self.history_psi = {name: [] for name in self.y_var_names}
            for name in self.y_var_names:
                if name not in self.param_vectors: self._init_parameters(name)

        # Determine output columns based on target_model_horizon (defaulting to 1 if None)
        if self.y_var_names:
            output_horizon = self.target_model_horizon if self.target_model_horizon is not None else 1
            pred_output_cols = [(name, output_horizon) for name in self.y_var_names]
        else: # Fallback if y_var_names couldn't be determined (should ideally not happen if Y_arg provided)
            if hasattr(self, 'y_columns') and self.y_columns is not None and len(self.y_columns) > 0:
                pred_output_cols = list(self.y_columns)
            else:
                pred_output_cols = [] # No specific columns targeted
        logger.debug(f"ARMAX update_fit: Prediction output columns determined: {pred_output_cols}")


        # Create output DataFrame for predictions made *during fitting*
        y_hat_cols_multi_index = pd.MultiIndex.from_tuples(
            pred_output_cols, names=['Variable', 'Horizon']
        )

        # Use the index of the input Y data for the output DataFrame
        output_index_ref = Y_arg
        output_index = output_index_ref.index if output_index_ref is not None and not output_index_ref.empty else pd.Index([])

        Y_hat_fitting = new_fc(index=output_index, columns=y_hat_cols_multi_index, dtype=float)
        Y_hat_fitting[:] = np.nan # Initialize with NaN

        # Prepare exogenous data needed for the fitting loop (aligns with Y_arg)
        exog_data_fit = self._prepare_exog(X_arg)
        logger.debug(f"ARMAX update_fit: Exogenous data for fitting prepared. Shape: {exog_data_fit.shape if exog_data_fit is not None else 'None'}")


        # Process each target variable
        for y_var_name in self.y_var_names:
            logger.info(f"ARMAX update_fit: Processing target variable '{y_var_name}'")
            y_series_for_var = None
            target_col_for_y = None

            # Find the appropriate column for this target variable in Y_arg
            if Y_arg is not None and not Y_arg.empty:
                if isinstance(Y_arg.columns, pd.MultiIndex):
                    target_y_col_tuple = (y_var_name, self.target_model_horizon)
                    na_y_col_tuple = (y_var_name, 'NA')

                    if self.target_model_horizon is not None and target_y_col_tuple in Y_arg.columns:
                        y_series_for_var = Y_arg[target_y_col_tuple]
                        target_col_for_y = target_y_col_tuple
                    elif na_y_col_tuple in Y_arg.columns:
                        y_series_for_var = Y_arg[na_y_col_tuple]
                        target_col_for_y = na_y_col_tuple
                    else: # Fallback: Find first column matching the variable name
                        cols_for_var = [col for col in Y_arg.columns if col[0] == y_var_name]
                        if cols_for_var:
                            y_series_for_var = Y_arg[cols_for_var[0]]
                            target_col_for_y = cols_for_var[0]
                elif y_var_name in Y_arg.columns: # Handle single-level column index
                    y_series_for_var = Y_arg[y_var_name]
                    target_col_for_y = y_var_name
            
            logger.debug(f"ARMAX update_fit for '{y_var_name}': Selected Y series from Y_arg using column {target_col_for_y}")


            # Process the data for the current target variable if found
            if y_series_for_var is not None and not y_series_for_var.empty:
                # Ensure we're working with a 1D series
                if isinstance(y_series_for_var, pd.DataFrame):
                    if y_series_for_var.shape[1] == 1:
                        y_series_for_var = y_series_for_var.iloc[:,0]
                    else:
                        warn(f"Target variable {y_var_name} selection resulted in >1 column. Taking first.", UserWarning)
                        logger.warning(f"Target variable '{y_var_name}' selection resulted in >1 column. Taking first.")
                        y_series_for_var = y_series_for_var.iloc[:,0]

                # Attempt conversion to float
                try:
                    y_series_for_var = y_series_for_var.astype(float)
                except Exception as e:
                    warn(f"Could not convert target series {y_var_name} to float: {e}", UserWarning)
                    logger.error(f"Could not convert target series '{y_var_name}' to float: {e}. Skipping this variable.")
                    continue # Skip this target variable if conversion fails

                # Drop NaN values from the target series for processing
                y_series_processed = y_series_for_var.dropna()
                logger.debug(f"ARMAX update_fit for '{y_var_name}': Processing {len(y_series_processed)} non-NaN Y values.")


                # Process each valid data point sequentially
                for i in range(len(y_series_processed)):
                    current_y_actual_original_idx = y_series_processed.index[i]
                    current_y_actual_val = float(y_series_processed.iloc[i])
                    logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i} (index {current_y_actual_original_idx}): Actual Y_t = {current_y_actual_val:.4f}")


                    # --- Check if enough history exists BEFORE predicting ---
                    # Simplified check: needs enough Y history for AR, Eps history for MA, Exog history for Exog
                    min_hist_needed_ar_ma = max(self.ar_order, self.ma_order)
                    min_hist_needed_exog = self.exog_order
                    can_predict_update = (len(self.history_y[y_var_name]) >= min_hist_needed_ar_ma and
                                          len(self.history_eps[y_var_name]) >= self.ma_order and
                                          (self.total_exog_params == 0 or len(self.history_exog[y_var_name]) >= min_hist_needed_exog) )
                    # ------------------------------------------------------
                    logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: can_predict_update = {can_predict_update}")


                    X_t = None # Initialize X_t
                    y_pred = np.nan # Initialize y_pred

                    if can_predict_update:
                        # Construct the regressor vector X_t using history UP TO T-1
                        X_t = self._construct_regressor(y_var_name)
                        logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: Constructed X_t = {X_t}")


                        # Make one-step prediction using parameters FROM T-1
                        y_pred = self._one_step_prediction(y_var_name, X_t)
                        logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: Predicted Y_hat_t|t-1 = {y_pred:.4f}")

                    
                    # Store the 1-step-ahead prediction (or NaN)
                    y_pred_float = float(y_pred) if np.isfinite(y_pred) else np.nan
                    for output_col in [col for col in pred_output_cols if col[0] == y_var_name]:
                         if output_col in Y_hat_fitting.columns:
                              Y_hat_fitting.loc[current_y_actual_original_idx, output_col] = y_pred_float

                    # Calculate prediction error
                    pred_error = np.nan
                    if np.isfinite(y_pred) and np.isfinite(current_y_actual_val):
                        pred_error = current_y_actual_val - y_pred
                    elif np.isfinite(current_y_actual_val): # Handle case where only prediction failed
                         # Error is undefined if prediction is NaN, but needed for history. Use 0? Or actual? Let's use 0.
                         pred_error = 0.0 # Or potentially current_y_actual_val, but 0 is safer for MA term
                    logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: Prediction error eps_t = {pred_error:.4f}")


                    # Update parameters using RPEM only if error is valid and X_t was constructible
                    if np.isfinite(pred_error) and can_predict_update and X_t is not None:
                        logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: Calling _update_parameters_rpem.")
                        self._update_parameters_rpem(y_var_name, X_t, pred_error)

                    # --- History Updates ---
                    # Update history_y with ACTUAL value y_t
                    self.history_y[y_var_name] = pd.concat([
                        self.history_y.get(y_var_name, pd.Series(dtype=float)), # Ensure Series exists
                        pd.Series([current_y_actual_val], index=[current_y_actual_original_idx], dtype=float)
                    ]).pipe(lambda s: s[~s.index.duplicated(keep='last')])

                    # Update history_exog with ACTUAL exog values for time t
                    if exog_data_fit is not None:
                         current_exog_hist = self.history_exog.get(y_var_name, pd.DataFrame(columns=self.exog_columns, dtype=float)) # Ensure DataFrame exists
                         if current_y_actual_original_idx in exog_data_fit.index:
                             exog_row_for_history = exog_data_fit.loc[[current_y_actual_original_idx]]
                             self.history_exog[y_var_name] = pd.concat([
                                 current_exog_hist,
                                 exog_row_for_history
                             ]).pipe(lambda df: df[~df.index.duplicated(keep='last')])
                         # Add fallback logic if needed, e.g., positional matching or NaN padding
                         elif i < len(exog_data_fit): # Fallback by position if index mismatch and length allows
                            exog_row_val_pos = exog_data_fit.iloc[[i]]
                            current_exog_hist_pos = pd.DataFrame(exog_row_val_pos.values,
                                                               index=[current_y_actual_original_idx],
                                                               columns=exog_data_fit.columns)
                            self.history_exog[y_var_name] = pd.concat([
                                current_exog_hist, current_exog_hist_pos
                            ]).pipe(lambda df: df[~df.index.duplicated(keep='last')])

                    # Update history_eps with calculated prediction error (use 0 if it was NaN)
                    error_to_store = pred_error if np.isfinite(pred_error) else 0.0
                    self.history_eps[y_var_name] = pd.concat([
                        self.history_eps.get(y_var_name, pd.Series(dtype=float)), # Ensure Series exists
                        pd.Series([error_to_store], index=[current_y_actual_original_idx], dtype=float)
                    ]).pipe(lambda s: s[~s.index.duplicated(keep='last')])
                logger.debug(f"ARMAX update_fit for '{y_var_name}', step {i}: History updated. Y len: {len(self.history_y[y_var_name])}, Exog len: {len(self.history_exog[y_var_name])}, Eps len: {len(self.history_eps[y_var_name])}, Psi len: {len(self.history_psi[y_var_name])}")

        # Return the DataFrame containing the 1-step-ahead predictions made *during* fitting
        logger.info(f"ARMAX update_fit completed. Returning Y_hat_fitting with shape {Y_hat_fitting.shape}")
        logger.debug(f"Y_hat_fitting content:\n{Y_hat_fitting.head()}")
        return ForecastTuple(Y_hat_fitting)



##----------------------------------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##










##----------------------------------------------------------------------------------------------------------------------------------------------------##
##---------------------------------------------------Online Support Vector Regression-----------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##
class OnlineSVR(Predictor):
    """
    Online Support Vector Regression predictor based on incremental/decremental learning.
    """
    params = ['C', 'epsilon', 'gamma', 'threshold', 'kernel', 'max_iter', 'tol'] # Added missing params

    def __init__(self, C=1.0, epsilon=0.1, gamma=0.1, kernel='rbf',
                 x_columns=None, y_columns=None, max_iter=70, tol=1e-3, threshold=None):
        super().__init__(x_columns=x_columns, y_columns=y_columns)
        self.C = float(C)
        self.gamma = float(gamma)
        self.kernel_type = kernel
        self.max_iter = int(max_iter)
        self.threshold = int(threshold) if threshold is not None else None

        # --- Corrected Tolerances ---
        # The user-provided 'tol' is for the minimum step size, not for KKT checks.
        # A separate, smaller tolerance is used for numerical comparisons to strictly
        # enforce the KKT conditions.
        self.epsilon  = float(epsilon)   # Epsilon-tube half-width
        self.step_tol = float(tol)        # Minimum significant step size for Δθ in _calculate_max_step
        self.kkt_tol  = 1e-8              # Strict numerical tolerance for all KKT and matrix stability checks

        # --- State Attributes ---
        self.X_internal: list[np.ndarray] = []
        self.y_internal: list[float] = []
        self.alphas: list[float] = []
        self.alphas_star: list[float] = []
        self.bias: float = 0.0

        # Sets for point classification
        self.E: set[int] = set()
        self.E_star: set[int] = set()
        self.R: set[int] = set()
        self.S_list: list[int] = []

        # Use R_inv for the inverse KKT matrix
        self.Q: np.ndarray | None = None
        self.R_inv: np.ndarray | None = None # Corresponds to matrix 'R' in the paper

        self._feature_dim: int | None = None
        self._initialized: bool = False

        # Internal state tracking
        self.active_processing_idx: int | None = None
        self._just_removed_last_support: int | None = None

        # Tolerances and constants are now unified under kkt_tol for consistency
        self.delta_jitter = self.kkt_tol
        self.gamma_pivot_tol = self.kkt_tol
        self.r_inv_pivot_tol = self.kkt_tol
        self.MACHINE_EPS = np.finfo(float).eps

        # Backward compatibility alias
        self.S = self.S_list



















    def set_state(self, state: dict):
        try:
            self.X_internal = copy.deepcopy(state.get('X_internal', []))
            self.y_internal = copy.deepcopy(state.get('y_internal', []))
            self.alphas = copy.deepcopy(state.get('alphas', []))
            self.alphas_star = copy.deepcopy(state.get('alphas_star', []))
            self.bias = state.get('bias', 0.0)
            self.S_list = copy.deepcopy(state.get('S_list', []))
            self.E = copy.deepcopy(state.get('E', set()))
            self.R = copy.deepcopy(state.get('R', set()))
            self.R_inv = copy.deepcopy(state.get('R_inv')) 
            self.Q = copy.deepcopy(state.get('Q'))      
            self._feature_dim = state.get('_feature_dim') 
            self._initialized = state.get('_initialized', False)

            n_points = len(self.X_internal)
            if not (n_points == len(self.y_internal) == len(self.alphas) == len(self.alphas_star)):
                raise ValueError("Loaded state arrays have inconsistent lengths.")

            if self._initialized and self._feature_dim is None and self.X_internal:
                first_pt = np.asarray(self.X_internal[0])
                self._feature_dim = first_pt.shape[0] if first_pt.ndim > 0 and first_pt.shape[0] > 0 else 1
            
            if self.S_list and (self.R_inv is None or self.R_inv.shape[0] != len(self.S_list) + 1):
                logging.warning("set_state: S_list is not empty but R_inv is None or inconsistent. Recomputing R_inv.")
                self._compute_R_inv_initial()
            elif not self.S_list:
                if self.R_inv is not None:
                     logging.info("set_state: S_list is empty, ensuring R_inv is None.")
                     self.R_inv = None
                     self.Q = None
        except Exception as e:
            logging.warning(f"Error setting SVR state: {e}. State might be corrupted or partially set.")





















    def initialize_model(self, x1: np.ndarray, y1: float, x2: np.ndarray, y2: float):
        """
        Initializes the SVR model with the first two data points, attempting to place
        them exactly on the epsilon-tube boundaries as support vectors.
        Includes a check for matrix degeneracy before attempting a two-point solution.
        """
        # ─────── RAW‐INPUT LOGGING ───────
        logging.debug(f"initialize_model: raw x1        = {x1}")
        logging.debug(f"initialize_model: raw x2        = {x2}")
        logging.debug(f"initialize_model: raw y1, y2    = {y1}, {y2}")
    
        # ─────── DUPLICATE‐DETECTION LOGGING ───────
        if np.allclose(x1, x2, atol=1e-8):
            logging.error("initialize_model: *** DUPLICATE FEATURE VECTORS *** x1 == x2")
        if abs(y1 - y2) < 1e-8:
            logging.error("initialize_model: *** DUPLICATE TARGET VALUES *** y1 == y2")
    
        # ─────── ARRAY‐SHAPE LOGGING ───────
        x1_arr = np.asarray(x1).reshape(1, -1)
        x2_arr = np.asarray(x2).reshape(1, -1)
        logging.debug(f"initialize_model: x1_arr shape = {x1_arr.shape}, contents = {x1_arr}")
        logging.debug(f"initialize_model: x2_arr shape = {x2_arr.shape}, contents = {x2_arr}")
        
        logging.info(f"OnlineSVR.initialize_model: Initializing with C={self.C}, epsilon={self.epsilon}")
    
        kkt_check_tol = 1e-8
    
        if self._feature_dim is None:
            if x1_arr.shape[1] > 0: self._feature_dim = x1_arr.shape[1]
            elif x2_arr.shape[1] > 0: self._feature_dim = x2_arr.shape[1]
            else:
                logging.error("initialize_model: Cannot determine feature dimension from input points.")
                self._initialized = False
                return
    
        self.X_internal = [
            x1_arr.squeeze(axis=0),
            x2_arr.squeeze(axis=0),
        ]
        logging.debug(
            "initialize_model: X_internal built →\n"
            f"  [0] = {self.X_internal[0]!r}\n"
            f"  [1] = {self.X_internal[1]!r}"
        )
        self.y_internal = [y1, y2]
        self.alphas = [0.0, 0.0]
        self.alphas_star = [0.0, 0.0]
        self.bias = 0.0
    
        K11_val = self._kernel(x1_arr, x1_arr)[0, 0]
        K12_val = self._kernel(x1_arr, x2_arr)[0, 0]
        K22_val = self._kernel(x2_arr, x2_arr)[0, 0]
        logging.info(f"  initialize_model: Initial kernel values: k11={K11_val:.6e}, k22={K22_val:.6e}, k12={K12_val:.6e}")
        logging.info(f"  initialize_model: RBF gamma parameter = {self.gamma:.6e}")
    
        successful_init_S_S = False
        final_theta_clipped_p0, final_theta_clipped_p1 = np.nan, np.nan
        final_b = np.nan
    
        # --- Augmented two-point KKT solve (Section 3.1.1 of the paper) ---
        # 1) Regularized kernel block with 1/C on the diagonal
        H = np.array([[K11_val, K12_val],
                      [K12_val, K22_val]]) + (1.0/self.C)*np.eye(2)
        logging.debug("initialize_model: H (with 1/C on diag) =\n%s", H)
    
        # 2+3) Try both sign‐scenarios (s0,s1)=(+1,−1) and (−1,+1), building Q_init per scenario
        sign_scenarios = [(1, -1), (-1, 1)]
        for s0, s1 in sign_scenarios:
            # build a fresh signed KKT matrix each trial (Section 3.1.1 Eqs. 11 & 13)
            Q_init = np.zeros((3, 3))
            Q_init[1:, 1:] = H
            Q_init[0,    1:] = [s0,   s1]
            Q_init[1:, 0   ] = [s0,   s1]
            logging.debug(
                "initialize_model: Q_init (s0=%d, s1=%d) =\n%s",
                s0, s1, Q_init
            )
            logging.info(f"initialize_model: Trying augmented KKT scenario (s0={s0}, s1={s1})")
            rhs = np.array([
                0.0,
                y1 + self.epsilon * s0,
                y2 + self.epsilon * s1
            ])
            #–– LOGGING: show the sign choices, ε and raw y’s
            logging.debug(
                "INIT_DEBUG ▶ y1=%.6e, y2=%.6e, ε=%.6e, s0=%d, s1=%d",
                y1, y2, self.epsilon, s0, s1
            )
            #–– LOGGING: the RHS you’re solving against
            logging.debug(
                "INIT_DEBUG ▶ RHS vector = %s",
                rhs
            )
            try:
                sol = np.linalg.solve(Q_init, rhs)
                #–– LOGGING: raw solution of [b_trial, β0, β1]
                logging.debug(
                    "INIT_DEBUG ▶ raw sol = %s", sol
                )
                #–– LOGGING: residual to check exact KKT satisfaction
                res = Q_init.dot(sol) - rhs
                logging.debug(
                    "INIT_DEBUG ▶ residual norm = %.6e, components = %s",
                    np.linalg.norm(res), res
                )
            except np.linalg.LinAlgError:
                logging.warning(f"  Augmented KKT scenario (s0={s0},s1={s1}) singular → skipping")
                continue
            
            b_trial, beta0, beta1 = sol
            # START of user-requested logging
            logging.debug(
                "PRE-CLIP_SOL ▶ raw solve b_trial=%.6e, beta0=%.6e, beta1=%.6e",
                b_trial, beta0, beta1
            )
            # compute margins on the unclipped beta’s
            h1_raw = K11_val*beta0 + K12_val*beta1 + b_trial - y1
            h2_raw = K12_val*beta0 + K22_val*beta1 + b_trial - y2
            logging.debug(
                "PRE-CLIP_MARGIN ▶ h1_raw=%.6e (|h1_raw|-ε=%.6e), h2_raw=%.6e (|h2_raw|-ε=%.6e)",
                h1_raw, abs(abs(h1_raw) - self.epsilon),
                h2_raw, abs(abs(h2_raw) - self.epsilon)
            )
            # check whether they’d already satisfy the ε‐tube
            logging.debug(
                "PRE-CLIP_MARGIN_OK ▶ margin_ok_0(raw)=%s, margin_ok_1(raw)=%s",
                abs(abs(h1_raw) - self.epsilon) <= kkt_check_tol,
                abs(abs(h2_raw) - self.epsilon) <= kkt_check_tol
            )
            # END of user-requested logging
            logging.info(
                "initialize_model: raw sol b_trial=%.6e, beta0=%.6e, beta1=%.6e",
                b_trial, beta0, beta1
            )
            # Clip to [−C, C]
            beta0_c = np.clip(beta0, -self.C, self.C)
            beta1_c = np.clip(beta1, -self.C, self.C)
            logging.debug(
                "initialize_model: clipped betas → beta0_c=%.6e, beta1_c=%.6e",
                beta0_c, beta1_c
            )
            h1 = K11_val*beta0_c + K12_val*beta1_c + b_trial - y1
            h2 = K12_val*beta0_c + K22_val*beta1_c + b_trial - y2
            
            # --- Added targeted logging ---
            logging.debug(
                "KKT_DEBUG ▶ scenario s0=%d, s1=%d · b_trial=%.6e · β0=%.6e · β1=%.6e",
                s0, s1, b_trial, beta0_c, beta1_c
            )
            logging.debug(
                "MARGIN_DEBUG ▶ h1=%.6e (|h1|-ε=%.6e) · h2=%.6e (|h2|-ε=%.6e)",
                h1, abs(abs(h1) - self.epsilon),
                h2, abs(abs(h2) - self.epsilon)
            )
            logging.debug(
                "MARGIN_CHECK ▶ margin_ok_0=%s · margin_ok_1=%s",
                abs(abs(h1) - self.epsilon) <= kkt_check_tol,
                abs(abs(h2) - self.epsilon) <= kkt_check_tol
            )
    
            # --- End of added logging ---
            
            logging.debug(
                "initialize_model: margins → h1=%.6e, h2=%.6e (should be ±%.6e)",
                h1, h2, self.epsilon
            )
    
            # Check that they’re true support (KKT)
            is_p0_S = (abs(beta0_c) >= kkt_check_tol) and (abs(beta0_c) <= self.C - kkt_check_tol)
            is_p1_S = (abs(beta1_c) >= kkt_check_tol) and (abs(beta1_c) <= self.C - kkt_check_tol)
            
            # --- Added targeted logging ---
            logging.debug(
                "KKT_CHECK ▶ is_p0_S=%s · is_p1_S=%s",
                is_p0_S, is_p1_S
            )
            accept = (is_p0_S and is_p1_S and
                      abs(abs(h1) - self.epsilon) <= kkt_check_tol and
                      abs(abs(h2) - self.epsilon) <= kkt_check_tol)
            logging.debug(
                "SCENARIO_ACCEPT ▶ scenario s0=%d, s1=%d → accept=%s",
                s0, s1, accept
            )
            # --- End of added logging ---
    
            if accept:
                logging.info(
                    "initialize_model: Accepting two‐point basis s0=%d, s1=%d · β0=%.6e · β1=%.6e · bias=%.6e",
                    s0, s1, beta0_c, beta1_c, b_trial
                )
                final_theta_clipped_p0 = beta0_c
                final_theta_clipped_p1 = beta1_c
                final_b = b_trial
                successful_init_S_S = True
                break
            
        if successful_init_S_S:
            logging.info(f"initialize_model: Successful two-point S-S basis found. "
                         f"θ0_clip={final_theta_clipped_p0:.3f}, θ1_clip={final_theta_clipped_p1:.3f}, bias={final_b:.3f}.")
            self.bias = final_b
            self.alphas[0] = max(0.0, min(self.C, final_theta_clipped_p0))
            self.alphas_star[0] = max(0.0, min(self.C, -final_theta_clipped_p0))
            self.alphas[1] = max(0.0, min(self.C, final_theta_clipped_p1))
            self.alphas_star[1] = max(0.0, min(self.C, -final_theta_clipped_p1))
    
            self.S_list = [0, 1]
            self.E.clear(); self.R.clear()
        else:
            # Fallback to single-point initialization
            logging.warning("initialize_model: S-S initialization failed or was skipped. Reverting to single-point initialization.")
            logging.debug(
                "initialize_model: FALLBACK_DEBUG ▶ h0=-y1=%.6e, h1=-y2=%.6e",
                -y1, -y2
            )
            ll0, ll1 = abs(-y1), abs(-y2)
            chosen_idx = 0 if ll0 >= ll1 else 1
            logging.debug(
                "initialize_model: FALLBACK_DEBUG ▶ chosen_idx=%d based on larger |h|",
                chosen_idx
            )
            y_ch, K_ch = (y1, K11_val) if chosen_idx == 0 else (y2, K22_val)
            h_ch = -y_ch
    
            sign_theta = np.sign(-h_ch) if abs(h_ch) > kkt_check_tol else 1.0
            theta_init = sign_theta * min(self.C, abs(h_ch) / (K_ch + 1.0/self.C + kkt_check_tol))
            theta_init = np.clip(theta_init, -(self.C - kkt_check_tol), (self.C - kkt_check_tol))
            
            logging.debug(
                "initialize_model: FALLBACK_DEBUG ▶ theta_init=%.6e",
                theta_init
            )
            
            if chosen_idx == 0:
                self.alphas[0] = max(0.0, theta_init)
                self.alphas_star[0] = max(0.0, -theta_init)
                self.alphas[1], self.alphas_star[1] = 0.0, 0.0
            else:
                self.alphas[1] = max(0.0, theta_init)
                self.alphas_star[1] = max(0.0, -theta_init)
                self.alphas[0], self.alphas_star[0] = 0.0, 0.0
    
            self.S_list = [chosen_idx]
            self.bias = y_ch - theta_init * K_ch - self.epsilon * np.sign(h_ch)
    
            self.E.clear(); self.R.clear()
            other_idx = 1 - chosen_idx
            self.R.add(other_idx)
    
        self._compute_R_inv_initial()
        self._initialized = True
        logging.info(f"initialize_model: Initialization complete. "
                     f"_initialized={self._initialized}, S_list={self.S_list}, "
                     f"R_inv.shape={self.R_inv.shape if self.R_inv is not None else 'None'}")
    























    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Computes the kernel matrix between X1 and X2.
        This version includes the corrected RBF kernel implementation.
        """
        # Attempt to infer feature_dim if not set and inputs allow
        if self._feature_dim is None:
            if X1.ndim == 1 and X1.size > 0:
                self._feature_dim = X1.shape[0]
            elif X2.ndim == 1 and X2.size > 0 and X1.ndim == 0:  # If X1 is empty or scalar
                self._feature_dim = X2.shape[0]
            elif X1.ndim == 2 and X1.shape[1] > 0:
                self._feature_dim = X1.shape[1]
            elif X2.ndim == 2 and X2.shape[1] > 0 and X1.shape[0] == 0:  # If X1 is empty 2D array
                self._feature_dim = X2.shape[1]

        # Ensure inputs are 2D for consistent calculations
        if X1.ndim == 1:
            X1 = X1.reshape(1, self._feature_dim if self._feature_dim is not None and self._feature_dim > 0 else -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, self._feature_dim if self._feature_dim is not None and self._feature_dim > 0 else -1)

        # Handle empty inputs to avoid errors
        if X1.size == 0 or X2.size == 0:
            return np.empty((X1.shape[0], X2.shape[0]))

        # --- CORRECTED KERNEL LOGIC ---
        if self.kernel_type == 'rbf':
            if self.gamma is None:
                raise ValueError("Gamma (self.gamma) is not set for RBF kernel")
            logging.debug(f"_kernel: RBF with gamma = {self.gamma}")
            try:
                # Pairwise squared distances using broadcasting as requested.
                # This computes the squared Euclidean distance between each vector in X1 and each in X2.
                D = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
                K = np.exp(-self.gamma * D)

                # --- Logging and validation for RBF kernel ---
                logging.debug(f"_kernel: pairwise_dists_sq (D) with shape {D.shape}")
                # Allow for small floating point inaccuracies around 1.0
                if np.any(K > 1.0 + 1e-9):
                    logging.warning(f"_kernel: K contains values > 1.0! Max K value = {np.max(K):.6e}")
                if np.any(K < 0.0):
                    logging.warning(f"_kernel: K contains negative values! Min K value = {np.min(K):.6e}")
                if np.any(np.isnan(K)):
                    logging.error(f"_kernel: K contains NaN values!")
                logging.debug(f"_kernel: K range = [{np.min(K):.6e}, {np.max(K):.6e}], mean = {np.mean(K):.6e}")
                return K
            except Exception as e:
                logging.error(f"Error in RBF kernel computation (X1 shape: {X1.shape}, X2 shape: {X2.shape}, feature_dim: {self._feature_dim}): {e}", exc_info=True)
                return np.full((X1.shape[0], X2.shape[0]), np.nan)

        elif self.kernel_type == 'linear':
            try:
                K = np.dot(X1, X2.T)
                # --- Logging and validation for linear kernel ---
                if X1.shape == X2.shape and np.allclose(X1, X2):
                    logging.debug(f"_kernel ▶ diag-call: first few diag entries = {np.diag(K)[:3]!r}")
                else:
                    logging.debug(f"_kernel ▶ shapes {X1.shape}×{X2.shape}, K[0,0]={K[0,0]:.3e}")
                if np.any(np.isnan(K)):
                    logging.error(f"_kernel: K contains NaN values!")
                return K
            except Exception as e:
                logging.error(f"Error in linear kernel computation (X1 shape: {X1.shape}, X2 shape: {X2.shape}, feature_dim: {self._feature_dim}): {e}", exc_info=True)
                return np.full((X1.shape[0], X2.shape[0]), np.nan)
        else:
            raise NotImplementedError(f"Kernel type '{self.kernel_type}' not supported")


























    def _compute_R_inv_initial(self) -> None:
        """
        Builds/rebuilds the INVERSE of the augmented KKT matrix Q_aug.

        [CORRECTED] This method now avoids Cholesky decomposition, which fails because
        the Q_aug matrix is not positive-definite. Instead, it computes the inverse
        directly using np.linalg.inv(). The result is stored in self.R_inv, which
        corresponds to the matrix 'R' in Martin's paper.
        """
        logging.debug(f"_compute_R_inv_initial ▶ ENTRY: S_list={self.S_list}")
        
        s_list = self.S_list.copy()
        m = len(s_list)

        # Clear old factor attributes if they exist, to avoid confusion.
        if hasattr(self, 'R_factor'): delattr(self, 'R_factor')
        if hasattr(self, 'R_factor_inv'): delattr(self, 'R_factor_inv')

        if m == 0:
            logging.warning("  _compute_R_inv_initial: S_list is empty → clearing R_inv and Q")
            self.R_inv = None
            self.Q = None
            return

        try:
            # 1. Construct the Q_aug matrix exactly as before.
            K_SS_pure = self._extract_pure_K_SS(s_list)
            H = K_SS_pure + (1.0 / self.C) * np.eye(m)
            
            Q_aug = np.zeros((m + 1, m + 1), dtype=float)
            Q_aug[1:, 1:] = H
            ones_m = np.ones(m, dtype=float)
            Q_aug[0, 1:] = ones_m
            Q_aug[1:, 0] = ones_m
            Q_aug[0, 0] = 0  # Not positive-definite
            Q_aug[1:, 1:] += self.delta_jitter * np.eye(m)

            self.Q = Q_aug.copy()
            logging.debug(f"_compute_R_inv_initial: Constructed Q_aug matrix (shape: {self.Q.shape})")

            # 2. [CORRECTION] Perform direct inversion of Q_aug.
            self.R_inv = np.linalg.inv(self.Q)
            
            logging.info(f"  _compute_R_inv_initial: Direct inversion successful. R_inv shape: {self.R_inv.shape}")
            logging.debug(f"  _compute_R_inv_initial: R_inv condition number = {np.linalg.cond(self.R_inv):.3e}")

        except np.linalg.LinAlgError as e:
            logging.error(f"  _compute_R_inv_initial: Failed during rebuild (inversion): {e}. Model state may be invalid.", exc_info=True)
            self.R_inv = None
            self.Q = None
        
        logging.debug("Exiting _compute_R_inv_initial")

















    def _extract_pure_K_SS(self, s_list_to_extract: list[int]) -> np.ndarray:
        """
        Helper to extract the pure K_SS kernel matrix (no diagonal jitter added here)
        for the support vectors specified in s_list_to_extract.
        """
        m_extract = len(s_list_to_extract)
        if m_extract == 0:
            return np.array([[]], dtype=float) # Or np.empty((0,0))

        K_SS_pure = np.zeros((m_extract, m_extract), dtype=float)
        
        # Prepare formatted S-vectors once
        formatted_S_vectors_extract = []
        for i_sv_idx in s_list_to_extract:
            x_arr = np.asarray(self.X_internal[i_sv_idx])
            if x_arr.ndim == 0: x_reshaped = x_arr.reshape(1, 1)
            elif x_arr.ndim == 1: x_reshaped = x_arr.reshape(1, -1)
            elif x_arr.ndim == 2 and x_arr.shape[0] == 1: x_reshaped = x_arr
            else: x_reshaped = x_arr.reshape(1, -1) # Fallback
            formatted_S_vectors_extract.append(x_reshaped)

        for idx_row, i_row_sv_idx in enumerate(s_list_to_extract):
            x_row_reshaped = formatted_S_vectors_extract[idx_row]
            for idx_col, i_col_sv_idx in enumerate(s_list_to_extract):
                if idx_col < idx_row: # Kernel matrix is symmetric
                    K_SS_pure[idx_row, idx_col] = K_SS_pure[idx_col, idx_row]
                else:
                    x_col_reshaped = formatted_S_vectors_extract[idx_col]
                    kernel_val = self._kernel(x_row_reshaped, x_col_reshaped)
                    K_SS_pure[idx_row, idx_col] = float(np.ravel(kernel_val)[0])

        logging.debug(f"  _extract_pure_K_SS: K_SS_pure shape = {K_SS_pure.shape}")
        logging.debug(f"  _extract_pure_K_SS: K_SS_pure diagonal = {np.diag(K_SS_pure)}")
        logging.debug(f"  _extract_pure_K_SS: K_SS_pure range = [{np.min(K_SS_pure):.6e}, {np.max(K_SS_pure):.6e}]")
        if K_SS_pure.shape[0] > 0:
            try:
                cond_num = np.linalg.cond(K_SS_pure)
                logging.debug(f"  _extract_pure_K_SS: K_SS_pure condition number = {cond_num:.6e}")
            except np.linalg.LinAlgError:
                logging.warning("  _extract_pure_K_SS: Could not compute condition number (matrix may be singular).")
        
        return K_SS_pure
    
















    def _update_R_inv_add_S(self, raw_idx_to_add: int) -> bool:
        """
        Updates self.R_inv when adding a support vector.

        [CORRECTED] This method now returns a boolean status and includes an
        assertion to guard the critical "add support" path for R->S migrations.
        - True: If the support set was successfully changed (either by rank-1
                update or a full rebuild).
        - False: If the operation was aborted (e.g., point already in S_list).

        NOTE: The assertion guard relies on `self._last_case_code` and
        `self._last_c_idx` being set on the instance prior to this call.
        These attributes should be updated within the main processing loops
        (like in `incremental_update`) before `_update_params` is called.
        """
        #print(f"[DBG][_update_R_inv_add_S] Entering with c_idx={raw_idx_to_add}; "
        #      f"current S_list={self.S_list}")
        logging.debug(
            f"[TRACE] _update_R_inv_add_S ENTRY: attempting to add idx={raw_idx_to_add}, "
            f"current S_list={self.S_list}"
        )
        # Guard: For an R→S migration (Case 5), only the candidate point `c`
        # should be added to the support set. This assertion prevents bugs where
        # a different limiting point is mistakenly added.
        if hasattr(self, '_last_case_code') and self._last_case_code == 5:
            assert raw_idx_to_add == self._last_c_idx, (
                f"R→S migration (Case 5) must add the candidate point {self._last_c_idx}, "
                f"but was called with {raw_idx_to_add}."
            )

        s_list_before_op = self.S_list.copy()
        m_old = len(s_list_before_op)
        logging.info(f"_update_R_inv_add_S: ENTRY for raw_idx_to_add={raw_idx_to_add}. S_list size={m_old}")
        logging.debug(
            f"[DEBUG-RINV-ADD ▶ ENTRY] raw_idx_to_add={raw_idx_to_add}, "
            f"S_list(before)={s_list_before_op}"
        )

        if raw_idx_to_add in self.S_list:
            logging.debug(f"[TRACE] _update_R_inv_add_S: idx={raw_idx_to_add} already in S_list, aborting")
            logging.warning(f"  _update_R_inv_add_S: Index {raw_idx_to_add} is already in S_list. Aborting.")
            logging.debug(
                f"[DEBUG-RINV-ADD ▶ EXIT] raw_idx_added={raw_idx_to_add}, "
                f"S_list(after)={self.S_list}"
            )
            success_flag = False
            logging.debug(
                f"[TRACE] _update_R_inv_add_S EXIT: success={success_flag}, "
                f"new S_list={self.S_list}"
            )
            #print(f"[DBG][_update_R_inv_add_S] After update: S_list={self.S_list}, "
            #      f"R_inv diag={[self.R_inv[i,i] for i in self.S_list]}")
            return False  # No structural change

        logging.debug("[S-branch] about to add support vector c=%d to R_inv", raw_idx_to_add)

        if not self.S_list: # First SV, initialize from scratch
            self.S_list.append(raw_idx_to_add)
            self._compute_R_inv_initial()
            logging.debug(
                f"[DEBUG-RINV-ADD ▶ EXIT] raw_idx_added={raw_idx_to_add}, "
                f"S_list(after)={self.S_list}"
            )
            success_flag = True
            logging.debug("[S-branch] successfully updated R_inv; new |R|=%d", len(self.R))
            logging.debug(
                f"[TRACE] _update_R_inv_add_S EXIT: success={success_flag}, "
                f"new S_list={self.S_list}"
            )
            #print(f"[DBG][_update_R_inv_add_S] After update: S_list={self.S_list}, "
            #      f"R_inv diag={[self.R_inv[i,i] for i in self.S_list]}")
            return True # Structural change occurred

        try:
            beta0_c, betaS_c, gamma_c = self._calculate_sensitivities(raw_idx_to_add, s_list_before_op, self.R_inv)

            if abs(gamma_c) <= self.gamma_pivot_tol:
                logging.warning(f"  _update_R_inv_add_S: Unstable pivot |gamma_c| ({abs(gamma_c):.2e}). Falling back to full rebuild.")
                self.S_list.append(raw_idx_to_add)
                self.S_list.sort()
                self._compute_R_inv_initial()
                logging.debug(
                    f"[DEBUG-RINV-ADD ▶ EXIT] raw_idx_added={raw_idx_to_add}, "
                    f"S_list(after)={self.S_list}"
                )
                success_flag = True
                logging.debug("[S-branch] successfully updated R_inv; new |R|=%d", len(self.R))
                logging.debug(
                    f"[TRACE] _update_R_inv_add_S EXIT: success={success_flag}, "
                    f"new S_list={self.S_list}"
                )
                #print(f"[DBG][_update_R_inv_add_S] After update: S_list={self.S_list}, "
                #      f"R_inv diag={[self.R_inv[i,i] for i in self.S_list]}")
                return True # Structural change occurred (via rebuild)

            R_inv_old = self.R_inv
            v_update = np.concatenate(([beta0_c], betaS_c, [1.0]))

            m_new = m_old + 1
            R_inv_block = np.zeros((m_new + 1, m_new + 1), dtype=float)
            if R_inv_old is not None and R_inv_old.shape[0] > 0:
                R_inv_block[:m_old + 1, :m_old + 1] = R_inv_old

            R_inv_new = R_inv_block + (1.0 / gamma_c) * np.outer(v_update, v_update)

            self.R_inv = R_inv_new
            self.S_list.append(raw_idx_to_add)
            self.S_list.sort()
            self.Q = np.linalg.inv(self.R_inv)
            logging.info(f"  _update_R_inv_add_S: Direct rank-1 update of R_inv succeeded for adding idx={raw_idx_to_add}")
            logging.debug(
                f"[DEBUG-RINV-ADD ▶ EXIT] raw_idx_added={raw_idx_to_add}, "
                f"S_list(after)={self.S_list}"
            )
            success_flag = True
            logging.debug("[S-branch] successfully updated R_inv; new |R|=%d", len(self.R))
            logging.debug(
                f"[TRACE] _update_R_inv_add_S EXIT: success={success_flag}, "
                f"new S_list={self.S_list}"
            )
            #print(f"[DBG][_update_R_inv_add_S] After update: S_list={self.S_list}, "
            #      f"R_inv diag={[self.R_inv[i,i] for i in self.S_list]}")
            return True # Structural change occurred

        except (ValueError, np.linalg.LinAlgError) as e:
            logging.warning(f"  _update_R_inv_add_S: Direct update failed ({e}). Falling back to full rebuild.")
            self.S_list.append(raw_idx_to_add)
            self.S_list.sort()
            self._compute_R_inv_initial()
            logging.debug(
                f"[DEBUG-RINV-ADD ▶ EXIT] raw_idx_added={raw_idx_to_add}, "
                f"S_list(after)={self.S_list}"
            )
            success_flag = True
            logging.debug("[S-branch] successfully updated R_inv; new |R|=%d", len(self.R))
            logging.debug(
                f"[TRACE] _update_R_inv_add_S EXIT: success={success_flag}, "
                f"new S_list={self.S_list}"
            )
            #print(f"[DBG][_update_R_inv_add_S] After update: S_list={self.S_list}, "
            #      f"R_inv diag={[self.R_inv[i,i] for i in self.S_list]}")
            return True # Structural change occurred (via rebuild)


























    def _get_Q_aug_for_S_list(self, target_s_list: list[int]) -> np.ndarray:
        """
        Constructs the target Q_aug matrix (KKT matrix for bias and SVs)
        for a given S_list, strictly following the paper's formulation.

        [CORRECTED] This version now mirrors the corrected _compute_R_inv_initial
        method. It sets Q_aug[0,0] to 0 and only applies regularization and
        jitter to the K_SS submatrix, avoiding the complex and incorrect
        Schur complement calculation for the bias term.
        """
        m = len(target_s_list)
        Q_aug = np.zeros((m + 1, m + 1), dtype=float)

        if m > 0:
            K_SS_pure = self._extract_pure_K_SS(target_s_list)
            
            # Form H matrix with standard (2/C) regularization
            H = K_SS_pure + (2.0 / self.C) * np.eye(m)
            
            # Assemble the Q_aug matrix
            Q_aug[1:, 1:] = H
            ones_m = np.ones(m, dtype=float)
            Q_aug[0, 1:] = ones_m
            Q_aug[1:, 0] = ones_m

        # Set bias term to 0 as per Martin's paper's Q matrix definition (eq. 20)
        Q_aug[0, 0] = 0

        # Add jitter only to the K_SS part for numerical stability
        if m > 0:
            Q_aug[1:, 1:] += self.delta_jitter * np.eye(m)
        
        logging.debug(f"  _get_Q_aug_for_S_list: Q_aug condition number = {np.linalg.cond(Q_aug):.6e}")
        return Q_aug





















    def _update_R_inv_remove_S(self, raw_idx_to_remove: int) -> bool:
        """
        Removes a support vector and performs a direct rank-1 DOWNDATE of R_inv,
        with a special base case for removing the last support vector.

        [EXPLANATION] This function does NOT need changes. It is called from
        `decremental_unlearn` BEFORE the point at `raw_idx_to_remove` is
        physically deleted from the dataset and before any indices are shifted.
        Therefore, all operations within this function correctly use the original,
        un-shifted indices present in `self.S_list`. The index shifting is handled
        later by the calling function (`decremental_unlearn`).

        Args:
            raw_idx_to_remove (int): The index of the support vector to remove.
                                     This is the index relative to the *current* dataset.

        Returns:
            bool: True if the support set was structurally changed, False otherwise.
        """
        logging.debug(
            f"[_update_R_inv_remove_S] removing support #{raw_idx_to_remove}; R_inv.shape before={self.R_inv.shape if self.R_inv is not None else 'None'}"
        )
        s_list_before = self.S_list.copy()
        logging.info(f"_update_R_inv_remove_S: ENTRY for raw_idx_to_remove={raw_idx_to_remove}. S_list size={len(s_list_before)}")

        # Guard: Ensure the index to remove is actually in the support vector list.
        if raw_idx_to_remove not in s_list_before:
            logging.warning(f"  _update_R_inv_remove_S: Index {raw_idx_to_remove} not in S_list. Aborting.")
            return False # No structural change

        # --- Base-case: removing the last support vector ---
        if len(s_list_before) == 1:
            # If this is the last SV, we simply reset the basis.
            self.S_list.remove(raw_idx_to_remove)
            self.R_inv = None
            self.Q     = None
            logging.info("  _update_R_inv_remove_S: Handled last-SV removal via base-case reset.")
            return True # Structural change occurred (basis reset)

        # --- Standard case: Rank-1 downdate for |S| > 1 ---
        try:
            # Find the position of the vector to remove within the S_list.
            # This position corresponds to the row/column in the R_inv matrix.
            # This works because S_list and R_inv are synchronized at this point.
            k_in_S = s_list_before.index(raw_idx_to_remove)
            k_aug = k_in_S + 1 # +1 for the bias term at index 0 of R_inv

            R_inv_old = self.R_inv
            pivot = R_inv_old[k_aug, k_aug]

            if abs(pivot) < self.r_inv_pivot_tol:
                # Pivot is too small, downdate is unstable. Fallback to full rebuild.
                raise np.linalg.LinAlgError(f"Unstable pivot for downdate R_inv[{k_aug},{k_aug}] = {pivot:.3e}")

            # Perform the Sherman-Morrison rank-1 downdate
            R_inv_k_col = R_inv_old[:, [k_aug]]
            R_inv_downdated_full = R_inv_old - (R_inv_k_col @ R_inv_k_col.T) / pivot

            # Delete the corresponding row and column to get the new, smaller R_inv
            R_inv_new = np.delete(np.delete(R_inv_downdated_full, k_aug, axis=0), k_aug, axis=1)

            # Update the model state by removing the index and setting the new matrices
            self.S_list.remove(raw_idx_to_remove)
            self.R_inv = R_inv_new if R_inv_new.size > 0 else None
            self.Q = np.linalg.inv(self.R_inv) if self.R_inv is not None else None

            logging.info(f"  _update_R_inv_remove_S: Direct rank-1 downdate of R_inv succeeded.")
            return True # Structural change occurred

        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to a full rebuild if the downdate fails for any reason
            logging.warning(f"  _update_R_inv_remove_S: Downdate failed ({e}). Falling back to full rebuild.")
            if raw_idx_to_remove in self.S_list:
                self.S_list.remove(raw_idx_to_remove)
            # Rebuild R_inv from scratch using the now-smaller S_list
            self._compute_R_inv_initial()
            return True # Structural change occurred (via rebuild)
























    def _rebuild_basis_with_single_point(self, c_idx: int):
        """
        Helper to reset the KKT basis to contain only a single support vector.
        This is the 'base case' from Martin et al. (2003), eq. 29.
        """
        logging.info(f"  _rebuild_basis_with_single_point: Resetting basis to S_list=[{c_idx}]")
        self.S_list = [c_idx]
        self.R.discard(c_idx)
        self.E.discard(c_idx)
        self.E_star.discard(c_idx)
        self._compute_R_inv_initial()

        # ─── Insert EXIT log ─────────────────────────────────────────────────────────
        logging.debug(
            "_rebuild_basis_with_single_point EXIT: "
            "S_list=%s, |R|=%d, |E|=%d, |E*|=%d, "
            "R_factor_inv.shape=%s, Q.shape=%s",
            self.S_list,
            len(self.R),
            len(self.E),
            len(self.E_star),
            getattr(self.R_inv, "shape", None),
            getattr(self.Q, "shape", None)
        )

































    def _rebuild_basis_with_single_point(self, c_idx: int):
        """
        Helper to reset the KKT basis to contain only a single support vector.
        This is the 'base case' from Martin et al. (2003), eq. 29.
        [CORRECTED] Logging statements updated to refer to self.R_inv.
        """
        logging.info(f"  _rebuild_basis_with_single_point: Resetting basis to S_list=[{c_idx}]")
        self.S_list = [c_idx]
        self.R.discard(c_idx)
        self.E.discard(c_idx)
        self.E_star.discard(c_idx)
        self._compute_R_inv_initial()
    
        # ─── Insert EXIT log (Corrected) ──────────────────────────────────────────
        logging.debug(
            "_rebuild_basis_with_single_point EXIT: "
            "S_list=%s, |R|=%d, |E|=%d, |E*|=%d, "
            "R_inv.shape=%s, Q.shape=%s",
            self.S_list,
            len(self.R),
            len(self.E),
            len(self.E_star),
            self.R_inv.shape if hasattr(self, 'R_inv') and self.R_inv is not None else "None",
            self.Q.shape if hasattr(self, 'Q') and self.Q is not None else "None"
        )


















    def _calculate_sensitivities(self,
                         c_idx: int,
                         s_list_base: list[int],
                         R_inv_base: np.ndarray | None
                         ) -> tuple[float, np.ndarray, float]:
        """
        Calculates sensitivities for a candidate point c_idx using the pre-computed
        inverse KKT matrix (R_inv_base).

        [CORRECTED] This method no longer uses Cholesky factors. It now directly
        implements the matrix-vector products described in the paper using R_inv.
        - beta_vector = -R_inv * q_c
        - gamma_c = (K_cc + 2/C) + q_c^T * beta_vector
        """
        logging.debug(f"[_calculate_sensitivities] Called with c_idx={c_idx}, S_basis={s_list_base}")
        n_s_base = len(s_list_base)

        x_c_arr = np.asarray(self.X_internal[c_idx]).reshape(1, -1)
        k_cc_val = self._kernel(x_c_arr, x_c_arr)[0, 0]

        # CASE A: S_base is empty
        if n_s_base == 0:
            beta0_c_val = -1.0
            betas_s_c_val = np.empty((0,), dtype=float)
            logging.debug(f"[SENS_DEBUG] c_idx={c_idx}: beta0={beta0_c_val:.6f}, betaS_sum={np.sum(np.abs(betas_s_c_val)):.6f}")
            # gamma_c = (K_cc + 2/C) + beta_0
            gamma_c_sens_val = (k_cc_val + 1.0 / self.C) + beta0_c_val
            logging.debug(f"[SENS_DEBUG] c_idx={c_idx}: gamma_c={gamma_c_sens_val:.6f}, k_cc={k_cc_val:.6f}")
            logging.debug(
                f"_calculate_sensitivities ▶ c_idx={c_idx}, basis={s_list_base}, "
                f"beta0={beta0_c_val:.6e}, betaS={betas_s_c_val}"
            )
            logger.debug(f"[_calc_sens] c_idx={c_idx} → β₀={beta0_c_val:.6e}, β_S={betas_s_c_val!r}, γ={gamma_c_sens_val:.6e}")
            logging.debug(
                f"[SENSITIVITY] c_idx={c_idx}  beta0_c={beta0_c_val:.6f}  betaS_c={betas_s_c_val}  γ_c={gamma_c_sens_val:.6f}"
            )
            logging.debug(f"[clamp] _calculate_sensitivities → beta0_c={beta0_c_val:.6f}")
            logging.debug(
                f"DEBUG(_calculate_sensitivities): for c_idx={c_idx} → "
                f"beta0_c={beta0_c_val:.6f}, betaS_c_len={betas_s_c_val.size}, γ_c={gamma_c_sens_val:.6f}"
            )
            logging.debug(
                f"[SENS LOG] c_idx={c_idx} · "
                f"β₀={beta0_c_val:.6f} · "
                f"γ={gamma_c_sens_val:.6f} · "
                f"Σβ_S={betas_s_c_val.sum():.6f}"
            )
            return beta0_c_val, betas_s_c_val, gamma_c_sens_val

        # CASE B: S_base is non-empty
        try:
            if R_inv_base is None or R_inv_base.shape[0] != n_s_base + 1:
                raise ValueError(f"R_inv_base is None or has inconsistent shape. Expected ({n_s_base+1},{n_s_base+1})")

            X_s_base_arr = np.vstack([np.asarray(self.X_internal[s_idx]).reshape(1, -1) for s_idx in s_list_base])
            k_c_S_base_vec = self._kernel(x_c_arr, X_s_base_arr).flatten()

            # Form the augmented kernel vector q_c = [1, K_cS]^T
            q_c_aug = np.concatenate(([1.0], k_c_S_base_vec))
            logging.debug(
                f"[SENS_DEBUG] c_idx={c_idx} · q_c (first 5)={q_c_aug[:5]!r}"
            )

            # 1. [CORRECTION] Calculate beta sensitivities directly using R_inv
            # beta_full = -R_inv * q_c
            beta_full_vector = -R_inv_base @ q_c_aug
            logging.debug(
                f"[SENS_DEBUG] c_idx={c_idx} · beta_vector (first 5)={beta_full_vector[:5]!r}"
            )
            beta0_c_val = float(beta_full_vector[0])
            betas_s_c_val = beta_full_vector[1:]
            logging.debug(f"[SENS_DEBUG] c_idx={c_idx}: beta0={beta0_c_val:.6f}, betaS_sum={np.sum(np.abs(betas_s_c_val)):.6f}")

            # 2. [CORRECTION] Calculate gamma_c sensitivity directly
            # gamma_c = (K_cc + 2/C) + q_c^T * beta_full
            gamma_c_sens_val = (k_cc_val + (1.0 / self.C)) + np.dot(q_c_aug, beta_full_vector)
            logging.debug(
                f"[SENS_DEBUG] c_idx={c_idx} · beta0_c={beta0_c_val:.6f} · "
                f"betas_s_c (first 5)={betas_s_c_val[:5]!r} · gamma_c={gamma_c_sens_val:.6f}"
            )
            logging.debug(f"[SENS_DEBUG] c_idx={c_idx}: gamma_c={gamma_c_sens_val:.6f}, k_cc={k_cc_val:.6f}")

            logging.debug(
                f"_calculate_sensitivities ▶ c_idx={c_idx}, basis={s_list_base}, "
                f"beta0={beta0_c_val:.6e}, betaS={betas_s_c_val}"
            )

            logging.info(
                f"_calculate_sensitivities: EXIT c_idx={c_idx}. "
                f"beta0={beta0_c_val:.4e}, "
                f"gamma_c={gamma_c_sens_val:.4e}"
            )
            logger.debug(f"[_calc_sens] c_idx={c_idx} → β₀={beta0_c_val:.6e}, β_S={betas_s_c_val!r}, γ={gamma_c_sens_val:.6e}")
            logging.debug(
                f"[SENSITIVITY] c_idx={c_idx}  beta0_c={beta0_c_val:.6f}  betaS_c={betas_s_c_val}  γ_c={gamma_c_sens_val:.6f}"
            )
            logging.debug(f"[clamp] _calculate_sensitivities → beta0_c={beta0_c_val:.6f}")
            logging.debug(
                f"DEBUG(_calculate_sensitivities): for c_idx={c_idx} → "
                f"beta0_c={beta0_c_val:.6f}, betaS_c_len={betas_s_c_val.size}, γ_c={gamma_c_sens_val:.6f}"
            )
            logging.debug(
                f"[SENS LOG] c_idx={c_idx} · "
                f"β₀={beta0_c_val:.6f} · "
                f"γ={gamma_c_sens_val:.6f} · "
                f"Σβ_S={betas_s_c_val.sum():.6f}"
            )
            return beta0_c_val, betas_s_c_val, gamma_c_sens_val

        except (np.linalg.LinAlgError, ValueError) as e:
            logging.error(f"  _calculate_sensitivities: c_idx={c_idx}: Method failed: {e}. Returning NaNs.", exc_info=True)
            beta0, betaS, gamma = np.nan, np.full(n_s_base, np.nan), np.nan
            logging.debug(
                f"DEBUG(_calculate_sensitivities): for c_idx={c_idx} → "
                f"beta0_c={beta0:.6f}, betaS_c_len={betaS.size}, γ_c={gamma:.6f}"
            )
            logging.debug(
                f"[SENS LOG] c_idx={c_idx} · "
                f"β₀={beta0:.6f} · "
                f"γ={gamma:.6f} · "
                f"Σβ_S={betaS.sum():.6f}"
            )
            return np.nan, np.full(n_s_base, np.nan), np.nan





























    def _get_margin(self, idx):
        """Calculates the margin h(x_idx) = f(x_idx) - y_idx."""
        if idx >= len(self.X_internal) or idx < 0:
            return np.nan
    
        #print(f"[DBG][_get_margin] idx={idx}; x={self.X_internal[idx]}, b={self.bias}")
    
        # Calculate prediction f(x_idx) = sum_j (alpha_j - alpha*_j) K(x_j, x_idx) + bias
        f_x = self.bias
        logging.debug(f"[MARGIN_DEBUG] _get_margin START for idx={idx}, initial bias={self.bias:.6f}")
        logging.debug(
            f"[MARGIN_DEBUG] idx={idx} · starting f_x={f_x:.6f} · y_true={self.y_internal[idx]:.6f}"
        )
        num_internal = len(self.X_internal)
    
        # Ensure alphas and X_internal are consistent
        if not (num_internal == len(self.alphas) == len(self.alphas_star)):
             # Attempt recovery or return NaN
             min_len = min(num_internal, len(self.alphas), len(self.alphas_star))
             if min_len == 0: return self.bias - self.y_internal[idx] # Best guess if possible
             thetas = (np.array(self.alphas[:min_len]) - np.array(self.alphas_star[:min_len]))
             X_all_list = [np.asarray(x) for x in self.X_internal[:min_len]]
             num_internal = min_len # Adjust for calculation
             warn("Inconsistent lengths detected in _get_margin, attempting recovery.")
        else:
            thetas = np.array(self.alphas) - np.array(self.alphas_star)
            X_all_list = [np.asarray(x) for x in self.X_internal]
    
    
        if num_internal > 0:
            try:
                # Ensure data is in consistent numpy array format
                if self._feature_dim == 1:
                     X_all = np.array([x.reshape(1) for x in X_all_list]).reshape(num_internal, 1)
                     x_i = np.asarray(self.X_internal[idx]).reshape(1, 1)
                else:
                     # Assuming _feature_dim is set correctly
                     X_all = np.array([x.reshape(self._feature_dim) for x in X_all_list])
                     x_i = np.asarray(self.X_internal[idx]).reshape(1, self._feature_dim)
    
                if X_all.ndim != 2 or x_i.ndim != 2:
                     raise ValueError(f"Incorrect array dimensions: X_all {X_all.ndim}D, x_i {x_i.ndim}D")
                if X_all.shape[1] != x_i.shape[1]:
                     raise ValueError(f"Feature dimension mismatch: X_all {X_all.shape[1]}, x_i {x_i.shape[1]}")
    
                # Compute kernel vector K(x_i, X_all)
                kernels = self._kernel(x_i, X_all).flatten() # Shape (num_internal,)
    
                if len(thetas) != len(kernels):
                    raise ValueError(f"Theta length {len(thetas)} != Kernel length {len(kernels)}")
                
                # NOTE: The requested per-element logging (`logging.debug(f"[MARGIN_DEBUG] idx={idx}, j={j}: ...")`)
                # cannot be added here without changing the function's vectorized logic to a loop.
                # The entire kernel contribution is calculated in the single line below.
                
                # Add kernel sum term
                f_x += np.dot(thetas, kernels)
    
            except Exception as e:
                return np.nan # Indicate error
    
        target_y = self.y_internal[idx]
        margin = f_x - target_y
        logging.debug(
            f"[MARGIN_DEBUG] idx={idx} · computed margin h={margin:.6f}"
        )
        logging.debug(f"[MARGIN_DEBUG] idx={idx}: f(x)={f_x:.6f}, y={self.y_internal[idx]:.6f}, margin h={margin:.6f}")
        logging.debug(f"_get_margin ▶ idx={idx}, f(x)={f_x:.6e}, y={target_y:.6e}, h={margin:.6e}")
        
        #print(f"[DBG][_get_margin] -> h={margin:.6f}")
        
        return margin
    
    
    





























    def _calculate_max_step(
    self,
    c_idx: int,
    q_direction: float,
) -> tuple[int, list[int], float, list[str], bool]:
        """
        Finds the minimum valid step |Δθ_c| by considering all possible KKT
        migration events for all points in the dataset.

        [CORRECTED] This version correctly handles zero-step migrations for points
        landing exactly on the ε-tube boundary, ensuring they are promoted to
        Support Vectors.
        """
        # 1. Get current margin, θ, and constants
        h_c = self._get_margin(c_idx)
        if np.isnan(h_c):
            logging.warning(f"_calculate_max_step: Margin h_c for c_idx={c_idx} is NaN.")
            return -1, [c_idx], 0.0, ["Margin h_c is NaN"], False

        theta_c = self._get_current_theta(c_idx)
        ε = self.epsilon
        C = self.C
        MACHINE_EPS = self.MACHINE_EPS
        kkt_tol = self.kkt_tol

        logging.debug(f"_DIAG ▶ _calculate_max_step ENTRY c_idx={c_idx} | h_c={h_c:.6f} | θ_c={theta_c:.6f} | γ_direction={q_direction:.6f}")

        logging.info(f"_calculate_max_step: ENTRY c_idx={c_idx}, q_direction={q_direction:.2f}, h_c={h_c:.4f}, θ_c={theta_c:.4f}")

        # 2. Compute sensitivities
        S_basis = self.S_list.copy()
        try:
            beta0_c, betaS_c, γ_c = self._calculate_sensitivities(c_idx, S_basis, self.R_inv)
            logging.debug(
                f"[calc_max ▶ ENTRY] c_idx={c_idx} · "
                f"beta0_c={beta0_c:.6f} · "
                f"betaS_c (first 5)={betaS_c[:5]!r}"
            )
            logger.debug(f"[_max_step] ENTER c_idx={c_idx}: β₀={beta0_c:.6e}, β_S={betaS_c!r}")
            if np.isnan(γ_c):
                return -1, [c_idx], 0.0, ["NaN γ sensitivity"], False
        except Exception as e:
            logging.error(f"Exception in sensitivities for c_idx={c_idx}: {e}", exc_info=True)
            return -1, [c_idx], 0.0, [f"Sensitivity error: {e}"], False

        logger.debug(f"[calc_step] Point {c_idx}: γ_c={γ_c:.6f}, θ_c={theta_c:.6f}, h_c={h_c:.6f}")

        if abs(γ_c) < self.gamma_pivot_tol:
            logging.info(f"  γ_c ({γ_c:.2e}) ≈ 0 → linear dependency.")
            return -2, [c_idx], 0.0, ["Linear dependency"], False

        candidate_steps = []

        # Initialize step variables for logging
        delta1, delta2, delta3 = np.nan, np.nan, np.nan

        # 3. Candidate steps for the point c itself
        if c_idx in self.S_list:
            # ── Far-out support vector: clamp to capacity (Case 2) only
            if abs(h_c) > ε + kkt_tol:
                tgt_th = C * np.sign(h_c)
                delta_cap = tgt_th - theta_c
                delta2 = delta_cap
                candidate_steps.append({
                    'case': 2,
                    'idx': c_idx,
                    'L_val': delta_cap,
                    'reason': f"Case2_c: clamp far-out SV to θ={tgt_th:.6f}",
                    'force_remove': False
                })
            else:
                # ── Boundary SV: margin-adjust to ε-tube (Case 1)
                tgt_h = np.sign(h_c) * ε
                delta_m = (tgt_h - h_c) / γ_c
                delta1 = delta_m
                candidate_steps.append({
                    'case': 1,
                    'idx': c_idx,
                    'L_val': delta_m,
                    'reason': f"Case1_c: margin-adjust to h_c={tgt_h:.6f}",
                    'force_remove': False
                })
                # …and still allow a clamp if later needed (Case 2)
                tgt_th = C * np.sign(h_c)
                delta_cap = tgt_th - theta_c
                delta2 = delta_cap
                candidate_steps.append({
                    'case': 2,
                    'idx': c_idx,
                    'L_val': delta_cap,
                    'reason': f"Case2_c: clamp SV to θ={tgt_th:.6f}",
                    'force_remove': False
                })
            # SV→R transition (Case 3)
            delta3 = -theta_c
            candidate_steps.append({
                'case': 3,
                'idx': c_idx,
                'L_val': delta3,
                'reason': "Case3_c(in S)→R (θ_c→0)",
                'force_remove': True
            })
        else:
            # ── Non-SV: always evaluate capacity clamp first (Case 2)
            tgt_th = C * np.sign(h_c)
            delta_cap = tgt_th - theta_c
            delta2 = delta_cap
            candidate_steps.append({
                'case': 2,
                'idx': c_idx,
                'L_val': delta_cap,
                'reason': f"Case2_c: clamp to θ={tgt_th:.6f}",
                'force_remove': False
            })
            # ── Then margin-move to ε-tube (Case 1), but **only** if we're inside or just at the boundary of the tube
            if abs(h_c) <= ε + kkt_tol:
                if abs(h_c) > kkt_tol:
                    tgt_h = np.sign(h_c) * ε
                else:
                    tgt_h = -q_direction * ε
                delta1 = (tgt_h - h_c) / γ_c
                candidate_steps.append({
                    'case': 1,
                    'idx': c_idx,
                    'L_val': delta1,
                    'reason': f"Case1_c: margin to h_c={tgt_h:.6f}",
                    'force_remove': False
                })

        # 4. Candidate steps for other SVs j in S_list
        if betaS_c.size == len(S_basis):
            for i, j in enumerate(S_basis):
                if j == c_idx:
                    continue
                θ_j = self._get_current_theta(j)
                β_j = betaS_c[i]
                if abs(β_j) > MACHINE_EPS:
                    # Case 3: S -> R for other SV
                    candidate_steps.append({
                        'case': 3,
                        'idx': j,
                        'L_val': -θ_j / β_j,
                        'reason': f"Case3_S#{j}→R",
                        'force_remove': True
                    })
                    # Case 2: S -> E/E* for other SV
                    target_θ_j = C * np.sign(β_j * q_direction)
                    candidate_steps.append({
                        'case': 2,
                        'idx': j,
                        'L_val': (target_θ_j - θ_j) / β_j,
                        'reason': f"Case2_S#{j}→E/E*",
                        'force_remove': True
                    })

        # 5. Candidate steps for non‐S points (R, E, E*)
        non_s = (self.E | self.E_star | self.R) - set(S_basis)
        for i in non_s:
            if i == c_idx:
                continue
            h_i = self._get_margin(i)
            if np.isnan(h_i):
                continue
            γ_i = self._get_gamma_sensitivity_for_non_s(i, c_idx, beta0_c, betaS_c, S_basis)
            if np.isnan(γ_i) or abs(γ_i) < MACHINE_EPS:
                continue
            case_code = 4 if i in (self.E | self.E_star) else 5
            # Case 4/5: R/E/E* -> S+
            candidate_steps.append({
                'case': case_code,
                'idx': i,
                'L_val': ( ε - h_i) / γ_i,
                'reason': f"Case{case_code}_i#{i}→S+",
                'force_remove': False
            })
            # Case 4/5: R/E/E* -> S-
            candidate_steps.append({
                'case': case_code,
                'idx': i,
                'L_val': (-ε - h_i) / γ_i,
                'reason': f"Case{case_code}_i#{i}→S-",
                'force_remove': False
            })

        for s in candidate_steps:
            logging.debug(f"_DIAG ▶ candidate step: case={s['case']} | L_val={s['L_val']:.6e} | limiter_idx={s['idx']}")

        logging.debug(
            f"[calc_max ▶ CANDIDATES] c_idx={c_idx} · "
            f"Δθ_case1={delta1:.6f}, "
            f"Δθ_case2={delta2:.6f}, "
            f"Δθ_case3={delta3:.6f}"
        )

        # 6. Select smallest valid step (now including zero-steps)
        valid = [
            c for c in candidate_steps
            if np.isfinite(c['L_val']) and (q_direction * c['L_val'] > -kkt_tol)
        ]
        if not valid:
            logging.error(f"_calculate_max_step c_idx={c_idx}: no valid Δθ found.")
            return -1, [c_idx], 0.0, ["No valid step"], False

        best = min(valid, key=lambda c: abs(c['L_val']))
        logging.debug(f"_DIAG ▶ best step: case={best['case']} | L_val={best['L_val']:.6e} | limiter={best['idx']}")

        assert not (abs(h_c) > ε and best['case'] == 1), \
            f"Out-of-tube point c_idx={c_idx} took margin path (case 1)!"

        # --- [CORRECTION] Prioritize Case 1 margin-hit if the step is zero ---
        # This handles the g_c=0 (or g*_c=0) event which forces a point into S.
        if best['case'] == 1 and abs(best['L_val']) <= kkt_tol:
            logging.info(
                f"_calculate_max_step ▶ c_idx={c_idx}, FORCED Case 1 (margin hit), "
                f"Δθ≈0, limiter={best['idx']}"
            )
            # A zero-step Case 1 is a successful migration to S.
            delta_theta_margin_hit = best['L_val']
            limiting_idxs_margin = [best['idx']]
            logger.debug(f"[_max_step] RETURN c_idx={c_idx}: Δθ_margin_hit={delta_theta_margin_hit:.6e}, "
                         f"limiting_idxs={limiting_idxs_margin}")

            chosen_case = 1
            chosen_DeltaTheta = delta_theta_margin_hit
            logging.debug(
                f"[calc_max ▶ CHOSEN] c_idx={c_idx} · "
                f"case={chosen_case} · "
                f"Δθ={chosen_DeltaTheta:.6f}"
            )

            force_remove = False # This case does not force a removal from S
            logging.debug(f"_DIAG ▶ RETURN _calculate_max_step c_idx={c_idx} | Δθ_c={best['L_val']:.6e} | force_remove={force_remove}")
            return 1, limiting_idxs_margin, delta_theta_margin_hit, [best['reason']], force_remove

        # --- Standard selection for non-zero steps ---
        tied = [c for c in valid if abs(abs(c['L_val']) - abs(best['L_val'])) < kkt_tol]

        limiting = sorted({int(c['idx']) for c in tied})
        case_code = best['case']
        Δθ_c = best['L_val']
        reasons = sorted({c['reason'] for c in tied})
        force_remove = any(c['force_remove'] for c in tied)

        chosen_case = case_code
        chosen_DeltaTheta = Δθ_c
        logging.debug(
            f"[calc_max ▶ CHOSEN] c_idx={c_idx} · "
            f"case={chosen_case} · "
            f"Δθ={chosen_DeltaTheta:.6f}"
        )

        logging.info(
            f"_calculate_max_step ▶ c_idx={c_idx}, case={case_code}, Δθ={Δθ_c:.6e}, "
            f"limiters={limiting}, force_remove={force_remove}"
        )
        logging.debug(
            f"[STEP DEBUG] c_idx={c_idx} → selected case={case_code}, "
            f"Δθ_c={Δθ_c:.6f}, limiter_idxs={limiting}, reason={reasons}"
        )
        logging.debug(f"_DIAG ▶ RETURN _calculate_max_step c_idx={c_idx} | Δθ_c={best['L_val']:.6e} | force_remove={force_remove}")
        return case_code, limiting, Δθ_c, reasons, force_remove





































    def _get_gamma_sensitivity_for_non_s(self, i_idx, c_idx, beta0_c, betaS_c, s_list_basis):
        """Helper to compute gamma_i = d(h_i)/d(theta_c) for a non-S point."""
        x_i_arr = np.asarray(self.X_internal[i_idx]).reshape(1, -1)
        x_c_arr = np.asarray(self.X_internal[c_idx]).reshape(1, -1)
        
        kernel_ic = self._kernel(x_i_arr, x_c_arr)[0, 0]
        logging.debug(
            f"[GAMMA_NON_S_DEBUG] i_idx={i_idx}, c_idx={c_idx} · "
            f"kernel_ic={kernel_ic:.6f} · beta0_c={beta0_c:.6f} · "
            f"gamma_i_base={kernel_ic + beta0_c:.6f}"
        )
        gamma_i = kernel_ic + beta0_c
        
        if s_list_basis:
            X_s_arr = np.vstack([np.asarray(self.X_internal[s]).reshape(1, -1) for s in s_list_basis])
            k_iS_vec = self._kernel(x_i_arr, X_s_arr).flatten()
            
            s_contrib = np.dot(k_iS_vec, betaS_c)
            gamma_i += s_contrib
            logging.debug(
                f"[GAMMA_NON_S_DEBUG] i_idx={i_idx}, added S-contrib={s_contrib:.6f} · "
                f"gamma_i_final={gamma_i:.6f}"
            )
            
        return gamma_i
































    def _update_alphas_from_delta(self, idx: int, delta_theta: float):
        """
        Helper to update alpha and alpha_star for index `idx` given `delta_theta`.
        Ensures 0 <= alpha, alpha* <= C.
        """
        if not (0 <= idx < len(self.alphas)):
            logging.error(f"_update_alphas_from_delta: Index {idx} out of bounds for alphas (len {len(self.alphas)}).")
            return

        old_theta = self._get_current_theta(idx)
        logging.debug(f"[clamp] _update_alphas_from_delta START for c_idx={idx}: old θ_c={old_theta:.6f}")

        alpha_i_old = self.alphas[idx]
        alpha_star_i_old = self.alphas_star[idx]
        theta_i_old = alpha_i_old - alpha_star_i_old
        new_theta_i = theta_i_old + delta_theta

        if new_theta_i >= 0:
            new_alpha_i = new_theta_i
            new_alpha_star_i = 0.0
        else:
            new_alpha_i = 0.0
            new_alpha_star_i = -new_theta_i

        logging.debug(
            f"[ALPHA UPDATE] idx={idx} · "
            f"before α={alpha_i_old:.6f}, α*={alpha_star_i_old:.6f} · "
            f"after  α={new_alpha_i:.6f}, α*={new_alpha_star_i:.6f}"
        )

        self.alphas[idx] = min(self.C, max(0.0, new_alpha_i))
        self.alphas_star[idx] = min(self.C, max(0.0, new_alpha_star_i))

        new_theta = self._get_current_theta(idx)
        logging.debug(f"[clamp] _update_alphas_from_delta DONE for c_idx={idx}: new θ_c={new_theta:.6f}")



























    def _get_current_theta(self, idx: int) -> float:
        """Helper to get the current theta (alpha - alpha_star) for a point."""
        if not (0 <= idx < len(self.alphas) and 0 <= idx < len(self.alphas_star)):
            logging.error(f"_get_current_theta: Index {idx} out of bounds for alpha arrays.")
            return np.nan
        return self.alphas[idx] - self.alphas_star[idx]


























    def _update_params(
self,
c_idx: int,
limiting_idxs: list[int],
chosen_DeltaTheta_c: float,
case_code: int,
beta0_c: float,
betaS_c: np.ndarray,
s_list_basis_for_betaS_c: list[int],
force_remove: bool
) -> tuple[float, bool]:
        """
        Applies the calculated step Δθ_c, propagates the change through the model,
        and handles the single structural migration that limited the step.

        [CORRECTED] This version fixes a critical logic flaw in the Case 1 handler.
        It now correctly adds a point to the Support Vector set (S) ONLY if its
        new margin |h| is approximately equal to epsilon. If |h| > epsilon, the
        point is correctly classified as an Error Support Vector (E or E*) without
        mutating the S set.
        """
        logging.debug(f"[UPD_PARAMS ENTRY] c_idx={c_idx} · case={case_code} · Δθ_c={chosen_DeltaTheta_c:.6f}")
        logger.debug(f"[upd_params] ENTER c_idx={c_idx}, case_code={case_code}, Δθ_c={chosen_DeltaTheta_c:.6e}")
        old_bias = self.bias
        logging.debug(
            f"[UPDATE_PARAMS_DEBUG] beta0_c={beta0_c:.6f} · "
            f"betaS_c (first 5)={betaS_c[:5]!r}"
        )
        s_list_changed = False
        kkt_tol = self.kkt_tol

        old_theta_c_candidate = self._get_current_theta(c_idx)

        # ADDED LOGGING (AT THE TOP)
        if limiting_idxs:
            limiter_idx_log = limiting_idxs[0]
            theta_before = self._get_current_theta(limiter_idx_log)
            h_before = self._get_margin(limiter_idx_log)
            #print(f"[Update-IN ] idx={limiter_idx_log}, case={case_code}, "
            #      f"θ_before={theta_before:.4f}, h_before={h_before:.4f}")


        # --- Step 1: Adiabatic Update of Coefficients ---
        if abs(chosen_DeltaTheta_c) > self.MACHINE_EPS:
            if case_code == 2:
                # ——— Capacity-clamp branch with detailed logging ———
                logging.debug(
                    f"[CLAMP ENTRY] c_idx={c_idx} · beta0_c={beta0_c:.6f} · "
                    f"supports_count={len(s_list_basis_for_betaS_c)} · "
                    f"betaS_c_len={betaS_c.size}"
                )

                if betaS_c.size != len(s_list_basis_for_betaS_c):
                    logging.warning(
                        f"[CLAMP VERIFY] FASTPATH: betaS_c.size ({betaS_c.size}) != "
                        f"supports_count ({len(s_list_basis_for_betaS_c)}) — "
                        "propagation of existing SVs will be skipped!"
                    )

                # 1) log + update bias
                logging.debug(f"[CLAMP] Before bias update: bias={self.bias:.6f}")
                self.bias -= beta0_c * chosen_DeltaTheta_c
                logging.debug(f"[CLAMP] After bias update: bias={self.bias:.6f}")

                # 2) propagate Δθ to every existing support vector
                for i, s_idx in enumerate(s_list_basis_for_betaS_c):
                    if s_idx == c_idx:
                        continue
                    delta_s = betaS_c[i] * chosen_DeltaTheta_c
                    logging.debug(
                        f"[CLAMP] Propagate to SV {s_idx}: "
                        f"β_s^c={betaS_c[i]:.6f} · Δθ_c={chosen_DeltaTheta_c:.6f} → Δθ_s={delta_s:.6f}"
                    )
                    self._update_alphas_from_delta(s_idx, delta_s)
                    logging.debug(
                        f"[CLAMP]   SV {s_idx} new θ={self._get_current_theta(s_idx):.6f}, "
                        f"h={self._get_margin(s_idx):.6f}"
                    )
                
                for s_idx in s_list_basis_for_betaS_c:
                    h_s = self._get_margin(s_idx)
                    logging.debug(
                        f"[CLAMP VERIFY] Post-update margin for SV {s_idx}: "
                        f"h={h_s:.6f} (should be ±{self.epsilon})"
                    )

                # 3) update the new point’s coefficient
                logging.debug(
                    f"[CLAMP] Updating new point {c_idx}: Δθ_c={chosen_DeltaTheta_c:.6f}"
                )
                self._update_alphas_from_delta(c_idx, chosen_DeltaTheta_c)
                logging.debug(
                    f"[CLAMP]   New point {c_idx} new θ={self._get_current_theta(c_idx):.6f}, "
                    f"h={self._get_margin(c_idx):.6f}"
                )
                
                new_h_c = self._get_margin(c_idx)
                logging.debug(
                    f"[CLAMP VERIFY] New point c_idx={c_idx} post-update: "
                    f"h_c={new_h_c:.6f}, θ_c={self._get_current_theta(c_idx):.6f}"
                )

                # ——— NEW: force-classify as Error SV ———
                new_h_c = self._get_margin(c_idx)
                if new_h_c >  self.epsilon:
                    self.E_star.add(c_idx)
                    self.E.discard(c_idx)
                elif new_h_c < -self.epsilon:
                    self.E.add(c_idx)
                    self.E_star.discard(c_idx)
                # in either case, it’s definitely not in R
                self.R.discard(c_idx)

                logging.debug(f"[CLAMP EXIT] c_idx={c_idx} · classified as {'E*' if new_h_c>self.epsilon else 'E'} · leaving _update_params")
                return chosen_DeltaTheta_c, True
            else:
                # standard propagation to all existing support vectors
                self.bias += beta0_c * chosen_DeltaTheta_c
                if betaS_c.size == len(s_list_basis_for_betaS_c):
                    for i, s_idx_in_basis in enumerate(s_list_basis_for_betaS_c):
                        if s_idx_in_basis == c_idx:
                            continue
                        delta_theta_for_s = betaS_c[i] * chosen_DeltaTheta_c
                        self._update_alphas_from_delta(s_idx_in_basis, delta_theta_for_s)
                self._update_alphas_from_delta(c_idx, chosen_DeltaTheta_c)

            logging.debug(
                f"[upd_params ▶ ALPHAS] c_idx={c_idx} · "
                f"alpha={self.alphas[c_idx]:.6f} · "
                f"alpha*={self.alphas_star[c_idx]:.6f}"
            )

        effective_delta_theta_c_candidate = self._get_current_theta(c_idx) - old_theta_c_candidate
        updated_theta = self._get_current_theta(c_idx)
        logger.debug(f"[upd_params] θ[{c_idx}] clamped to {updated_theta:.6f}")
        logging.debug(
            f"[UPDATE_PARAMS_DEBUG] Post-adiabatic → "
            f"bias={self.bias:.6f} · "
            f"θ_c_old={old_theta_c_candidate:.6f} · "
            f"θ_c_new={updated_theta:.6f} · "
            f"effective_Δθ={effective_delta_theta_c_candidate:.6f}"
        )

        # --- Step 2: Handle the Single Structural Migration ---
        if not limiting_idxs:
            logger.debug(f"[upd_params] EXIT  c_idx={c_idx}: θ_c={self._get_current_theta(c_idx):.6f}, S_list={self.S_list}")
            return effective_delta_theta_c_candidate, False

        limiter_idx = limiting_idxs[0]

        new_h = self._get_margin(limiter_idx)
        #print(f"[DBG][_update_params] c_idx={c_idx}, case={case_code}, limiter_idx={limiter_idx}, " f"raw h={new_h:.6f}, | |h|-ε |= {abs(abs(new_h)-self.epsilon):.6f}, kkt_tol={self.kkt_tol}")
        new_h_for_log = self._get_margin(c_idx)
        logging.debug(
            f"[upd_params ▶ MARGIN] c_idx={c_idx} · "
            f"new_h={new_h_for_log:.6f} · ε={self.epsilon:.6f}"
        )
        new_theta = self._get_current_theta(limiter_idx)
        logging.debug(
            f"[UPDATE_PARAMS_DEBUG] Case={case_code} · limiter_idx={limiter_idx} · "
            f"new_margin={new_h:.6f}"
        )
        
        logging.debug(
            f"[CLASSIFY CHECK] c_idx={c_idx} · new_h={new_h:.6f} · "
            f"ε={self.epsilon:.6f} · tol={kkt_tol:.6f}"
        )
        classification = (
            "E*" if new_h > self.epsilon
            else "E" if new_h < -self.epsilon
            else "R"
        )
        logging.debug(
            f"[upd_params ▶ CLASSIFY] c_idx={limiter_idx} · "
            f"classification={classification}"
        )

        # --- [CORRECTED] Case 1: New point `c` is moved towards the margin ---
        if case_code == 1 and limiter_idx == c_idx:
            # Check if the point landed EXACTLY on the tube boundary
            if abs(abs(new_h) - self.epsilon) < kkt_tol:
                # 1) log the exact promotion
                logging.info(
                    f"  _update_params: Case 1 tube‐hit for c_idx={c_idx} (h={new_h:.4f}). "
                    "Promoting to Support Vector (S)."
                )
                #print(f"[DBG][_update_params] Tube‐hit: adding idx={c_idx} to S_list")

                # 2) perform the S‐set addition
                added_successfully = self._update_R_inv_add_S(c_idx)
                #print(f"[DBG][_update_params] _update_R_inv_add_S({c_idx}) returned {added_successfully}; S_list={self.S_list}")

                # 3) clean up R/E/E* only if the matrix update succeeded
                if added_successfully:
                    self.R.discard(c_idx)
                    self.E.discard(c_idx)
                    self.E_star.discard(c_idx)
                    s_list_changed = True
            else:
                # The point is still outside the tube, classify as Error Vector
                logging.info(f"  _update_params: Case 1 resulted in an error vector for c_idx={c_idx} (h={new_h:.4f}). Classifying as E/E*.")
                if new_h > self.epsilon: # Corresponds to g_i* < 0
                    self.E_star.add(c_idx); self.E.discard(c_idx); self.R.discard(c_idx)
                elif new_h < -self.epsilon: # Corresponds to g_i < 0
                    self.E.add(c_idx); self.E_star.discard(c_idx); self.R.discard(c_idx)
                # s_list_changed remains False because the S set was not mutated.

            # ADDED LOGGING
            #print(f"[Update-OUT] idx={limiter_idx}, case={case_code}, "
            #      f"θ_after={new_theta:.4f}, h_after={new_h:.4f}")

            logger.debug(f"[upd_params] EXIT  c_idx={c_idx}: θ_c={self._get_current_theta(c_idx):.6f}, S_list={self.S_list}")
            new_h_final     = self._get_margin(c_idx)
            new_theta_final = self._get_current_theta(c_idx)
            logging.debug(
                f"[UPDATE_PARAMS_FINAL] c_idx={c_idx} · case={case_code} · "
                f"Δθ={chosen_DeltaTheta_c:.6f} → h={new_h_final:.6f}, θ={new_theta_final:.6f}"
            )
            return effective_delta_theta_c_candidate, s_list_changed

        # --- Standard CASE-BASED STRUCTURAL MIGRATION (Cases 2, 3, 4, 5) ---

        # Case 3: An existing Support Vector `s` migrates to R (theta -> 0).
        if case_code == 3:
            removal_succeeded = self._update_R_inv_remove_S(limiter_idx)
            if removal_succeeded:
                s_list_changed = True
                self.R.add(limiter_idx); self.E.discard(limiter_idx); self.E_star.discard(limiter_idx)

            # ADDED LOGGING
            new_theta = self._get_current_theta(limiter_idx)
            new_h = self._get_margin(limiter_idx)
            #print(f"[Update-OUT] idx={limiter_idx}, case={case_code}, "
            #      f"θ_after={new_theta:.4f}, h_after={new_h:.4f}")


        # Cases 4, 5: A non-S point `i` becomes a new Support Vector.
        elif case_code in [4, 5]:
            point_to_add = limiter_idx
            addition_succeeded = self._update_R_inv_add_S(point_to_add)
            if addition_succeeded:
                self.R.discard(point_to_add); self.E.discard(point_to_add); self.E_star.discard(point_to_add)
                s_list_changed = True

            # ADDED LOGGING
            new_theta = self._get_current_theta(limiter_idx)
            new_h = self._get_margin(limiter_idx)
            #print(f"[Update-OUT] idx={limiter_idx}, case={case_code}, "
            #      f"θ_after={new_theta:.4f}, h_after={new_h:.4f}")

        new_h_final = self._get_margin(c_idx)
        new_theta_final = self._get_current_theta(c_idx)
        logging.debug(
            "_update_params END: c_idx=%d, case=%d, Δθ=%.6f → new h=%.6f, new θ=%.6f, bias=%.6f",
            c_idx, case_code, chosen_DeltaTheta_c, new_h_final, new_theta_final, self.bias
        )
        logger.debug(f"[upd_params] EXIT  c_idx={c_idx}: θ_c={self._get_current_theta(c_idx):.6f}, S_list={self.S_list}")
        new_h_final     = self._get_margin(c_idx)
        new_theta_final = self._get_current_theta(c_idx)
        logging.debug(
            f"[UPDATE_PARAMS_FINAL] c_idx={c_idx} · case={case_code} · "
            f"Δθ={chosen_DeltaTheta_c:.6f} → h={new_h_final:.6f}, θ={new_theta_final:.6f}"
        )
        return effective_delta_theta_c_candidate, s_list_changed
    
    


    
    
    
    
    
    



















    def _update_sets(self):
        """
        Re-evaluates KKT conditions for all points and iteratively attempts to
        drive violating points to KKT satisfaction.

        [CORRECTED] This version includes a "fast-path" clamp for R -> E/E* violators.
        It ensures that points that are far outside the epsilon-tube and have a near-zero
        theta are immediately saturated to the +/- C boundary, preventing them from being
        incorrectly moved to the support set.
        """
        logging.info(f"Entering _update_sets ▶ initial S_list={self.S_list}")
        logging.info(
            f"_update_sets ENTRY: S_list={self.S_list}, E={self.E}, R={self.R}, E*={self.E_star}"
        )

        if not self.X_internal:
            self.S_list, self.E, self.E_star, self.R = [], set(), set(), set()
            self.R_inv, self.Q = None, None
            self._just_removed_last_support = None
            logging.info("_update_sets: EXIT (empty X_internal). Sets and inverse matrix cleared.")
            return

        MAX_OVERALL_KKT_ITERATIONS = 10

        for overall_iter_count in range(MAX_OVERALL_KKT_ITERATIONS):
            logging.info(f"_update_sets: Overall KKT satisfaction pass #{overall_iter_count + 1}/{MAX_OVERALL_KKT_ITERATIONS}")

            # Stage 1: Identify all KKT violators for the current pass
            violating_points_indices = []
            for i in range(len(self.X_internal)):
                if i == self._just_removed_last_support:
                    continue # Temporarily skip the point just removed from S

                kkt_satisfied, _ = self._check_kkt_satisfied_for_point(i)
                if not kkt_satisfied:
                    violating_points_indices.append(i)

            logging.info(f"  _update_sets pass {overall_iter_count + 1}: Found {len(violating_points_indices)} KKT violators: {violating_points_indices}")

            if not violating_points_indices:
                logging.info(f"  _update_sets pass {overall_iter_count + 1}: No KKT violators found. Convergence achieved.")
                self._just_removed_last_support = None # Clear the hold once stable
                break

            basis_mutated_in_pass = False
            # Stage 2: Process each violator one by one
            for c_idx_kkt_violator in violating_points_indices:
                logging.info(f"    _update_sets pass {overall_iter_count + 1}: Processing violator c_idx={c_idx_kkt_violator}")

                # --- [CORRECTION] Fast-path clamp for R -> E/E* violators ---
                h_violator_check = self._get_margin(c_idx_kkt_violator)
                theta_violator_check = self._get_current_theta(c_idx_kkt_violator)

                if (abs(h_violator_check) > self.epsilon + self.kkt_tol and
                    abs(theta_violator_check) < self.kkt_tol):

                    logging.info(f"      _update_sets: Detected R->E/E* violator c_idx={c_idx_kkt_violator}. Applying fast-path clamp.")

                    # 1. Calculate the change in theta needed to saturate at +/- C
                    theta_i = theta_violator_check
                    delta_theta_clamp = -np.sign(h_violator_check) * (self.C - theta_i)

                    # 2. Get sensitivities for the one-shot update
                    s_list_basis_clamp = self.S_list.copy()
                    beta0_c_clamp, betaS_c_clamp, _ = self._calculate_sensitivities(
                        c_idx_kkt_violator, s_list_basis_clamp, self.R_inv
                    )

                    # 3. Perform the one-shot update
                    self._update_params(
                        c_idx=c_idx_kkt_violator,
                        limiting_idxs=[c_idx_kkt_violator],
                        chosen_DeltaTheta_c=delta_theta_clamp,
                        case_code=2, # This is a Case 2 move (saturating at boundary)
                        beta0_c=beta0_c_clamp,
                        betaS_c=betaS_c_clamp,
                        s_list_basis_for_betaS_c=s_list_basis_clamp,
                        force_remove=False # The point is not in S, so nothing to remove from S
                    )

                    # 4. Update the point's classification set
                    if h_violator_check > 0:
                        self.E_star.add(c_idx_kkt_violator)
                    else:
                        self.E.add(c_idx_kkt_violator)
                    self.R.discard(c_idx_kkt_violator)
                    self.S_list.remove(c_idx_kkt_violator) if c_idx_kkt_violator in self.S_list else None


                    # 5. Skip the rest of the machinery for this violator
                    continue
                # --- END CORRECTION ---

                # Main adiabatic KKT satisfaction loop for other violators
                for inner_loop_iter_violator in range(self.max_iter):
                    kkt_satisfied_violator, _ = self._check_kkt_satisfied_for_point(c_idx_kkt_violator)
                    if kkt_satisfied_violator:
                        break # Done with this violator

                    h_violator = self._get_margin(c_idx_kkt_violator)
                    if np.isnan(h_violator): break

                    theta_violator_iter = self._get_current_theta(c_idx_kkt_violator)
                    q_direction = np.sign(-h_violator)
                    if abs(q_direction) < self.MACHINE_EPS:
                        q_direction = -np.sign(theta_violator_iter) if abs(theta_violator_iter) >= self.tol else 1.0

                    if q_direction == 0.0: break

                    s_list_basis_for_sens = self.S_list.copy()

                    h_violator = self._get_margin(c_idx_kkt_violator)
                    logging.debug(
                        f"[DBG ▶ _update_sets] violator c_idx={c_idx_kkt_violator} · RAW h={h_violator:.6f} · ε={self.epsilon:.6f}"
                    )
                    branch = (
                        "E/E* clamp" if abs(h_violator) > self.epsilon + self.kkt_tol
                        else "S-tube hit"  if abs(abs(h_violator) - self.epsilon) <= self.kkt_tol
                        else "R-inside"
                    )
                    logging.info(
                        f"[DEC ▶ _update_sets] c_idx={c_idx_kkt_violator} · chose '{branch}' (RAW h={h_violator:.6f})"
                    )

                    try:
                        beta0_c_viol, betaS_c_viol, gamma_c_viol = self._calculate_sensitivities(
                            c_idx_kkt_violator, s_list_basis_for_sens, self.R_inv
                        )
                        if np.isnan(beta0_c_viol) or (betaS_c_viol.size > 0 and np.isnan(betaS_c_viol).any()):
                            break
                    except Exception as e_sens:
                        logging.error(f"  iter {inner_loop_iter_violator}: Exception calculating sensitivities for c_idx={c_idx_kkt_violator}: {e_sens}", exc_info=True)
                        break

                    case_code_viol_step, limiting_idxs_viol_step, chosen_DeltaTheta_c_viol_step, _, force_remove_flag = \
                        self._calculate_max_step(c_idx_kkt_violator, q_direction)

                    is_existing_support = c_idx_kkt_violator in self.S_list
                    if case_code_viol_step == 2 and is_existing_support:
                        logging.info(f"    _update_sets: Overriding Case 2 for existing support SV c_idx={c_idx_kkt_violator}.")
                        delta_h = (-q_direction * self.epsilon) - h_violator

                        if abs(gamma_c_viol) > self.gamma_pivot_tol:
                            chosen_DeltaTheta_c_viol_step = delta_h / gamma_c_viol
                            case_code_viol_step = 1
                            limiting_idxs_viol_step = [c_idx_kkt_violator]
                            force_remove_flag = False
                            logging.info(f"      Overridden to Case 1. New Δθ={chosen_DeltaTheta_c_viol_step:.4e}")
                        else:
                            logging.warning(f"    _update_sets: Cannot override Case 2 for SV c_idx={c_idx_kkt_violator} because gamma is unstable. Skipping update.")
                            continue

                    if abs(chosen_DeltaTheta_c_viol_step) < self.MACHINE_EPS: break

                    try:
                        _, basis_mutated_in_pass = self._update_params(
                            c_idx=c_idx_kkt_violator,
                            limiting_idxs=limiting_idxs_viol_step,
                            chosen_DeltaTheta_c=chosen_DeltaTheta_c_viol_step,
                            case_code=case_code_viol_step,
                            beta0_c=beta0_c_viol,
                            betaS_c=betaS_c_viol,
                            s_list_basis_for_betaS_c=s_list_basis_for_sens,
                            force_remove=force_remove_flag
                        )
                        if basis_mutated_in_pass: continue
                    except Exception as e_update:
                        logging.error(f"  iter {inner_loop_iter_violator}: Exception in _update_params for c_idx={c_idx_kkt_violator}: {e_update}", exc_info=True)
                        break

                if basis_mutated_in_pass:
                    logging.info("  _update_sets: Basis mutated—restarting KKT pass")
                    break

            if basis_mutated_in_pass:
                self._just_removed_last_support = None
            else:
                logging.info(f"  _update_sets pass {overall_iter_count + 1}: Completed full pass over violators without basis mutation. Exiting.")
                break # Exit the main overall loop if a full pass completes without mutations


        # Final re-classification of all points to ensure sets are perfectly consistent
        logging.info("_update_sets: Starting Final KKT Re-classification phase...")
        final_S_set, final_E, final_R, final_E_star = set(), set(), set(), set()

        for i in range(len(self.X_internal)):
            _, label = self._check_kkt_satisfied_for_point(i)
            if label == "S": final_S_set.add(i)
            elif label == "E": final_E.add(i)
            elif label == "E*": final_E_star.add(i)
            else: final_R.add(i)

        new_S_list_ordered = sorted(list(final_S_set))
        old_S_list = self.S_list.copy()

        if set(old_S_list) != set(new_S_list_ordered):
            logging.info(
                f"_update_sets ▶ S_list changed from {old_S_list} to {new_S_list_ordered}, applying changes."
            )
            removed_svs = set(old_S_list) - set(new_S_list_ordered)
            added_svs = set(new_S_list_ordered) - set(old_S_list)
            logging.info(f"Final KKT re-class requires S_list change. Removing: {removed_svs}, Adding: {added_svs}")

            # It's safer and cleaner to just rebuild R_inv if the S_list changes this way
            self.S_list = new_S_list_ordered
            if self.S_list:
                self._compute_R_inv_initial()
            else:
                self.R_inv = self.Q = None


        self.E, self.R, self.E_star = final_E, final_R, final_E_star
        s_elements_final = set(self.S_list)
        self.E.difference_update(s_elements_final)
        self.E_star.difference_update(s_elements_final)
        self.R.difference_update(s_elements_final | self.E | self.E_star)

        logging.info(
            f"_update_sets EXIT: S_list={self.S_list}, E={self.E}, R={self.R}, E*={self.E_star}"
        )
        logging.info(
            "_update_sets EXIT: Final sizes → |S_list|=%d, |R|=%d, |E|=%d, |E*|=%d",
            len(self.S_list), len(self.R), len(self.E), len(self.E_star)
        )































        


    def _check_kkt_satisfied_for_point(self,
                                   point_idx: int,
                                   ) -> tuple[bool, str]:
        """
        Checks if Karush-Kuhn-Tucker (KKT) conditions are satisfied for a given point.

        This corrected version uses a dedicated small tolerance (self.kkt_tol) for
        all numerical comparisons and handles boundary conditions precisely to
        prevent misclassification. The order of checks is important to resolve
        ambiguities at the boundaries between sets.

        Args:
            point_idx: The index of the point to check.

        Returns:
            A tuple containing (is_satisfied, kkt_set_label_string).
        """
        tol = self.kkt_tol

        if not (0 <= point_idx < len(self.X_internal) and
                0 <= point_idx < len(self.alphas) and
                0 <= point_idx < len(self.alphas_star) and
                0 <= point_idx < len(self.y_internal)):
            logging.warning(f"_check_kkt_satisfied_for_point: point_idx {point_idx} out of bounds for internal arrays.")
            satisfied, label = False, "Index_Error"
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            return satisfied, label

        h = self._get_margin(point_idx)
        logger.debug(f"[KKT_chk] ENTER i={point_idx}: α={self.alphas[point_idx]:.6f}, α*={self.alphas_star[point_idx]:.6f}, h={h:.6f}")

        if np.isnan(h):
            logging.warning(f"_check_kkt_satisfied_for_point for idx {point_idx}: Margin h is NaN.")
            satisfied, label = False, "Margin_NaN"
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            return satisfied, label

        alpha_i = self.alphas[point_idx]
        alpha_star_i = self.alphas_star[point_idx]
        theta_i = alpha_i - alpha_star_i
        C_val = self.C
        epsilon_val = self.epsilon
        
        logging.debug(
            f"[KKT_ENTRY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
            f"ε={epsilon_val:.6f} · C={C_val:.6f} · tol={tol:.6f}"
        )

        logging.debug(
            f"Checking KKT ▶ idx={point_idx}, θ={theta_i:.6e}, h={h:.6e}, ε={epsilon_val:.2e}, tol={tol:.1e}"
        )

        # --- CORRECTED KKT CONDITION CHECKS ---
        # The order of these checks is crucial to correctly classify points
        # that lie on the boundaries between sets (e.g., θ=C and h=-ε).

        # E-set (Error): At the +C boundary and on or outside the lower tube boundary.
        if abs(theta_i - C_val) <= tol and h <= -epsilon_val + tol:
            logging.debug(f"KKT Case→E ▶ idx={point_idx} (θ≈C, h≤-ε)")
            satisfied, label = True, "E"
            logging.debug(
                f"[KKT_CLASSIFY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
                f"C={C_val:.6f} · ε={epsilon_val:.6f} → label={label}"
            )
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            theta_i = alpha_i - alpha_star_i
            logging.debug(
                f"[KKT_LABEL_DEBUG] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · ε={epsilon_val:.6f} · → label={label}"
            )
            return satisfied, label

        # E*-set (Error Star): At the -C boundary and on or outside the upper tube boundary.
        if abs(theta_i + C_val) <= tol and h >= epsilon_val - tol:
            logging.debug(f"KKT Case→E* ▶ idx={point_idx} (θ≈-C, h≥ε)")
            satisfied, label = True, "E*"
            logging.debug(
                f"[KKT_CLASSIFY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
                f"C={C_val:.6f} · ε={epsilon_val:.6f} → label={label}"
            )
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            theta_i = alpha_i - alpha_star_i
            logging.debug(
                f"[KKT_LABEL_DEBUG] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · ε={epsilon_val:.6f} · → label={label}"
            )
            return satisfied, label

        # S-set (Support Vector): Exactly on the ε-tube boundary.
        # This check comes after E/E* to avoid misclassifying boundary error vectors.
        if (
            abs(abs(h) - epsilon_val) <= tol
            and theta_i > tol
            and theta_i < C_val - tol
        ):
            logging.debug(f"KKT Case→S ▶ idx={point_idx} (|h|≈ε and 0<θ<C)")
            satisfied, label = True, "S"
            logging.debug(
                f"[KKT_CLASSIFY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
                f"C={C_val:.6f} · ε={epsilon_val:.6f} → label={label}"
            )
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            theta_i = alpha_i - alpha_star_i
            logging.debug(
                f"[KKT_LABEL_DEBUG] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · ε={epsilon_val:.6f} · → label={label}"
            )
            return satisfied, label

        # R-set (Remainder): Strictly inside the tube. θ must be zero.
        if abs(theta_i) <= tol and abs(h) < epsilon_val - tol:
            logging.debug(f"KKT Case→R ▶ idx={point_idx} (θ≈0, |h|<ε)")
            satisfied, label = True, "R"
            logging.debug(
                f"[KKT_CLASSIFY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
                f"C={C_val:.6f} · ε={epsilon_val:.6f} → label={label}"
            )
            logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
            theta_i = alpha_i - alpha_star_i
            logging.debug(
                f"[KKT_LABEL_DEBUG] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · ε={epsilon_val:.6f} · → label={label}"
            )
            return satisfied, label

        # If none of the strict conditions are met, it's a KKT violation.
        violation_label = f"KKT_VIOLATION(h={h:.2e},θ={theta_i:.2e})"
        logging.debug(f"KKT VIOLATION ▶ idx={point_idx}, label='{violation_label}'")
        satisfied, label = False, "Violation"
        logging.debug(
            f"[KKT_CLASSIFY] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · "
            f"C={C_val:.6f} · ε={epsilon_val:.6f} → label={label}"
        )
        logger.debug(f"[KKT_chk] EXIT  i={point_idx}: satisfied={satisfied}, label={label}")
        theta_i = alpha_i - alpha_star_i
        logging.debug(
            f"[KKT_LABEL_DEBUG] idx={point_idx} · h={h:.6f} · θ={theta_i:.6f} · ε={epsilon_val:.6f} · → label={label}"
        )
        return satisfied, label


























    def incremental_update(self, x_new: np.ndarray, y_new: float):
        """
        Incrementally adds a new sample (x_new, y_new) and updates the model.

        This corrected version restores the early exit after an initial margin-hit
        adjustment. This prevents points that are far outside the epsilon-tube
        (E/E* sets) from being incorrectly added to the support set (S). Only
        points that land exactly on the epsilon boundary are promoted to S.
        """
        logging.debug(f"ENTER incremental_update: sample_dim={x_new.shape}  y={y_new}")
        previous_active_idx = getattr(self, 'active_processing_idx', None)
        c_idx = len(self.X_internal)
        self.active_processing_idx = c_idx

        logging.debug(f"ENTER incremental_update: c_idx={c_idx}, _initialized={self._initialized}")

        # Guard prevents the second point from ever being processed here.
        # Initialization is handled by update_fit.
        if c_idx == 1 and not self._initialized:
            logging.debug("Guard hit: skipping incremental_update for c_idx=1 (second sample is handled by update_fit)")
            self.active_processing_idx = previous_active_idx
            return

        try:
            logging.info(f"incremental_update: ENTER c_idx={c_idx}, current S_list={self.S_list}, active_idx set to {self.active_processing_idx}")
            logging.debug("+++ [incremental_update] START new point: x_new=%r, y_new=%r", x_new, y_new)

            MACHINE_EPS = self.MACHINE_EPS
            x_new_arr = np.asarray(x_new)

            # Determine feature dimension if not already set
            if self._feature_dim is None:
                if x_new_arr.ndim > 0 and x_new_arr.size > 0:
                    self._feature_dim = x_new_arr.shape[0] if x_new_arr.ndim == 1 else x_new_arr.shape[1]
                elif self.X_internal:
                        first_pt_arr = np.asarray(self.X_internal[0])
                        self._feature_dim = first_pt_arr.shape[0] if first_pt_arr.ndim == 1 else first_pt_arr.shape[1]
                else:
                    logging.error(f"IncUpdate c_idx={c_idx}: Cannot determine feature dimension.")
                    self.active_processing_idx = previous_active_idx
                    return

            # Add the new point to internal data structures with theta = 0
            self.X_internal.append(x_new_arr.squeeze())
            self.y_internal.append(y_new)
            self.alphas.append(0.0)
            self.alphas_star.append(0.0)
            logging.info(f"incremental_update: Appended new point. c_idx={c_idx} is now active. len(X_internal)={len(self.X_internal)}.")

            # Handle the very first point (c_idx=0).
            if not self._initialized and c_idx == 0:
                self.alphas[0], self.alphas_star[0] = 0.0, 0.0
                self.bias = y_new
                self.R.add(0)
                logging.info(f"incremental_update: First point (c_idx=0) added. Set bias={self.bias:.3g}. Added to R set.")
                self.active_processing_idx = previous_active_idx
                return

            h_c = self._get_margin(c_idx)

            # Immediate check for points landing exactly on the S-boundary
            sat, lbl = self._check_kkt_satisfied_for_point(c_idx)
            if sat and lbl == "S":
                # point is exactly on ε-tube → must be a support vector. Add and exit.
                logging.info(f"incremental_update: New point c_idx={c_idx} lands on ε-tube (h_c={h_c:.4f}). Adding to S and exiting.")
                self._update_R_inv_add_S(c_idx)
                logging.debug("incremental_update: CASE 1 was true tube‐hit, returning early")
                self.active_processing_idx = previous_active_idx
                return

            # Initial adjustment for any point on or outside the tube
            if abs(h_c) >= self.epsilon - self.kkt_tol:
                logging.debug(f"incremental_update: in CASE 1 block, margin before={h_c:.4f}, epsilon={self.epsilon}")
                logging.info(f"incremental_update: New point c_idx={c_idx} is on or outside ε-tube (h_c={h_c:.4f}). Performing initial margin-hit update.")
                
                s_list_basis = self.S_list.copy()
                beta0_c, betaS_c, gamma_c = self._calculate_sensitivities(c_idx, s_list_basis, self.R_inv)

                if np.isnan(gamma_c) or abs(gamma_c) < self.gamma_pivot_tol:
                    logging.warning(f"  incremental_update c_idx={c_idx}: Unstable gamma_c ({gamma_c:.2e}). Cannot perform margin-hit. Falling to KKT loop.")
                else:
                    target_h = np.sign(h_c) * self.epsilon
                    delta_theta_margin_hit = (target_h - h_c) / gamma_c
                    
                    logging.debug(f"incremental_update: resolved case_code=1 at idx={c_idx}")
                    self._update_params(
                        c_idx=c_idx,
                        limiting_idxs=[c_idx],
                        chosen_DeltaTheta_c=delta_theta_margin_hit,
                        case_code=1, # This is an adiabatic move towards the margin
                        beta0_c=beta0_c,
                        betaS_c=betaS_c,
                        s_list_basis_for_betaS_c=s_list_basis,
                        force_remove=False
                    )
                    logging.info(f"  incremental_update c_idx={c_idx}: Initial Case 1 update complete. Exiting.")
                    
                    # --- [CORRECTION] Restore the early exit ---
                    # This step only moves the point *towards* the tube. It does not
                    # make it an SV unless it lands exactly on the boundary, which is
                    # handled by the check above. After this move, the point is left
                    # in its appropriate set (E, E*, or R) and we exit.
                    logging.debug("incremental_update: CASE 1 was true tube‐hit, returning early")
                    self.active_processing_idx = previous_active_idx
                    return
            
            # KKT Check for points already satisfied (e.g., in R-set after being added)
            kkt_satisfied_initial, kkt_label_initial = self._check_kkt_satisfied_for_point(c_idx)
            if kkt_satisfied_initial:
                logging.info(f"incremental_update: Point {c_idx} is already KKT-satisfied as '{kkt_label_initial}'. Adding to appropriate set and exiting.")
                if kkt_label_initial == "R":
                    self.R.add(c_idx)
                # The S case is handled by the immediate check, and E/E* cases are handled
                # by the margin-hit block above.
                self.active_processing_idx = previous_active_idx
                return

            # --- Adiabatic KKT Satisfaction Loop (for marginal violators) ---
            # This loop is now correctly entered only by points that were initially
            # inside the tube but are KKT violators (a rare edge case), or if the
            # initial margin-hit had unstable sensitivities.
            logging.info(
                "incremental_update: Point c_idx=%d is a KKT violator. Starting KKT adjustment loop.", c_idx
            )
            loop_count = 0
            while loop_count < self.max_iter:
                loop_count += 1

                kkt_c_satisfied_current, kkt_c_label_current = self._check_kkt_satisfied_for_point(c_idx)
                if kkt_c_satisfied_current:
                    logging.info(f"  iter {loop_count}: KKT for c_idx={c_idx} is SATISFIED as '{kkt_c_label_current}'. Breaking adjustment loop.")
                    break

                h_c_current_iter = self._get_margin(c_idx)
                if np.isnan(h_c_current_iter):
                    logging.warning(f"  iter {loop_count}: Margin h_c for c_idx={c_idx} became NaN. Breaking loop.")
                    break

                theta_c_current_iter = self._get_current_theta(c_idx)
                q_direction = np.sign(h_c_current_iter)
                if abs(q_direction) < MACHINE_EPS:
                    q_direction = -np.sign(theta_c_current_iter) if abs(theta_c_current_iter) >= self.tol else 1.0

                if q_direction == 0.0:
                    logging.warning(f"  iter {loop_count}: Direction q is 0 for c_idx={c_idx}, but KKT not met ('{kkt_c_label_current}'). Breaking loop.")
                    break

                s_list_basis_for_sens = self.S_list.copy()
                try:
                    beta0_c_sens, betaS_c_sens, _ = self._calculate_sensitivities(c_idx, s_list_basis_for_sens, self.R_inv)
                    if np.isnan(beta0_c_sens) or np.any(np.isnan(betaS_c_sens)):
                        break
                except Exception as e_sens:
                    logging.error(f"  iter {loop_count}: Exception calculating sensitivities for c_idx={c_idx}: {e_sens}", exc_info=True)
                    break

                case_code_step, limiting_idxs_step, chosen_DeltaTheta_c_step, _, force_remove_step = \
                    self._calculate_max_step(c_idx, q_direction)

                try:
                    if case_code_step < 0:
                        break
                    
                    logging.debug(f"incremental_update: resolved case_code={case_code_step} at idx={c_idx}")
                    s_list_basis_for_update = self.S_list.copy()
                    _, s_list_changed = self._update_params(
                        c_idx=c_idx,
                        limiting_idxs=limiting_idxs_step,
                        chosen_DeltaTheta_c=chosen_DeltaTheta_c_step,
                        case_code=case_code_step,
                        beta0_c=beta0_c_sens,
                        betaS_c=betaS_c_sens,
                        s_list_basis_for_betaS_c=s_list_basis_for_update,
                        force_remove=force_remove_step
                    )

                    if s_list_changed:
                        if case_code_step == 1 and c_idx in limiting_idxs_step:
                             break
                        else:
                            continue

                    kkt_after_update, label_after_update = self._check_kkt_satisfied_for_point(c_idx)
                    if kkt_after_update:
                        break

                except Exception as e_update:
                    logging.error(f"  iter {loop_count}: Exception in _update_params for c_idx={c_idx}: {e_update}", exc_info=True)
                    break

            if loop_count >= self.max_iter:
                logging.warning(f"incremental_update: c_idx={c_idx} MAX_ITER ({self.max_iter}) reached in KKT adjustment loop.")

        finally:
            # After any incremental update, re-classify the point to its correct final set.
            # This is a safety net to ensure consistency.
            if 0 <= c_idx < len(self.X_internal):
                is_sat, final_label = self._check_kkt_satisfied_for_point(c_idx)
                if is_sat:
                    if final_label == "E": self.E.add(c_idx); self.R.discard(c_idx)
                    elif final_label == "E*": self.E_star.add(c_idx); self.R.discard(c_idx)
                    elif final_label == "R": self.R.add(c_idx)
                    # S-set addition is handled by its own path.
                else:
                    logging.warning(f"Point {c_idx} ended incremental_update in a KKT-violating state.")

            self.active_processing_idx = previous_active_idx
            logging.info(f"incremental_update: EXIT for c_idx={c_idx}. Restored active_processing_idx to {previous_active_idx}. Final self.S_list={self.S_list.copy()}")































    def decremental_unlearn(self, c_idx: int):
        """
        Removes data point c_idx from the model by iteratively driving its
        coefficient theta_c to zero while maintaining KKT conditions for other points.

        [CORRECTED] This version now correctly passes self.R_inv to the sensitivity
        calculation within its main loop and cleans up the correct attributes
        (self.R_inv and self.Q) upon completion. It uses the proper rank-1 downdate
        procedure (_update_R_inv_remove_S) to preserve the integrity of the algorithm.

        [CORRECTED 2025-06-18] This version implements robust index shifting for all
        internal sets (S_list, R, E, E_star) after a point is physically removed. This
        ensures that all indices correctly refer to the positions in the updated
        dataset, preventing invalid index errors and incorrect kernel calculations.
        """
        previous_active_idx = self.active_processing_idx
        self.active_processing_idx = c_idx # Mark this point as the focus
        MACHINE_EPS = np.finfo(float).eps

        if not (0 <= c_idx < len(self.X_internal)):
            logging.warning(f"decremental_unlearn: c_idx {c_idx} is out of bounds. No action taken.")
            self.active_processing_idx = previous_active_idx
            return

        theta_c_initial = self._get_current_theta(c_idx)
        logging.info(f"decremental_unlearn: ENTRY for c_idx={c_idx}. Initial theta_c={theta_c_initial:.4e}.")

        # Fast path for points that are already non-contributing (in the R set)
        if abs(theta_c_initial) < self.tol:
            logging.info(f"decremental_unlearn: θ({c_idx})≈0; dropping point immediately.")

            s_list_before_shift = self.S_list.copy()

            # Physically remove the point from data lists
            if 0 <= c_idx < len(self.X_internal):
                del self.X_internal[c_idx]
                del self.y_internal[c_idx]
                del self.alphas[c_idx]
                del self.alphas_star[c_idx]

            # --- [Correction] Update all indices greater than c_idx in all sets ---
            # S_list is a list and must be handled separately. A new list is built.
            new_S_list = []
            for idx in self.S_list:
                if idx > c_idx:
                    new_S_list.append(idx - 1)
                elif idx < c_idx:
                    new_S_list.append(idx)
            self.S_list = new_S_list

            # R, E, and E_star are sets. They are cleared and updated in place.
            for set_to_update in [self.R, self.E, self.E_star]:
                updated_set = set()
                for idx in set_to_update:
                    if idx > c_idx:
                        updated_set.add(idx - 1)
                    elif idx < c_idx:
                        updated_set.add(idx)
                set_to_update.clear()
                set_to_update.update(updated_set)
            # --- End Correction ---

            self.active_processing_idx = previous_active_idx

            # If the S_list was affected by the index shift, a rebuild is necessary
            if any(i >= c_idx for i in s_list_before_shift):
                 self._compute_R_inv_initial()
            return

        # Main unlearning loop to drive theta_c to zero
        loop_count = 0
        while loop_count < self.max_iter:
            loop_count += 1
            theta_c_current = self._get_current_theta(c_idx)
            if abs(theta_c_current) < self.tol:
                logging.info(f"  iter {loop_count}: theta_c for c_idx={c_idx} is now ~0. Converged in loop.")
                break

            h_c = self._get_margin(c_idx)
            if np.isnan(h_c): break

            q_perturb_for_max_step = np.sign(h_c)
            if abs(q_perturb_for_max_step) < MACHINE_EPS:
                q_perturb_for_max_step = -np.sign(theta_c_current) if abs(theta_c_current) >= self.tol else 1.0

            s_list_base_for_sens = self.S_list.copy()

            beta0_c_sens, betaS_c_sens, _ = self._calculate_sensitivities(c_idx, s_list_base_for_sens, self.R_inv)

            if np.isnan(beta0_c_sens) or (betaS_c_sens.size > 0 and np.isnan(betaS_c_sens).any()):
                break

            case_code_step, limiting_idxs_step, chosen_DeltaTheta_c_boundary, _, force_remove_step = \
                self._calculate_max_step(c_idx, q_perturb_for_max_step)

            Delta_theta_c_to_zero = -theta_c_current
            applied_Delta_theta_c_for_c = 0.0
            final_case_code_for_update = case_code_step

            if abs(Delta_theta_c_to_zero) <= abs(chosen_DeltaTheta_c_boundary) or chosen_DeltaTheta_c_boundary == 0.0:
                applied_Delta_theta_c_for_c = Delta_theta_c_to_zero
                final_case_code_for_update = 2 # Treat as hitting its own boundary
            else:
                applied_Delta_theta_c_for_c = chosen_DeltaTheta_c_boundary

            if abs(applied_Delta_theta_c_for_c) < MACHINE_EPS:
                break

            _, basis_changed = self._update_params(
                c_idx=c_idx, limiting_idxs=limiting_idxs_step,
                chosen_DeltaTheta_c=applied_Delta_theta_c_for_c, case_code=final_case_code_for_update,
                beta0_c=beta0_c_sens, betaS_c=betaS_c_sens,
                s_list_basis_for_betaS_c=s_list_base_for_sens,
                force_remove=force_remove_step
            )
            if basis_changed:
                continue
            
        # --- Post-loop cleanup ---
        logging.info(f"decremental_unlearn: c_idx={c_idx}: Loop finished. Starting final cleanup.")

        if 0 <= c_idx < len(self.alphas):
            self.alphas[c_idx], self.alphas_star[c_idx] = 0.0, 0.0

        if c_idx in self.S_list:
            logging.info(f"  Cleanup: c_idx={c_idx} was in S_list. Downdating R_inv.")
            self._update_R_inv_remove_S(c_idx)

        self.E.discard(c_idx)
        self.R.discard(c_idx)
        self.E_star.discard(c_idx)

        logging.info(f"  Cleanup: Calling _update_sets to restore global KKT consistency.")
        self._update_sets()

        s_list_before_shift = self.S_list.copy()
        logging.info(f"  Cleanup: Physically removing point {c_idx} and shifting indices.")
        if 0 <= c_idx < len(self.X_internal):
            del self.X_internal[c_idx]
            del self.y_internal[c_idx]
            del self.alphas[c_idx]
            del self.alphas_star[c_idx]

        # --- [Correction] Update all indices greater than c_idx in all sets ---
        # S_list is a list and must be handled separately. A new list is built.
        new_S_list = []
        for idx in self.S_list:
            if idx > c_idx:
                new_S_list.append(idx - 1)
            elif idx < c_idx:
                new_S_list.append(idx)
        self.S_list = sorted(new_S_list) # Keep the S_list sorted

        # R, E, and E_star are sets. They are cleared and updated in place.
        for set_to_update in [self.R, self.E, self.E_star]:
            updated_set = set()
            for idx in set_to_update:
                if idx > c_idx:
                    updated_set.add(idx - 1)
                elif idx < c_idx:
                    updated_set.add(idx)
            set_to_update.clear()
            set_to_update.update(updated_set)
        # --- End Correction ---

        if any(i >= c_idx for i in s_list_before_shift):
             if self.S_list:
                logging.info(f"  Cleanup: S_list indices shifted. Rebuilding R_inv.")
                self._compute_R_inv_initial()
             else:
                self.R_inv = self.Q = None

        self.active_processing_idx = previous_active_idx
        logging.info(f"decremental_unlearn: EXIT for original c_idx={c_idx}. Final S_list={self.S_list}")



























    def predict(self, X_pred: np.ndarray):
        """Predicts target values for new input samples X_pred."""
        if not self._initialized or not self.X_internal:
            # Return bias? Or zeros? Bias seems more reasonable if known.
            bias_to_return = self.bias if hasattr(self, 'bias') else 0.0
            return np.full(X_pred.shape[0], bias_to_return)

        # Input validation and reshaping
        if X_pred.ndim == 1:
            # Try reshaping based on known feature dim
            if self._feature_dim is not None and X_pred.shape[0] == self._feature_dim:
                 X_pred = X_pred.reshape(1, self._feature_dim)
            else:
                 # Fallback reshape (might be wrong if single sample != feature dim)
                 try:
                    X_pred = X_pred.reshape(1,-1)
                    if self._feature_dim is not None and X_pred.shape[1] != self._feature_dim:
                        raise ValueError("Reshaped 1D input dim doesn't match model feature dim.")
                 except Exception as e:
                     return np.full(X_pred.shape[0], np.nan)

        elif X_pred.ndim != 2:
            return np.full(X_pred.shape[0], np.nan)

        # Check feature dimension consistency
        if self._feature_dim is not None and X_pred.shape[1] != self._feature_dim:
             return np.full(X_pred.shape[0], np.nan)


        num_internal = len(self.X_internal)
        # Check state consistency
        if not (num_internal == len(self.alphas) == len(self.alphas_star)):
             return np.full(X_pred.shape[0], np.nan)

        # Prepare internal data for kernel calculation
        try:
            # Combine alphas
            thetas = np.array(self.alphas) - np.array(self.alphas_star)

            # Get internal X data as a 2D numpy array
            X_all_list = [np.asarray(x) for x in self.X_internal]
            if self._feature_dim == 1:
                 X_all = np.array([x.reshape(1) for x in X_all_list]).reshape(num_internal, 1)
            else:
                 X_all = np.array([x.reshape(self._feature_dim) for x in X_all_list])

            if X_all.ndim != 2 and num_internal > 0: # Reshape if somehow flattened
                 X_all = X_all.reshape(num_internal, -1)

        except Exception as e:
            return np.full(X_pred.shape[0], np.nan)


        # Handle prediction if no internal points exist (should be caught earlier, but safe check)
        if X_all.size == 0:
             return np.full(X_pred.shape[0], self.bias)

        # Compute Kernels K(X_pred, X_all)
        try:
            kernels = self._kernel(X_pred, X_all) # Shape (n_pred, n_internal)
        except Exception as e:
            return np.full(X_pred.shape[0], np.nan)

        # Calculate final predictions: predictions = Kernels * thetas + bias
        try:
            if kernels.shape[1] != len(thetas):
                 raise ValueError(f"Kernel columns ({kernels.shape[1]}) don't match theta length ({len(thetas)})")

            predictions = np.dot(kernels, thetas) + self.bias
            return predictions

        except Exception as e:
            return np.full(X_pred.shape[0], np.nan)



































    def update_fit(self, X: "DataFrame", Y: "DataFrame", X_pred: "DataFrame", **kwargs) -> "ForecastTuple":
        """
        Incrementally/decrementally updates the SVR model with new data samples and hyperparameters.

        This method processes a batch of data (X, Y) to update the model's state. It handles:
        - Updating hyperparameters (C, epsilon, gamma) if provided in kwargs.
        - Initializing the model if it hasn't seen enough data.
        - Incrementally adding new valid samples one by one.
        - Decrementally unlearning the oldest sample if the 'threshold' is exceeded.
        - Finally, it produces predictions for the `X_pred` dataframe based on the updated model state.

        CORRECTION: The initialization logic has been moved directly into this method to
        prevent the second data point from being processed by the `incremental_update`
        method's KKT satisfaction loop. The first two points are now used to call
        `initialize_model` directly, and only subsequent points are passed to
        `incremental_update`.
        """
        logging.debug(f"ENTER update_fit: _initialized={self._initialized}")

        updated_params = {}
        gamma_changed = False

        # Check for hyperparameter updates from kwargs
        if 'C' in kwargs:
            try:
                new_C = float(kwargs['C'][0]) if isinstance(kwargs['C'], (np.ndarray, list, tuple)) and len(kwargs['C']) > 0 else float(kwargs['C'])
                if abs(self.C - new_C) > 1e-9:
                    self.C = new_C
                    updated_params['C'] = self.C
            except (TypeError, ValueError, IndexError) as e_c:
                logging.warning(f"OnlineSVR.update_fit: Could not parse C parameter from kwargs['C']={kwargs.get('C')}. Error: {e_c}")
                pass
        if 'epsilon' in kwargs:
            try:
                new_epsilon = float(kwargs['epsilon'][0]) if isinstance(kwargs['epsilon'], (np.ndarray, list, tuple)) and len(kwargs['epsilon']) > 0 else float(kwargs['epsilon'])
                if abs(self.epsilon - new_epsilon) > 1e-9:
                    self.epsilon = new_epsilon
                    updated_params['epsilon'] = self.epsilon
            except (TypeError, ValueError, IndexError) as e_eps:
                logging.warning(f"OnlineSVR.update_fit: Could not parse epsilon parameter from kwargs['epsilon']={kwargs.get('epsilon')}. Error: {e_eps}")
                pass
        if 'gamma' in kwargs and self.kernel_type == 'rbf':
            try:
                new_gamma = float(kwargs['gamma'][0]) if isinstance(kwargs['gamma'], (np.ndarray, list, tuple)) and len(kwargs['gamma']) > 0 else float(kwargs['gamma'])
                if abs(self.gamma - new_gamma) > 1e-9:
                    self.gamma = new_gamma
                    updated_params['gamma'] = self.gamma
                    gamma_changed = True # Flag that a full rebuild is necessary
            except (TypeError, ValueError, IndexError) as e_gamma:
                logging.warning(f"OnlineSVR.update_fit: Could not parse gamma parameter from kwargs['gamma']={kwargs.get('gamma')}. Error: {e_gamma}")
                pass
            
        if updated_params:
            logging.info(f"OnlineSVR.update_fit: Model hyperparameters updated: {updated_params}")

        # If gamma changed, the kernel matrix is invalid, forcing a rebuild of R_inv and a global KKT check.
        if gamma_changed and self._initialized:
            logging.info("OnlineSVR.update_fit: Gamma changed, forcing KKT matrix rebuild and global consistency check.")
            self._compute_R_inv_initial()
            self._update_sets()

        # Convert pandas inputs to numpy arrays for processing
        try:
            x_np = X.to_numpy(dtype=float)
            y_np = Y.to_numpy(dtype=float).flatten()
            x_pred_np = X_pred.to_numpy(dtype=float)
        except Exception as e:
            logging.error(f"OnlineSVR.update_fit: Error converting input DataFrames to NumPy arrays: {e}")
            out_cols = self.y_columns if hasattr(self, 'y_columns') and self.y_columns is not None else (Y.columns if Y is not None and not Y.empty else pd.MultiIndex.from_tuples([("Y_pred", "NA")]))
            if not isinstance(out_cols, pd.MultiIndex) and out_cols is not None:
                out_cols = pd.MultiIndex.from_tuples([(str(c), "NA") for c in out_cols])
            pred_df = new_fc(np.full((len(X_pred.index), len(out_cols)), np.nan), index=X_pred.index, columns=out_cols)
            return ForecastTuple(pred_df)

        if self._feature_dim is None:
            first_valid_idx_mask = ~np.isnan(x_np).any(axis=1)
            first_valid_idx = np.where(first_valid_idx_mask)[0]
            if len(first_valid_idx) > 0:
                self._feature_dim = x_np[first_valid_idx[0]].shape[0]
            elif len(self.X_internal) > 0:
                first_internal_x = np.asarray(self.X_internal[0])
                if first_internal_x.ndim > 0:
                    self._feature_dim = first_internal_x.shape[0]
                else:
                    self._feature_dim = 1
            if self._feature_dim is not None:
                logging.info(f"OnlineSVR.update_fit: _feature_dim determined to be {self._feature_dim}")

        n_t = x_np.shape[0]
        start_i = 0

        # Model initialization logic, now handled explicitly before the main loop
        if not self._initialized:
            logging.debug("update_fit: taking INIT branch")
            valid_indices_in_batch = [idx for idx in range(n_t) if not (np.isnan(x_np[idx]).any() or np.isnan(y_np[idx]))]
            num_existing_points = len(self.X_internal)

            if num_existing_points == 0 and len(valid_indices_in_batch) >= 2:
                # Case A: 0 existing points, >= 2 in batch -> Initialize from batch
                idx0, idx1 = valid_indices_in_batch[0], valid_indices_in_batch[1]
                logging.info(f"OnlineSVR.update_fit: Initializing on two samples from batch indices {idx0} and {idx1}.")
                self.initialize_model(x_np[idx0], y_np[idx0], x_np[idx1], y_np[idx1])
                logging.debug("update_fit: set _initialized=True")
                start_i = idx1 + 1 # Start main loop after the second point

            elif num_existing_points == 1 and len(valid_indices_in_batch) >= 1:
                # Case B: 1 existing point, >= 1 in batch -> Initialize with existing and new point
                idx1 = valid_indices_in_batch[0]
                logging.info(f"OnlineSVR.update_fit: Initializing with one stored point and new sample from batch index {idx1}.")
                x0_arr, y0_val = np.asarray(self.X_internal[0]), self.y_internal[0]
                self.initialize_model(x0_arr, y0_val, x_np[idx1], y_np[idx1])
                logging.debug("update_fit: set _initialized=True")
                start_i = idx1 + 1 # Start main loop after the second point

            elif num_existing_points == 0 and len(valid_indices_in_batch) == 1:
                # Case C: 0 existing points, 1 in batch -> Add the first point and wait for the next
                idx0 = valid_indices_in_batch[0]
                logging.info(f"OnlineSVR.update_fit: Processing single valid point at batch index {idx0}.")
                self.incremental_update(x_np[idx0], y_np[idx0]) # It's safe to use for point 0
                start_i = idx0 + 1

            else: # Not enough points to initialize or process
                if num_existing_points == 0 and len(valid_indices_in_batch) == 0:
                    logging.warning("OnlineSVR.update_fit: No valid data in this batch to process or initialize with.")
                start_i = n_t # Skip main loop


        # Main loop for processing samples from the batch (points 3 onwards)
        for i in range(start_i, n_t):
            logging.debug(f"update_fit: received sample #{i}")
            current_x = x_np[i]
            current_y = y_np[i]

            data_idx_info = f"input batch row {i}"
            if hasattr(X, 'index') and i < len(X.index):
                data_idx_info = f"input batch index {X.index[i]} (row {i})"

            if np.isnan(current_x).any() or np.isnan(current_y):
                logging.debug(f"OnlineSVR.update_fit: Skipping {data_idx_info} due to NaN values.")
                continue
            if self._feature_dim is not None and current_x.shape[0] != self._feature_dim:
                logging.warning(f"OnlineSVR.update_fit: Skipping {data_idx_info} due to feature dimension mismatch (expected {self._feature_dim}, got {current_x.shape[0]}).")
                continue
            
            logging.debug(f"update_fit ▶ sample #{i}: x={current_x}, y={current_y}")

            # Decremental unlearning if threshold is met
            if self.threshold is not None and len(self.X_internal) > self.threshold:
                if self.X_internal:
                    logging.info(f"OnlineSVR.update_fit ({data_idx_info}): Threshold met. Decrementally unlearning oldest sample.")
                    try:
                        self.decremental_unlearn(0)
                    except Exception as e_decr:
                        logging.error(f"OnlineSVR.update_fit ({data_idx_info}): Error during decremental_unlearn: {e_decr}", exc_info=True)
                else:
                    logging.warning(f"OnlineSVR.update_fit ({data_idx_info}): Threshold met but X_internal is empty. Cannot unlearn.")

            # Incremental learning for the new point
            logging.debug(f"OnlineSVR.update_fit: Incrementally updating with sample from {data_idx_info}.")
            try:
                self.incremental_update(current_x, current_y)
            except Exception as e_incr:
                logging.error(f"OnlineSVR.update_fit: Error during incremental_update for {data_idx_info}: {e_incr}", exc_info=True)
                logging.warning(f"OnlineSVR.update_fit: Skipping point {data_idx_info} due to an error in its update process.")
                if len(self.X_internal) > 0:
                    failed_point_internal_idx = len(self.X_internal) - 1
                    try:
                        del self.X_internal[failed_point_internal_idx]
                        del self.y_internal[failed_point_internal_idx]
                        del self.alphas[failed_point_internal_idx]
                        del self.alphas_star[failed_point_internal_idx]
                    except IndexError:
                        logging.error(f"OnlineSVR.update_fit: Could not roll back addition of point {data_idx_info}.")
                continue
        
        # Make predictions on X_pred
        if len(self.X_internal) > 0:
            sample_margins = [self._get_margin(i) for i in range(min(3, len(self.X_internal)))]
            logging.debug(f"[FIT_DEBUG] Sample margins before prediction: {sample_margins}")
        predictions_np = np.full(x_pred_np.shape[0], np.nan, dtype=float)
        if x_pred_np.shape[0] > 0:
            if self._feature_dim is None and self._initialized and self.X_internal:
                first_internal_x_pred = np.asarray(self.X_internal[0])
                self._feature_dim = first_internal_x_pred.shape[0] if first_internal_x_pred.ndim > 0 else 1

            valid_indices_mask = ~np.isnan(x_pred_np).any(axis=1)
            valid_indices = np.where(valid_indices_mask)[0]

            if len(valid_indices) > 0:
                valid_x_pred_np = x_pred_np[valid_indices]
                try:
                    if self._feature_dim is not None and valid_x_pred_np.shape[1] != self._feature_dim:
                         logging.warning(f"OnlineSVR.update_fit: X_pred feature dimension mismatch. Predictions will be NaN.")
                    else:
                        valid_predictions = self.predict(valid_x_pred_np)
                        predictions_np[valid_indices] = valid_predictions
                except Exception as e_pred:
                    logging.error(f"OnlineSVR.update_fit: Error during final prediction on X_pred: {e_pred}", exc_info=True)
            else:
                logging.debug("OnlineSVR.update_fit: No valid (non-NaN) rows in X_pred to make predictions for.")
        else:
            predictions_np = np.array([])

        # Format predictions into DataFrame
        out_cols = self.y_columns
        if out_cols is None and Y is not None and not Y.empty:
            out_cols = Y.columns
        if out_cols is None:
            out_cols = pd.MultiIndex.from_tuples([("Y_pred", "NA")], names=['Variable', 'Horizon'])
        elif not isinstance(out_cols, pd.MultiIndex):
             out_cols = pd.MultiIndex.from_tuples([(str(c),"NA") for c in out_cols])

        num_outputs = len(out_cols) if out_cols is not None else 0
        pred_df = None

        if predictions_np.size > 0 or len(X_pred.index) == 0:
            try:
                if num_outputs == 0 and predictions_np.size > 0:
                    num_outputs = 1
                    out_cols = pd.MultiIndex.from_tuples([("Y_pred", "NA")])

                if predictions_np.ndim == 1 and num_outputs == 1:
                    reshaped_preds = predictions_np.reshape(-1, 1)
                elif predictions_np.ndim == 2 and predictions_np.shape[1] == num_outputs:
                    reshaped_preds = predictions_np
                elif predictions_np.ndim == 1 and num_outputs > 1 and predictions_np.size == 0 and len(X_pred.index) > 0:
                    reshaped_preds = np.full((len(X_pred.index), num_outputs), np.nan)
                elif predictions_np.ndim == 1 and num_outputs > 1 and predictions_np.size > 0:
                    if predictions_np.shape[0] == len(X_pred.index):
                        logging.warning(f"OnlineSVR.update_fit: Replicating single prediction series for {num_outputs} expected outputs.")
                        reshaped_preds = np.tile(predictions_np.reshape(-1,1), (1, num_outputs))
                    else:
                        raise ValueError("Shape mismatch between prediction series and X_pred for multi-output.")
                elif predictions_np.ndim == 0 and len(X_pred.index) == 1 and num_outputs ==1:
                    reshaped_preds = np.array([[predictions_np]])
                else:
                    if not (predictions_np.size == 0 and len(X_pred.index) == 0):
                        raise ValueError(f"Incompatible prediction shape {predictions_np.shape} for {num_outputs} outputs.")
                    else:
                        reshaped_preds = np.empty((0,num_outputs))

                pred_df = new_fc(reshaped_preds, index=X_pred.index, columns=out_cols)
            except Exception as e_format:
                logging.error(f"OnlineSVR.update_fit: Error formatting predictions: {e_format}. Returning NaNs.", exc_info=True)
                pred_df = new_fc(np.full((len(X_pred.index), num_outputs if num_outputs > 0 else 1), np.nan), index=X_pred.index, columns=out_cols if num_outputs > 0 else pd.MultiIndex.from_tuples([("Y_pred", "NA")]))
        else:
            pred_df = new_fc(np.empty((len(X_pred.index), num_outputs if num_outputs > 0 else 0)) * np.nan, index=X_pred.index, columns=out_cols)

        return ForecastTuple(pred_df)









##----------------------------------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##












##----------------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------Adaptive Random Forests Regression-----------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##
class ARFR(Predictor):
    """
    Hoeffding Regression Trees (HRT)
    An adaptation of the Hoeffding Trees algorithm for regression tasks on evolving data streams,
    now with horizon awareness and fixed seeding.
    """
    params = ['n_trees','max_features','lambda_value','delta_warning','delta_drift','burn_in',
              'grace_period','max_depth','split_confidence', 'target_model_horizon']

    def __init__(self, n_trees=10, max_features=None, lambda_value=6,
                 delta_warning=0.000015, delta_drift=0.00001, burn_in=1,
                 x_columns=None, y_columns=None, random_state=None,
                 target_model_horizon=None, grace_period=200, max_depth=None, split_confidence=1e-7,
                 model_selector_decay=0.95, leaf_prediction='adaptive'):
        super().__init__(x_columns=x_columns, y_columns=y_columns)
        self.n_trees = n_trees
        self.max_features_config = max_features
        self.lambda_value = lambda_value
        self.delta_warning = delta_warning
        self.delta_drift = delta_drift
        self.burn_in = burn_in
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.target_model_horizon = target_model_horizon
        self.target_model_selector_decay = model_selector_decay
        self.leaf_prediction = leaf_prediction
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()
        self.target_model_horizon = target_model_horizon
        self._n_updates = 0

        # Ensemble components
        self.trees = []
        self.drift_detectors = []
        self.warning_detectors = []
        self.background_trees = {}
        self.weights = []

        # Feature bookkeeping
        self.feature_names_original = None
        self.feature_names_for_model = None
        self.feature_map = None
        self.n_features_original = 0
        self.n_features_for_model = 0
        self.actual_max_features = 0

        self._is_initialized = False

        

    def _initialize(self, X: pd.DataFrame, Y: pd.DataFrame):
        if X.empty:
            warn("ARFR initialization called with empty X.", UserWarning)
            self._is_initialized = False
            return

        

        # Flatten MultiIndex if needed
        if isinstance(X.columns, pd.MultiIndex):
            self.feature_names_original = [
                f"{c[0]}_{c[1]}" if c[1] not in (None, 'NA') else c[0]
                for c in X.columns
            ]
            self.feature_map = {
                fn: col for fn, col in zip(self.feature_names_original, X.columns)
            }
        else:
            self.feature_names_original = X.columns.tolist()
            self.feature_map = {fn: fn for fn in self.feature_names_original}

        self.n_features_original = len(self.feature_names_original)
        # Horizon filtering
        if (self.target_model_horizon is not None
                and isinstance(X.columns, pd.MultiIndex)):
            self.feature_names_for_model = []
            temp_map = {}
            for col in X.columns:
                var, hor = col
                if hor == 'NA' or int(hor) == self.target_model_horizon:
                    flat = f"{var}_{hor}" if hor not in (None, 'NA') else var
                    self.feature_names_for_model.append(flat)
                    temp_map[flat] = col
            if self.feature_names_for_model:
                self.feature_map = temp_map
        else:
            self.feature_names_for_model = self.feature_names_original

        self.n_features_for_model = len(self.feature_names_for_model)
        

        # Determine actual_max_features
        if self.max_features_config is None or self.max_features_config == 'sqrt':
            self.actual_max_features = max(1, int(np.sqrt(self.n_features_for_model)))
        elif isinstance(self.max_features_config, int):
            self.actual_max_features = max(
                1, min(self.max_features_config, self.n_features_for_model)
            )
        else:
            warn(f"Invalid max_features {self.max_features_config}, defaulting.", UserWarning)
            self.actual_max_features = max(1, int(np.sqrt(self.n_features_for_model)))

        # Build trees + detectors
        self.trees = []
        self.drift_detectors = []
        self.warning_detectors = []
        self.weights = []
        for i in range(self.n_trees):
            tc = self._create_tree()
            if tc:
                self.trees.append(tc)
                self.drift_detectors.append(PageHinkleyTest(delta=self.delta_drift,  lambda_=0.0))
                self.warning_detectors.append(PageHinkleyTest(delta=self.delta_warning,  lambda_=0.0))
                self.weights.append(1.0)
            else:
                warn(f"Failed to create tree {i}.", UserWarning)

        
        self._is_initialized = bool(self.trees)

    def _create_tree(self):
        if not self.feature_names_for_model:
            return None
        tree = HoeffdingTreeRegressor()
        k = min(self.actual_max_features, self.n_features_for_model)
        try:
            feats = self.random_state.choice(
                self.feature_names_for_model, size=k, replace=False
            ).tolist()
        except ValueError:
            feats = self.random_state.choice(
                self.feature_names_for_model, size=k, replace=True
            ).tolist()
        return {'tree': tree,
                'features': feats,
                'n_samples': 0,
                'created_at_update': self._n_updates}

    def _train_tree(self, tree_component, x_row: pd.Series,
                    y_val: float, k_poisson: int):
        if not tree_component['features']:
            return
        xdict = {fn: x_row.get(self.feature_map.get(fn), np.nan)
                 for fn in tree_component['features']}
        for _ in range(k_poisson):
            tree_component['tree'].learn_one(xdict, y_val)
        tree_component['n_samples'] += 1


    def _predict_tree(self, tree_component, x_row: pd.Series) -> float:
        if (not tree_component['features'] 
                or tree_component['n_samples'] < self.burn_in):
            return np.nan
        try:
            xdict = {fn: x_row.get(self.feature_map.get(fn), np.nan)
                     for fn in tree_component['features']}
            pred = tree_component['tree'].predict_one(xdict)
            return pred if pred is not None else np.nan
        except Exception:
            return np.nan

    def _detect_warnings_and_drifts(self, errors):
        warns, drifts = [], []
        for i, (w, d) in enumerate(
                zip(self.warning_detectors, self.drift_detectors)):
            err = errors[i]
            if np.isnan(err):
                continue
            w.update(err)
            d.update(err)
            if w.drift_detected:
                warns.append(i)
            if d.drift_detected:
                drifts.append(i)
        return warns, drifts

    def _update_weights(self, errors):
        old = self.weights.copy()
        eps = 1e-10
        new_w = [0.0 if (np.isnan(e) or e < 0) else 1.0/(e+eps)
                 for e in errors]
        s = sum(new_w)
        if s > 0:
            self.weights = [w/s for w in new_w]
        else:
            valid = [w for w in new_w if w > 0]
            if valid:
                self.weights = [
                    1.0/len(valid) if w>0 else 0.0
                    for w in new_w
                ]
        

    def update_fit(self, X: pd.DataFrame, Y: pd.DataFrame,
               X_pred: pd.DataFrame, **params) -> ForecastTuple:
        """
        Perform one-step ahead (or multi-step) forecast with alignment of inputs and targets.
        """
        # Align target Y for forecasting horizon
        h = self.target_model_horizon or 0
        if h > 0:
            # Shift Y so that Y_train[t] corresponds to original Y[t+h]
            Y_shifted = Y.shift(-h)
            X_train = X.iloc[:-h]
            Y_train = Y_shifted.iloc[:-h]
        else:
            X_train = X
            Y_train = Y

        # Reinitialize if needed
        if not self._is_initialized:
            self._initialize(X_train, Y_train)
            if not self._is_initialized:
                warn("ARFR not initialized; returning NaNs.", UserWarning)
                nan_arr = np.full((len(X_pred), Y.shape[1] if Y is not None else 1),
                                  np.nan)
                cols = Y.columns if Y is not None else [("Y_hat", self.target_model_horizon or "NA")]
                return ForecastTuple(new_fc(nan_arr, index=X_pred.index, columns=cols))

        n = len(X_train)
        num_out = Y_train.shape[1] if Y_train is not None else 1
        preds = np.full((len(X_pred), num_out), np.nan)

        for i in range(n):
            x_i = X_train.iloc[i]
            y_i = Y_train.iloc[i, 0] if num_out == 1 else Y_train.iloc[i].mean()
            if x_i.isna().any() or np.isnan(y_i):
                continue

            self._n_updates += 1
            tree_preds, errors = [], []

            # 1) Make predictions and compute squared errors
            for t, tc in enumerate(self.trees):
                p = self._predict_tree(tc, x_i)
                tree_preds.append(p)
                err = (p - y_i)**2 if not np.isnan(p) else np.nan
                errors.append(err)

            # 2) Detect warnings and drifts
            warns, drifts = self._detect_warnings_and_drifts(errors)

            # 3) Spawn background trees on warnings
            for t in warns:
                if t not in self.background_trees:
                    self.background_trees[t] = self._create_tree()

            # 4) Swap in background (or rebuild) on drifts; reinit detectors
            for t in drifts:
                if t in self.background_trees:
                    self.trees[t] = self.background_trees.pop(t)
                else:
                    self.trees[t] = self._create_tree()
                self.drift_detectors[t]   = PageHinkleyTest(delta=self.delta_drift,  lambda_=0.0)
                self.warning_detectors[t] = PageHinkleyTest(delta=self.delta_warning,  lambda_=0.0)

            # 5) Update ensemble weights
            self._update_weights(errors)

            # 6) Train primary trees
            for t, tc in enumerate(self.trees):
                k_p = self.random_state.poisson(self.lambda_value)
                self._train_tree(tc, x_i, y_i, k_p)

            # 7) Train background trees
            for t, b in self.background_trees.items():
                k_pb = self.random_state.poisson(self.lambda_value)
                self._train_tree(b, x_i, y_i, k_pb)

            # 8) Compute and store ensemble forecast
            ens = np.dot(self.weights, tree_preds)
            preds[i, :] = ens

        # Build forecast container with raw preds and original X_pred index
        cols = (Y.columns if Y is not None else
                pd.MultiIndex.from_tuples([("Y_hat", self.target_model_horizon or "NA")]))
        fc = new_fc(preds, index=X_pred.index, columns=cols)
        ft = ForecastTuple(fc)
        if h > 0:
            ft = ft.get_lagged_subset()
        return ft


class PageHinkleyTest:
    """
    Page-Hinkley Test for drift detection in regression tasks.
    
    This test monitors the prediction error of a model and detects
    when the error rate increases significantly.
    """
    
    def __init__(self, delta=0.005, lambda_=50, alpha=0.999):
        """
        Initialize Page-Hinkley Test.
        
        Parameters:
        -----------
        delta : float, default=0.005
            Detection threshold.
        lambda_ : float, default=50
            Lambda parameter (magnitude of changes that are allowed).
        alpha : float, default=0.9999
            The forgetting factor (higher means more weight to recent errors).
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        
        self.sum = 0.0
        self.sum_min = 0.0
        self.x_mean = 0.0
        self.n = 0
        # Initialize drift_detected attribute
        self.drift_detected = False
    
    def update(self, x):
        """
        Update the detector with a new error value.
        
        Parameters:
        -----------
        x : float
            The new error value to add.
            
        Returns:
        --------
        drift_detected : bool
            True if drift is detected, False otherwise.
        """
        self.n += 1
        
        # Update mean
        if self.n == 1:
            self.x_mean = x
        else:
            self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * x
        
        # Calculate deviation
        deviation = x - self.x_mean - self.lambda_
        
        # Update sum
        self.sum += deviation
        
        # Update minimum
        self.sum_min = min(self.sum_min, self.sum)
        
        # Check for drift
        ph = self.sum - self.sum_min
        
        # Set the drift_detected attribute
        self.drift_detected = (ph > self.delta)
        
        if self.drift_detected:
            # Reset the detector
            self.sum = 0.0
            self.sum_min = 0.0
            
        return self.drift_detected


##----------------------------------------------------------------------------------------------------------------------------------------------------##
##----------------------------------------------------------------------------------------------------------------------------------------------------##











class ForecastModel(ABC):

    def __init__(self, predictor: Type[Predictor], predictor_init_params = {}, predictor_params = {}, kseq = None, separate_outputs = False, estimate_variance = False):
        # TODO: consider adding method "add_input" isntead of specifying inputs in __init__.
        # TODO: add input argument, defaulting to all other columns than output
        super().__init__()
        self.input_data: DataFrame = None
        self._predictor_type = predictor
        self._predictor_init_params = predictor_init_params
        self.predictor_params = {}
        self._n_updates = 0
        for k, v in predictor_params.items():
            if k in predictor.params:
                self.predictor_params[k] = v
            else:
                raise KeyError(f"Predictor {Predictor} has no parameter: {k}.")
        self.predictors = None
        self.separate_outputs = separate_outputs
        self.estimate_variance = estimate_variance
        self.data_columns = None

        self.max_dependency_depth = 100
        self.reg_prms = {}
        self.prm_bounds = {}

        self.transform_comps: dict[TransformComponent] = {"raw_data_inputs": TransformComponent(Map(), regressor=True),
                                                  "raw_data_outputs": TransformComponent(Map(), regressand = True)}        
        
        self.kseq = kseq

    def transform_data(self, data: DataFrame, use_recursive_pars = False) -> DataFrame:
        
        transformed_data = {}

        transforms = self.get_transforms()

        for name, transform in transforms.items(): 


            # Apply transformation
            if use_recursive_pars and len(transform.recursion_pars) > 0:
                recursive_pars = self.transform_comps[name].recursion_pars
            else:
                recursive_pars = {}

            # Map transform inputs to data names
            str_vars = []
            transform_vars = []
            all_data_vars = []
            for v in transform.inst_inputs.values():
                if v is RawData:
                    all_data_vars.append(v)
                elif isinstance(v, Transformation):
                    transform_vars.append(v)
                elif isinstance(v, str):
                    str_vars.append(v)
                else:
                    raise ValueError(f'Transform {v} not recognised.')
 
            # Fetch data
            evaluate_data = data.fc.get_data(*str_vars)
            for t in transform_vars:
                evaluate_data[t] = transformed_data[t]
            for v in all_data_vars:
                evaluate_data[v] = data

            # Map data to evaluate_args
            evaluate_args = {}

            # Check inputs
            for new_name, var in transform.inst_inputs.items():
                if var in evaluate_data:
                    evaluate_args[new_name] = evaluate_data[var]
                else:
                    raise ValueError(f"Input {var} not found in data.")

            # Add recursion parameters
            evaluate_args.update(recursive_pars)

            # Include shape arguments if required
            if "n_t" in transform.evaluate_args:
                evaluate_args["n_t"] = data.fc.n_t

            # TODO: consider calling without data.fc.kseq (since data is now passed directly)
            new_data, rec_pars = transform.apply(kseq = data.fc.kseq, return_recursion_pars=True, evaluate_args = evaluate_args, data = data.copy())

            # Store transformed data and recursion parameters
            if use_recursive_pars:
                self.transform_comps[name].recursion_pars = rec_pars
   
            transformed_data[transform] = new_data
        
        # Rename columns in transformed data and ensure aligned index

        for name, t in transforms.items():
            if len(transformed_data[t].columns) > 0:
                if not t.preserve_names:
                    if len(transformed_data[t].columns) == 1:
                        transformed_data[t].columns = pd.MultiIndex.from_tuples([(f'{name}',k) for _, k in transformed_data[t].columns])
                    else:
                        transformed_data[t].columns = pd.MultiIndex.from_tuples([(f'{name}_{var}',k) for var,k in transformed_data[t].columns])
                
            transformed_data[t].index = data.index

        # Form design and response matrices
        dfs_X = [transformed_data[tc.transform] for tc in self.transform_comps.values() if tc.regressor]
        dfs_Y = [transformed_data[tc.transform] for tc in self.transform_comps.values() if tc.regressand]
        X = pd.concat(dfs_X, axis = 1)
        Y = pd.concat(dfs_Y, axis = 1)

        return X, Y

    def update(self, data, n_keep = None, return_Y = False, **predictor_params):
        
        if self.data_columns is None:
            self.data_columns = data.columns
        else:
            if not self.data_columns.equals(data.columns):
                raise ValueError("Columns do not match previous data.")

        n_new = len(data)

        # Check predictor parameters
        predictor_params = self.predictor_params.copy() | predictor_params
        for k in predictor_params:
            if not k in self._predictor_type.params:
                raise KeyError(f"Predictor {self._predictor_type.__name__} has no parameter: {k}.")

        # Prepare data
        X, Y = self.transform_data(data, use_recursive_pars=True)

        # Initialise predictors and predictions
        kseq = X.fc.kseq if self.kseq is None else self.kseq
        if self.predictors is None:
            outputs = Y.columns.get_level_values(0).unique().tolist() if self.separate_outputs else [None]
            self.predictors = {(o, k): None for o in outputs for k in kseq}
            self.predictions = {(o, k): ForecastTuple() for o in outputs for k in kseq}

        output_names = set(Y.columns.get_level_values(0))
        horizons = set(kseq)
        n_y = len(Y.columns)
        n_k = len(X.fc.kseq)
        k_max = max(kseq)
        if n_keep is None:
            n_keep = k_max

        # Append to stored data
        if self.input_data is None:
            self.input_data = X.copy()
        else:
            self.input_data = self.input_data.fc.append(X)

        # Construct lagged input data
#        X_fit = self.input_data.fc.get_lagged_subset(extend = False).iloc[-n_new:] # Last n_new rows of lagged input data
   
        # Prepare result container
        result = {}

        # Update fit for each predictor
        for key in self.predictors:
            
            # Subset data if required
            o, k = key
            kseq = ("NA", k)
            outputs = () if o is None else (o,)

            # TODO: implement check for whether data matches self.predictors (i.e. correct outputs and horizons)
            #x_fit = X_fit.fc.subset(kseq = kseq) # All regressors, horizon k - lagged
#            x_fit = self.input_data.fc.get_lagged_subset(kseq = kseq, extend = False).iloc[-n_new:] # All regressors, horizon k - lagged
            x_fit = self.input_data.fc.subset(kseq = kseq).shift(k).iloc[-n_new:] # All regressors, horizon k and observations, lagged k

            y_fit = Y.fc.subset(*outputs, kseq=kseq) # Selected outputs (horizons usually irrelevant) - newest
            x_pred = X.fc.subset(kseq = kseq) # All regressors, horizon k - newest
            # prev_preds = self.predictions[key]

            # Initialise regressor if not done
            if self.predictors[key] is None:
                self.predictors[key] = self._predictor_type(x_columns = x_fit.columns, y_columns = y_fit.columns, **self._predictor_init_params)

            # Update fit with subset of data
            params = {k: v for k, v in predictor_params.items() if k in self.predictors[key].params}
            prediction = self.predictors[key].update_fit(x_fit, y_fit, x_pred, **params)

            # Check prediction
            if isinstance(prediction, pd.DataFrame):
                prediction = ForecastTuple(prediction)
            elif isinstance(prediction, tuple):
                prediction = ForecastTuple(*prediction)

            for i, df in enumerate(prediction): 

                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Prediction output must be a DataFrame.")

                n_df = len(df.columns)

                if i == 0: # Checks only relevant for first dataframe, i.e. output predictions. TODO: consider simplyfying or removing this.

                    # Dataframe provided by predictor specific to output and horizon
                    if k is not None and o is not None:
                        if n_df != 1:
                            raise ValueError(f"Expected single column in prediction, got {n_df}.")
                        df.columns = pd.MultiIndex.from_tuples([(o, k)], names = ["Variable", "Horizon"])

                    # Dataframe provided by predictor specific to horizon
                    elif k is not None:
                        if n_df != n_y:
                            raise ValueError(f"Expected {n_y} columns in prediction, got {n_df}.")
                        
                        if not set(df.columns.get_level_values(0)) == output_names:
                            raise ValueError("Output names do not match. Check prediction output.")

                        df.columns = pd.MultiIndex.from_tuples([(col[0], k) for col in df.columns], names = ["Variable", "Horizon"])

                    # Dataframe provided by predictor specific to output
                    elif o is not None:
                        if n_df != n_k:
                            raise ValueError(f"Expected {n_k} columns in prediction, got {n_df}.")
                        
                        if not set(df.columns.get_level_values(1)) == horizons:
                            raise ValueError("Horizon names do not match. Check prediction output.")

                        df.columns = pd.MultiIndex.from_tuples([(o, col[1]) for col in df.columns], names = ["Variable", "Horizon"])

                    if not df.fc.check():
                        df.columns = y_fit.columns

            # Append predictions    
            result[key] = prediction.copy()

            self.predictions[key] = self.predictions[key].append(prediction)

            # Make and include variance estimates if required
            # NOTE: this may not be generally applicable (i.e. might not work for temporal hierarchies?). TODO: consider updating or moving into predictors. Could also be moved into a wrapper outside the update method.
            if self.estimate_variance:

                # Get residuals
                l_pred = self.predictions[key][0].fc.get_lagged_subset(extend = False)[-n_new:] # Assume output predictions are in first dataframe
                res = get_residuals(l_pred, y_fit, apply_lag = False)
                
                # Update variance estimates
                result[key] += (call_with_kwargs(self.predictors[key].update_var_est, residuals = res, **predictor_params),)

        # Store only necessary data
        self.input_data.fc.remove_old_data(n_keep)

        # Update prediction storage
        for key in self.predictions:
            self.predictions[key].remove_old_data(n_keep)

        # Update result

        # Combine prediction dataframes accross horizons
        outputs = set([key[0] for key in result.keys()])
        for o in outputs:
            keys = [key for key in result.keys() if key[0] == o]

            # Remove predictions for all horizons for specific output
            o_pred = [result.pop(key) for key in keys]

            # Replace with joined predictions
            result[(o, None)] = concat_forecast_tuples(*o_pred)

        # Combine prediction dataframes accross outputs if provided by separate predictors
        if self.separate_outputs:
            horizons = set([key[1] for key in result.keys()])
            for k in horizons:
                keys = [key for key in result.keys() if key[1] == k]

                # Remove predictions for all outputs for specific horizon
                k_pred = [result.pop(key) for key in keys]

                # Replace with joined predictions
                result[(None, k)] = concat_forecast_tuples(*k_pred)

        # Simplify result structure and ensure formats
        for key, pred in result.items():
            if len(pred) == 1:
                result[key] = pred[0].iloc[-n_new:]
            else:
                result[key] = ForecastTuple(*[df.iloc[-n_new:] for df in pred])

        self._n_updates += 1

        # Return fit results
        if len(result) == 1:
            result = next(iter(result.values()))

        if return_Y:
            result = result, Y

        return result

    def reset_state(self):
        # Rest recursion parameters for inputs
        for tc in self.transform_comps.values():
            tc.reset()

        # Reset transformed data
        self.input_data = None
        self.predictors = None
        self.predictions = None
        self.data_columns = None
        self._n_updates = 0

    def save_model(self, name: str = None):
        if name is None:
            return pickle.dumps(self)
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, data: DataFrame, n_keep = None, join_horizons = True, join_outputs = True, **predictor_params):
        self.reset_state()
        return self.update(data, n_keep = n_keep, **predictor_params)

    def prepare_data(self, data: DataFrame) -> tuple[DataFrame, ...]:
        # Catch output data
        outputs = [self.output] if isinstance(self.output, str) else self.output
        Y = data.fc.subset(*outputs)

        # Transform data
        X = self.transform_data(data, use_recursive_pars=True)

        return X, Y

    def get_params(self):
        t_params = {name: tc.params for name, tc in self.transform_comps.items() if len(tc.params) > 0}
        return {"predictor": self.predictor_params.copy()} | t_params

    def update_params(self, target = None, **params):

        if target is None:
            for k, v in params.items():
                found = False
                if k in self._predictor_type.params:
                    found = True
                    self.predictor_params[k] = v
                else:
                    for tc in self.transform_comps.values():
                        if k in tc.params:
                            found = True
                            tc.update_params(**{k: v})

                if not found:
                    raise KeyError(f"Parameter {k} not found.")

        elif target in self.transform_comps:
            self.transform_comps[target].update_params(**params)
        elif target == "predictor":
            for k, v in params.items():
                if k in self.predictor_params:
                    self.predictor_params[k] = v
                else:
                    raise KeyError(f"Parameter {k} not found.")
        else:
            raise KeyError(f"Target {target} not found.")

    def add_inputs(self,*inputs, **new_inputs):
        for input in inputs:
            if not isinstance(input, str):
                raise ValueError("Input must be a string reffering to a column name or added as a new input.")
            if not input in self.transform_comps["raw_data_inputs"].transform.vars:
                self.transform_comps["raw_data_inputs"].transform.vars.append(input)
                
        to_add = {}
        for name, val in new_inputs.items():
            if name in self.transform_comps:
                self.transform_comps[name].regressor = True
                if self.transform_comps[name].regressand:
                    warn(f"Transform {name} already included as output.")
            else:
                to_add[name] = val
        
        self.add_transforms(**to_add, regressor = True)

    def add_outputs(self, *outputs, **new_outputs):
        for output in outputs:
            if not isinstance(output, str):
                raise ValueError("Output must be a string reffering to a column name or added as a new output.")
            if not output in self.transform_comps["raw_data_outputs"].transform.vars:
                self.transform_comps["raw_data_outputs"].transform.vars.append(output)

        to_add = {}
        for name, val in new_outputs.items():
            if name in self.transform_comps:
                self.transform_comps[name].regressand = True

                if self.transform_comps[name].regressor:
                    warn(f"Transform {name} already included as input.")
            else:
                to_add[name] = val

        self.add_transforms(**to_add, regressand = True)

    def add_transforms(self, regressor = False, regressand = False, **transforms):
        for name, val in transforms.items():
            if isinstance(val, str):
                transform = eval(val, {"__builtins__": None},{cls.__name__: cls for cls in Transformation.__subclasses__()} | {"RawData": RawData})
            else:
                transform = val
            if transform in self.transform_comps:
                raise ValueError(f"Transform {transform} already added.")
            
            self.transform_comps[name] = TransformComponent(transform, regressor = regressor, regressand = regressand)


    def get_transforms(self):

        # Get all transforms
        current_transforms = [tc.transform for tc in self.transform_comps.values()]
        check_list = current_transforms
        deps_to_include = set()

        i = 0
        while len(check_list) > 0:
            deps = []
            for t in check_list:
                deps.extend(t.dependencies)
            check_list = [t for t in deps if t not in current_transforms]
            deps_to_include.update(deps)
            i += 1
            if i > self.max_dependency_depth:
                raise ValueError("Dependency depth exceeded. Check for circular dependencies or increase max_dependency_depth.")

        # Remove dependencies not required
        names = list(self.transform_comps.keys())
        for name in names:
            tc = self.transform_comps[name]
            t = tc.transform

            if not (tc.regressor or tc.regressand) and t not in deps_to_include:
                del self.transform_comps[name]

        # Add missing dependencies
        self.add_transforms(**{str(id(t)): t for t in deps_to_include if t not in current_transforms})

        # Sort transforms
        to_sort = [tc.transform for tc in self.transform_comps.values()]
        sorted_transforms = []
        i = 0
        while len(to_sort) > 0:
            for t in to_sort:
                deps = t.dependencies
                if all(d in sorted_transforms for d in deps):
                    sorted_transforms.append(t)
                    to_sort.remove(t)
            i += 1
            if i > self.max_dependency_depth:
                raise ValueError("Dependency depth exceeded. Check for circular dependencies or increase max_dependency_depth.")

        # sort transforms in order of dependency
        inv_transforms = {v.transform: k for k, v in self.transform_comps.items()}

        # Return dict of transforms in dependency order
        result = {inv_transforms[t]: t for t in sorted_transforms}
        return result
        
    def set_regprm(self, *params, target = None, **bounded_params):

        all_params = {k: (None, None) for k in params} | bounded_params

        # Extract inputs for which to set regression parameters
        if target is None:
            targets = ["predictor"] + list(self.transform_comps.keys())
        elif target == self or target == "predictor":
            targets = ["predictor"]
        elif target in self.transform_comps:
            targets = [target]

        model_params = self.get_params()

        # Iterate inputs
        for name, bounds in all_params.items():

            name_found = False

            # Iterate data containers (general model parameters, or specific input parameters)
            for key in targets:

                if name in model_params[key].keys(): # TODO: more checks for shape of bounds, and more checks that it is not a variable 
                    name_found = True

                    # Setup reg_prms if not present
                    if not key in self.reg_prms:
                        self.reg_prms[key] = {}

                    if isinstance(bounds, dict):
                        self.reg_prms[key][name] = (bounds["lower"], bounds["upper"])
                    elif isinstance(bounds, (tuple, list)):
                        self.reg_prms[key][name] = tuple(bounds)
                    else:
                        raise ValueError("Bounds input not recognised.")                    

            if not name_found:
                 raise ValueError(f'Regression parameter {name} not valid.')

    def remove_regprm(self, *params, transform = None):
        if transform is None:
            inputs = list(self.reg_prms.keys()).copy()
        elif transform == self | transform == "predictor":
            inputs = ["predictor"]
        elif transform in self.transform_comps:
            inputs = [transform]

        # Iterate data containers (general model parameters, or specific input parameters)
        for key in inputs:
            for p in params:
                if p in self.reg_prms[key]:
                    del self.reg_prms[key][p]
            if len(self.reg_prms[key]) == 0:
                del self.reg_prms[key]

    def _get_flat_regprms(self):
        """
        Method to retrieve regression parameters in specific format required by rls_optim.
        """
        values = []
        bounds = []
        order = {}

        order_index = 0
        model_params = self.get_params()
        for target, reg_prms in self.reg_prms.items():

            order[target] = {}

            for prm, bound in reg_prms.items():

                # Fetch param value
                p_v = model_params[target][prm]
                if isinstance(p_v, np.ndarray):
                    n_p = len(p_v)
                    # Add values
                    values.extend(p_v)
                else:
                    n_p = 1
                    values.append(p_v)

                # Add bounds
                if isinstance(bound[0], np.ndarray) and isinstance(bound[1], np.ndarray) and len(bound[0]) == len(bound[1]):
                    bounds.extend(zip(bound[0].flatten(), bound[1].flatten()))   
                else:
                    bounds.append(bound)

                # Store indices
                order[target][prm] = (order_index, order_index + n_p)

                # Increment index
                order_index += n_p

        return values, bounds, order


    def _set_flat_regprms(self, prm_values, order):
        for target, pars in order.items():
            params = {par: prm_values[slice(*indices)] for par, indices in pars.items()}
            self.update_params(target = target, **params)


    def optim_fit_wrapper(self, data, par_order, scorefun, burn_in):

        def wrapper(params):
            # Distribute params
            self._set_flat_regprms(params, par_order)
            predictions, score_data = self.fit(data, return_Y = True)
        
            residuals = get_residuals(predictions.iloc[burn_in:], score_data.iloc[burn_in:])

            score = 0
            for col in residuals:
                score += scorefun(residuals[col])
            return score

        return wrapper

    def optim(self, data, scorefun, set_params = False, store_intermediate = False, method = 'Nelder-Mead', burn_in = None, predictor_params = {}, **optim_params):

        if len(self.reg_prms) == 0:
            raise ValueError("No regression parameters set.")

        # Create test model
        m = copy.deepcopy(self)  
        m.estimate_variance = False      

        m.predictor_params.update(predictor_params)

        # Retrieve parameter bounds
        x0, bounds, order = m._get_flat_regprms()
        
        if burn_in is None:
            burn_in = 0

        fun = m.optim_fit_wrapper(data, order, scorefun, burn_in)

        # Store intermediate function values
        callback = None

        if store_intermediate:
            intermediate_values = {"params": [], "score": []}
            def callback(xk):
                intermediate_values["params"].append(xk)
                intermediate_values["score"].append(fun(xk))

        x_min = minimize(fun, x0, method=method, bounds=bounds, callback=callback, **optim_params)

        m._set_flat_regprms(x_min.x, order)

        result = {input: {p: v for p, v in m.get_params()[input].items() if p in m.reg_prms[input]} for input in m.reg_prms}

        if set_params:
            self._set_flat_regprms(x_min.x, order)

        return result, x_min, intermediate_values


    
    def optim_pso(self, data, scorefun, set_params=False, store_intermediate=False, n_particles=35, max_iter=30, burn_in=None, predictor_params={}, **optim_params):
        """
        Optimize model parameters using Particle Swarm Optimization (PSO) via PySwarms.
    
        Args:
            data (DataFrame): Training data conforming to the forecast matrix structure.
            scorefun (callable): Function to evaluate model performance (e.g., rmse).
                                  Takes a pandas Series of residuals as input and returns a scalar score.
            set_params (bool, optional): Whether to update the original model instance
                                          with the best parameters found. Defaults to False.
            store_intermediate (bool, optional): Whether to store intermediate results.
                                                  Note: PySwarms history might need specific handling. Defaults to False.
            n_particles (int, optional): Number of particles in the swarm. Defaults to 35.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to 30.
            burn_in (int, optional): Number of initial samples to skip when calculating the score
                                     on the training data. If None, uses 0.
            predictor_params (dict, optional): Additional fixed parameters to pass to the predictor's
                                               update_fit method during optimization and the final fit.
                                               These should NOT be parameters that are being optimized. Defaults to {}.
            **optim_params: Additional keyword arguments passed directly to the PySwarms
                            GlobalBestPSO optimizer's options dictionary (e.g., c1, c2, w).
    
        Returns:
            tuple: A tuple containing:
                - result (dict): Dictionary of optimized parameter values, structured by target ('predictor' or transform name).
                - optim_result (dict): Dictionary containing the final cost ('cost') and best position ('position') from PSO.
                - intermediate_values (dict or None): Dictionary storing intermediate optimizer history if
                                                      store_intermediate is True and history is accessible. Otherwise, None.
                - svr_states (dict): Dictionary mapping predictor keys to their internal states,
                                     extracted *after* a final fit with optimal parameters (only for OnlineSVR predictors).
        """
        if len(self.reg_prms) == 0:
            raise ValueError("No regression parameters set. Use set_regprm() before calling optim_pso.")
    
        # Create a deep copy of the model to avoid modifying the original during optimization runs
        m = copy.deepcopy(self)
        m.estimate_variance = False # Variance estimation not needed for optimization score
    
        # Update the test model's fixed predictor parameters (those NOT being optimized initially)
        # These are parameters passed in predictor_params that might not be part of the optimization itself
        m.predictor_params.update(predictor_params)
    
        # Retrieve initial parameter values, bounds, and structure for optimization
        x0, bounds_list, order = m._get_flat_regprms()
        n_dims = len(x0)
    
        if burn_in is None:
            burn_in = 0
    
        # Identify keys of parameters being optimized specifically within the 'predictor' target
        optimized_predictor_keys = set()
        if "predictor" in m.reg_prms:
            optimized_predictor_keys = set(m.reg_prms["predictor"].keys())
    
        # --- Define the objective function for PySwarms ---
        def objective_func_vectorized(params_array):
            n_particles_current = params_array.shape[0]
            scores = np.zeros(n_particles_current)
    
            for i in range(n_particles_current):
                particle_params = params_array[i]
    
                # 1. Set the model's parameters for the current particle
                try:
                    m._set_flat_regprms(particle_params, order)
                except Exception as e:
                    # print(f"Warning: Error setting parameters for particle {i}: {e}") # Debug
                    scores[i] = 1e18 # Assign high penalty
                    continue
    
                # 2. Prepare predictor_params for the fit call, excluding optimized predictor keys
                current_run_predictor_params = {
                    k: v for k, v in predictor_params.items()
                    if k not in optimized_predictor_keys # Exclude optimized keys if they were also passed in predictor_params
                }
                # Add any fixed params already in the model that weren't passed in predictor_params and aren't optimized
                for k_model, v_model in m.predictor_params.items():
                     if k_model not in optimized_predictor_keys and k_model not in current_run_predictor_params:
                           current_run_predictor_params[k_model] = v_model
    
                # 3. Run the model fit/update process (needs reset for each particle eval)
                try:
                    # Use fit (which calls reset_state) to ensure clean state for each particle evaluation
                    predictions, score_data = m.fit(data.copy(), return_Y=True, **current_run_predictor_params)
    
                    # 4. Calculate residuals
                    if predictions is None or predictions.empty:
                        scores[i] = 1e18
                        continue
                    if len(predictions) != len(score_data):
                        scores[i] = 1e18
                        continue
    
                    residuals = get_residuals(predictions.iloc[burn_in:], score_data.iloc[burn_in:])
    
                    # 5. Calculate the score - MODIFIED to handle NaNs like optim() does
                    try:
                        particle_score = 0
                        for col in residuals.columns:
                            # Pass residuals directly to scorefun without filtering NaNs
                            # The scorefun (e.g., rmse) will handle NaNs internally
                            particle_score += scorefun(residuals[col])
                        
                        if not np.isfinite(particle_score):
                            particle_score = 1e18
                            
                    except Exception as e:
                        particle_score = 1e18
                    
                    scores[i] = particle_score
    
                except Exception as e:
                    # print(f"Warning: Error during fit/scoring for particle {i}: {e}") # Debug
                    scores[i] = 1e18 # Assign high penalty
    
            return scores
        # --- End of objective function definition ---
    
        # Convert bounds for PySwarms
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds_list])
        ub = np.array([b[1] if b[1] is not None else np.inf for b in bounds_list])
        pso_bounds = (lb, ub)
    
        # PSO options
        pso_options = {'c1': 0.6, 'c2': 0.4, 'w': 0.9}
        pso_options.update(optim_params) # Update with user-provided options
    
        # Initialize PSO
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                            dimensions=n_dims,
                                            options=pso_options,
                                            bounds=pso_bounds,
                                            init_pos=np.array(x0).reshape(1, -1) if n_particles==1 else None)
    
        # --- Run the optimization ---
        cost, pos = optimizer.optimize(objective_func_vectorized, iters=max_iter, verbose=False)
    
        # Intermediate values (optional, based on optimizer history)
        intermediate_values = None
        if store_intermediate:
            print("Warning: Storing intermediate values with default PySwarms optimize loop requires accessing optimizer history attributes post-run (e.g., optimizer.cost_history), if available and enabled.")
            try:
                intermediate_values = {"score": optimizer.cost_history}
                # if hasattr(optimizer, 'pos_history'): intermediate_values["params"] = optimizer.pos_history
            except AttributeError:
                print("Could not retrieve intermediate history from optimizer attributes.")
                intermediate_values = None
    
        # --- Post-Optimization Steps ---
        # 1. Set the optimal parameters found on the copied model 'm'
        m._set_flat_regprms(pos, order)
    
        # 2. --- ADDED: Perform a final fit on 'm' with optimal parameters ---
        # This ensures the predictor's internal state reflects the optimal parameters applied to the entire optimization dataset.
        print("Performing final fit on copied model with optimal PSO parameters...")
        final_fit_params = {
            k: v for k, v in predictor_params.items()
            if k not in optimized_predictor_keys # Exclude optimized predictor keys from kwargs
        }
        # Add any fixed params already in the model that weren't passed in predictor_params and aren't optimized
        for k_model, v_model in m.predictor_params.items():
             if k_model not in optimized_predictor_keys and k_model not in final_fit_params:
                   final_fit_params[k_model] = v_model
        try:
            # Use fit (which resets state) on the entire training data with final params
            m.fit(data, **final_fit_params)
        except Exception as e:
            warn(f"Error during final fit after PSO optimization: {e}. Predictor state might not be fully updated.")
    
        # 3. --- ADDED: Extract internal state ONLY from OnlineSVR predictors ---
        svr_states = {}
        if m.predictors: # Check if predictors were initialized by the final fit
            for key, predictor_instance in m.predictors.items():
                if isinstance(predictor_instance, OnlineSVR):
                    print(f"Extracting state from OnlineSVR instance with key: {key}")
                    try:
                        state = {
                            'X_internal': copy.deepcopy(predictor_instance.X_internal),
                            'y_internal': copy.deepcopy(predictor_instance.y_internal),
                            'alphas': copy.deepcopy(predictor_instance.alphas),
                            'alphas_star': copy.deepcopy(predictor_instance.alphas_star),
                            'bias': predictor_instance.bias,
                            'S': copy.deepcopy(predictor_instance.S),
                            'E': copy.deepcopy(predictor_instance.E),
                            'R': copy.deepcopy(predictor_instance.R),
                            'R_inv': copy.deepcopy(predictor_instance.R_inv),
                            'Q': copy.deepcopy(predictor_instance.Q),
                            '_feature_dim': predictor_instance._feature_dim,
                            '_initialized': predictor_instance._initialized
                        }
                        svr_states[key] = state
                    except Exception as e:
                        warn(f"Could not extract state from OnlineSVR instance {key}: {e}")
    
        # 4. Extract the optimized parameters in the original structure
        result = {}
        best_params_structured = m.get_params()
        for target, reg_prms_dict in m.reg_prms.items():
            if target not in result: result[target] = {}
            for p_name in reg_prms_dict.keys():
                if target in best_params_structured and p_name in best_params_structured[target]:
                    result[target][p_name] = best_params_structured[target][p_name]
    
        # 5. Update the original model instance if requested
        if set_params:
            self._set_flat_regprms(pos, order)
            # Ensure the original model's fixed predictor params are also consistent
            fixed_params_to_set = {k: v for k, v in predictor_params.items() if k not in self.predictor_params}
            self.predictor_params.update(fixed_params_to_set)
    
        # 6. Return results including SVR states
        return result, {"cost": cost, "position": pos}, intermediate_values, svr_states # Added svr_states


"""
    def get_weights(self):
        names = self.output if isinstance(self.output, list) else [self.output]
        rows = pd.MultiIndex.from_tuples([(name, k) for name in names for k in self.kseq])
        result = pd.DataFrame(columns = list(self.transformed_data.fc.variables), index = rows)

        for k, fit in self.LFits.items():
            if fit is not None:
                for i, name in enumerate(names):
                    result.loc[(name, k)] = fit.theta[:,i]
        return result

    @property
    def LFits(self):
        return {key[1]: fit for key, fit in self.predictors.items()}
"""







"""
#%%
data = sample_data.fc.subset(kseq = (1,2,3))
result = new_fc()
#%%
new_data = data.fc["Ta"]
new_cols = pd.MultiIndex.from_product([["Ta"], new_data.columns], names = result.columns.names)
new_data.columns = new_cols
result = result.join(new_data, how = "right")

#%%
result = new_fc()
mew_data = data.fc


#%%
# Create a new MultiIndex for df_B columns
new_columns = pd.MultiIndex.from_product([['B'], df_B.columns])

# Assign the new MultiIndex to df_B columns
df_B.columns = new_columns

# Concatenate along the columns
result = pd.concat([df_A, df_B], axis=1)

print(result)
#%%
# Create a new MultiIndex for df_B columns
new_columns = pd.MultiIndex.from_product([['B'], df_B.columns])

# Assign the new MultiIndex to df_B
df_B.columns = new_columns

# Concatenate along the columns
result = pd.concat([df_A, df_B], axis=1)

print(result)
#%%
# Assign the new MultiIndex to df_B
df_B.columns = new_columns

# Concatenate along the columns
result = pd.concat([df_A, df_B], axis=1)

print(result)

"""

def load_model(file_name):
    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def rmse(x):
    return np.sqrt(np.mean(x**2))


def aligned_pairs(predictions, observations, apply_lag = True, extend = False):
    if apply_lag:
        lagged_subset = predictions.fc.get_lagged_subset(extend = extend)
    else:
        lagged_subset = predictions
    if isinstance(observations, pd.Series):
        variables = [observations.name]
        get_obs = lambda name: observations
    else:
        variables = observations.columns.get_level_values(0).unique()
        get_obs = lambda name: observations.fc[name]

    for var in variables:
        yield get_obs(var), lagged_subset.fc[var]


def get_residuals(predictions, observations, apply_lag = True, extend = False):
    residuals = pd.DataFrame(columns = predictions.columns)
    for obs, pred in aligned_pairs(predictions, observations, apply_lag = apply_lag, extend = extend):
        for col in pred:
            residuals[(obs.name), col] = obs - pred[col]
    return residuals

def evaluate_result(predictions: DataFrame, observations: DataFrame, scorefun = rmse):
    aligned_predictions = predictions.fc.get_lagged_subset()
    score = 0
    kseq = predictions.fc.kseq
    for name in observations.fc.variables:
        for k in kseq:
            col = (name, k)
            col_hat = (name + "_hat", k)
            if col_hat in aligned_predictions:
                residuals = observations[col] - aligned_predictions[col_hat]
                residuals = residuals[~pd.isna(residuals)]
                score += scorefun(residuals)
    return score

#%%
def lag_and_extend(predictions, var_est = None):
    lagged_predictions = predictions.fc.get_lagged_subset(extend=True)
    if var_est is not None:
        n_extend = len(lagged_predictions) - len(predictions)
        var_est = var_est.fc.append(pd.DataFrame(np.tile(var_est.iloc[-1].values, (n_extend, 1)), columns=var_est.columns))
        return lagged_predictions, var_est
    return lagged_predictions


def get_normal_confidence_interval(lagged_predictions: DataFrame, var_est: DataFrame, alpha = 0.05):
    colummns = pd.MultiIndex.from_tuples([c + (interval,) for c in lagged_predictions.columns for interval in ["pred", "lo", "hi"]])
    result = pd.DataFrame(index=lagged_predictions.index, columns=colummns)
    z = norm.ppf(1 - alpha / 2)
    for col in lagged_predictions:
        s = np.sqrt(var_est[col])
        z_s = z*s
        result[col + ("lo",)] = lagged_predictions[col] - z_s
        result[col + ("hi",)] = lagged_predictions[col] + z_s
    return result


# %%
