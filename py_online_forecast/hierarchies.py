#%%
import numpy as np
from . import core as c
#import core as c
import pandas as pd
import copy

def minT(S, Y: c.DataFrame, Y_hat: c.DataFrame, l_shrink = 0):
    # y: base level observations
    # y_hat: base forecasts at all levels
    cols = Y_hat.columns
    y = Y.to_numpy()
    y_hat = Y_hat.to_numpy()
    tmp = y @ S.T - y_hat
    cov = 1/len(y) * tmp.T @ tmp
    cov_shrink = (1-l_shrink)*cov + l_shrink*np.diag(np.diag(cov)) # Shrink towards diagonal
    W = np.linalg.inv(cov_shrink)
    StW = S.T @ W
    P = np.linalg.inv(StW @ S) @ StW
    res = (S @ P @ y_hat.T).T
    return P, c.new_fc(res, columns = cols, index = Y_hat.index)

def construct_temporal_hierarchy(shape: list[tuple[int, int]]) -> np.ndarray:
    level_matrices = [np.kron(np.eye(q), np.ones(k)) for k, q in shape]    
    return np.vstack(level_matrices)

def hierarchy_shape(*h, as_groups = False):
    """
    Returns a list of tuples giving a hierarchy shape [(k_n, q_n), ..., (k_i, q_i), ..., (k_0, q_0)], where k_i is the number of 
    base forecasts required for level i, and q_i is the number of entries of level i in the hierarchy. 
    Here i >= 0 and i = 0 is the bottom level so that k_i > k_j for i < j.

    h: tuple of integers. If as_groups is False, h represents k_i. if as_groups
    is True, h represents the sequence of groupings, i.e. h_i = k_{n-(i+1)}/k_{n-i} where k_n = 1.
    """
    if as_groups:
        levels = [1]
        for g in h:
            if g == 1:
                raise ValueError("Grouping should be greater than 1")
            levels.insert(0, levels[0] * g)
    else:
        levels = h
        if not levels[-1] == 1:
            levels = levels + (1,)

    m = levels[0]
    lens = []
    for k in levels:
        quotient, remainder = divmod(m, k)
        if remainder != 0:
            raise ValueError(f"level {k} is not a factor of {m}")
        lens.append((k, quotient))
    return lens

def form_glm(Y, Y_hat, S_top):
    
    nsm, m = S_top.shape

    # Top level forecasts are the first n-m columns of Y_hat
    Y_hat_top = Y_hat[:, :nsm]

    # Bottom level forecasts are the last m columns of Y_hat
    Y_hat_bot = Y_hat[:, -m:]

    # Top level forecasts minus summed bottom level forecasts (reconciliation error)
    X_out = Y_hat_top - (Y_hat_bot @ S_top.T)

    # Observations minus bottom level forecasts (base forecast errors)
    Y_out = Y - Y_hat_bot

    return X_out, Y_out


class HierarchyTransform(c.Transformation):

    # Transformation to map hierarchy data to GLM. The output follows the matrix GLM formulation, 
    # i.e. Y = X^T B + E, where Y are the base forecast errors, X the reconciliation error, and E is the base forecast errors.
    # X is shape (t, n-m), Y is shape (t, m), and B is shape (n-m, m).
    # The input data assumes first m columns are bottom level observations, and the remaining n columns are base forecasts at all levels,
    # arranged in decreasing order of hierachy level. Forecast horizons should be the same for all levels.

    params = {}
    recursion_pars = ["Y_hat_bot_old"]

    def __init__(self, S_top, data = c.RawData):
        super().__init__(data = data)
        self.S_top = S_top

    def evaluate(self, data, Y_hat_bot_old = None):

        nsm, m = self.S_top.shape

        Y = data.iloc[:,:m] # Assumes Y is first m columns
        Y_hat = data.iloc[:,m:] # Assumes Y_hat is remaining n columns

        # Ensure that all forecasts are aligned on the same horizon
        kseq = Y_hat.fc.kseq
        if len(kseq) > 1:
            raise ValueError("Multiple forecast horizons in bottom level forecasts detected in HierachyTransform")

        k_max = max(kseq)

        # Top level forecasts are the first n-m columns of Y_hat
        Y_hat_top = Y_hat.iloc[:, :nsm]

        # Bottom level forecasts are the last m columns of Y_hat
        Y_hat_bot = Y_hat.iloc[:, -m:]

        # Top level forecasts minus summed bottom level forecasts (reconciliation error)
        X_out = Y_hat_top - (Y_hat_bot @ self.S_top.T).values

        if Y_hat_bot_old is None:
            Y_hat_bot_old = np.full((k_max, m), np.nan)

        Y_hat_bot_fit = np.vstack((Y_hat_bot_old, Y_hat_bot.values))

        # Observations minus bottom level forecasts (base forecast errors)
        Y_out = Y - Y_hat_bot_fit[:-k_max]
    
        # Store data in a dataframe containing 1) base forecast errors (first m columns), 2) top level reconciliation errors (next n-m columns), and 3) base forecasts (last n columns)
        result = pd.concat([Y_out, X_out, Y_hat], axis = 1)

        return result, {"Y_hat_bot_old": Y_hat_bot_fit[-k_max:]}

class TemporalHierarchyTransform(c.Transformation):

    # Transform to map simple temporal hiearchy data to reconcilable format
    # The input data should conform to the "standard" simple temporal hierarchy format.

    recursion_pars = ["Y_old"]

    def __init__(self, S_top, data = c.RawData):
        super().__init__(data = data)
        self.S_top = S_top

    def evaluate(self, data, Y_old = None):

        nsm, m = self.S_top.shape
        n = m + nsm

        # TODO: implement check that all columns have same horizon (this should be true for hierarchies)
        k_max = max(data.fc.kseq)

        # First m columns are observations, next n columns are base forecasts at all levels
        Y_hat = data.iloc[:, :n]
        Y = data.iloc[:, -1]

        Y_hat_bot_cols = Y_hat.columns[-m:]

        t = len(Y)

        if Y_old is None:
            Y_old = np.full(k_max, np.nan)

        Y_all = np.hstack((Y_old, Y))

        Y_lagged = np.zeros((t, m))

        for i, k in enumerate(Y_hat_bot_cols.get_level_values(1)):
            end_index = -(k_max - k) if k != k_max else None
            Y_lagged[:, i] = Y_all[-(t + k_max - k):end_index]

        #X_out, Y_out = form_glm(Y_lagged, Y_hat, self.S_top)

        # Store in regular format with Y first m columns and X next n-m columns, column names as integers with kmax as common horizon
#        new_cols = list(Y_hat_bot_cols) + [(col[0], k_max) for col in Y_hat.columns]
        new_cols = [(col[0], "NA") for col in Y_hat_bot_cols] + [(col[0], k_max) for col in Y_hat.columns]
        columns = pd.MultiIndex.from_tuples(new_cols)
        #result = c.new_fc(np.hstack((Y_out, X_out)), index = raw_data.index, columns = columns)

        return c.new_fc(np.hstack((Y_lagged, Y_hat)), index = data.index, columns = columns), {"Y_old": Y_all[-k_max:]}

class ShrinkagePredictor(c.Predictor): # Tikhonov regularised least squares with moving window. Uses hyperparameters corresponding to shrinkage.

    # TODO: consider making a hierarchy predictor (or including in wrapper) and including variance stimates using base_err_top = X - Y @ S_top.T, and all errors = [base_err_top, Y]

    def __init__(self, S_top, x_columns = None, y_columns = None, skip_k = False, estimate_variance = False):
        super().__init__(x_columns, y_columns)
        self.data = c.ForecastTuple()
        self.sigma_bot_hat = None
        self.sigma_top_hat = None
        self.beta = None
        self.skip_k = skip_k # If True, window size is adjusted so that only every k'th row is used for fitting. This is useful for temporal hierarchies to avoid reusing the same data.
        self.S_top = S_top
        self.estimate_variance = estimate_variance

    @property
    def P(self):
        nsm, m = self.S_top.shape
        n = m + nsm
        null = np.zeros_like((n-m, n-m))
        return (np.vstack((null, np.eye(m))) + np.vstack((np.eye(n-m), -self.S_top.T)) @ self.beta).T

    def shrinkage_update(self, X, Y, l_shrink = 0):

        S_top = self.S_top

        mask = ~(X.isna().any(axis=1) | Y.isna().any(axis=1))
        X = X[mask].to_numpy()
        Y = Y[mask].to_numpy()

        t = len(X)
        
        base_err_top = X - Y @ S_top.T # Assumes X = Y_hat_top - Y_hat_bot_S_top, Y = Y_obs - Y_hat_bot with sign

        if self.estimate_variance:
            # Get full error matrix
            base_err = np.hstack((base_err_top, Y))
            cov_base_err = base_err.T @ base_err/t # np.cov(base_err.T)
            cov_base_err_d = np.diag(cov_base_err)
            cov_top_hat_d = np.diag(cov_base_err_d[:base_err_top.shape[1]])
            cov_bot_hat_d = np.diag(cov_base_err_d[base_err_top.shape[1]:])
        else:
            # Estimate of covariance diagonal (instead of full covariance matrix)
            cov_bot_hat_d = 1/t*np.diag(np.sum(Y*Y, 0))
            cov_top_hat_d = 1/t*np.diag(np.sum(base_err_top*base_err_top, 0))

        # Prepare for Tikhonov regularised least squares
        P = np.eye(t)
        b0 = cov_top_hat_d + S_top @ cov_bot_hat_d @ S_top.T
        b1 = S_top @ cov_bot_hat_d
        Q = l_shrink*t/(1-l_shrink)*b0
        B0 = np.linalg.solve(b0, b1)

        # Use Tikhonov regularised least squares to estimate beta
        self.beta = tikhonov(X, Y, P, Q, B0)


        # TODO: investigate if parameter variance estimate can be computed generically in the tikhnov_rls function or elsewhere
        if self.estimate_variance:
            
            # Compute projection matrix
            nsm, m = S_top.shape
            n = m + nsm
            null = np.zeros_like((n-m, n-m))
            P = (np.vstack((null, np.eye(m))) + np.vstack((np.eye(n-m), -S_top.T)) @ self.beta).T

            # Compute Shrinkage estimator of Sigma_s,
            cov_s = (1 - l_shrink)*cov_base_err + l_shrink*cov_base_err_d

            # Compute MAP estimate (A.43)
            self.sigma_r = (t/(1 - l_shrink))/(t + nsm)*(P @ cov_s @ P.T - l_shrink*np.linalg.inv((np.linalg.inv(cov_bot_hat_d) + S_top.T @ np.linalg.inv(cov_top_hat_d) @ S_top)))
#            self.sigma_r = P @ cov_base_err @ P.T

            # Compute parameter variance estimate
            self.beta_cov = np.kron(self.sigma_r, np.linalg.inv(X.T @X))

    def update_fit(self, X, Y, X_pred, window: int = None, l_shrink = 0):

        if self.skip_k:
            k = max(X.fc.kseq)

            # Amend window size to account for k
            if window is not None:
                window = window*k


        if window is None:

            if self.skip_k:
                # Select only every k'th row
                X = X.iloc[::-k].iloc[::-1]
                Y = Y.iloc[::-k].iloc[::-1]
            
            self.shrinkage_update(X, Y, l_shrink)

            # Predict
            predictions = self.predict(X_pred)

        else:

            # Update stored data
            self.data = self.data.append(c.ForecastTuple(X,Y))

            n_t = len(self.data[0])
            n_t_new = len(X)
            start_index = n_t - n_t_new

            predictions = c.new_fc(columns = self.y_columns,index = X_pred.index)

            # Iterate in sliding windows of size window
            for i in range(n_t_new):
                j = i + start_index # Window end index

                # Subset data
                start = max(0, j - window)
                X_i, Y_i = self.data[0].iloc[start:j], self.data[1].iloc[start:j]

                if self.skip_k:
                    # Select only every k'th row
                    X_i = X_i.iloc[::k]
                    Y_i = Y_i.iloc[::k]

                X_i_ready = not np.isnan(X_i).any(axis=None) and len(X_i) > 0
                Y_i_ready = not np.isnan(Y_i).any(axis = None) and len(Y_i) > 0

                if X_i_ready and Y_i_ready:

                    self.shrinkage_update(X_i, Y_i, l_shrink)

                # Predict
                if not np.isnan(X_pred.iloc[i]).any() and not self.beta is None and not np.isnan(self.beta).any():
                    predictions.iloc[i] = self.predict(X_pred.iloc[i])

            # Store only what is needed for subsequent predictions
            self.data.remove_old_data(window)


        return predictions

    def predict(self, X):
        res = X @ self.beta

        if self.estimate_variance:

#            var_est = pd.DataFrame(index = X.index, columns = res.columns)
#            for i in range(len(X)):
#                X_full_i = np.kron(np.eye(self.beta.shape[1]), X.iloc[i])
#                var_est.iloc[i] = np.diag(X_full_i @ self.beta_cov @ X_full_i.T + self.sigma_r)

            
            var_est = pd.DataFrame(index = X.index, columns = ["cov"])
            for i in range(len(X)):
                X_full_i = np.kron(np.eye(self.beta.shape[1]), X.iloc[i])
                var_est.at[i, "cov"] = X_full_i @ self.beta_cov @ X_full_i.T + self.sigma_r


#        invXX = np.linalg.inv(X.T @ X)
#        leverages = np.einsum('ij,jk,ik->i', X, invXX, X)  # shape: (T,)
#        diag_beta_cov = np.diag(self.beta_cov)              # shape: (m,)
#        sigma_r_diag = np.diag(self.sigma_r)                # shape: (m,)
#        var_est = np.outer(leverages, diag_beta_cov) + np.tile(sigma_r_diag, (X.shape[0], 1))

            #return res, var_est
            return res, var_est

        return res

def tikhonov(X, Y, P, Q, B0):
    #TODO: Find source for this or use instead ||X^t @ B - Y||^2 + ||L(B - B_0)||^2 (many sources)
    # P is redundant for current use case
    tmp1 = X.T @ P @ X + Q
    tmp2 = X.T @ P @ Y + Q @ B0

    return np.linalg.solve(tmp1, tmp2)

class RSP(c.Predictor):

    # Recursive Shrinkage Predictor

    def __init__(self, x_columns = None, y_columns = None):
        super().__init__(x_columns, y_columns)
        self.t = 0

    def predict(self, X):
        return X @ self.beta

    def update_fit(self, X, Y, X_pred, l_shrink = 0, l_mem = 1):


        S_top = self.S_top
        nms, m = S_top.shape
        base_err_top = X - Y @ S_top.T
        base_err = np.hstack((base_err_top, Y))

        # Iterate over rows
        for x_i, y_i, x_pred_i, base_err_i in zip(X, Y, X_pred, base_err):

            self.R = l_mem*self.R + np.outer(x_i, x_i)
            self.h = l_mem*self.h + np.outer(x_i, y_i)



            ### SHRINKAGE PARAMETERS ###

            # Update variance estimates and optimal shrinkage parameter recursively
            # TODO: consider: to match original GLM, this should not be recursive, but rather use the full covariance matrix
            # Alternatively the GLM form with recursively updated variance estimates should proved to be equivalent.
            if self.estimate_variance:
                self.cov = l_mem*self.cov + (1-l_mem)*np.outer(base_err_i, base_err_i)
                self.var = np.diag(self.cov)
            else:
                self.var = l_mem*self.var + (1-l_mem)*np.diag(np.sum(base_err_i*base_err_i, 0))

            # Compute shrinkage parameters (should in best case be estimated recursively)
            b0 = self.var[:nms] + S_top @ self.var[nms:] @ S_top.T
            b1 = S_top @ self.var[nms:]
            Q = l_shrink*t/(1-l_shrink)*b0


            b0 = cov_top_hat_d + S_top @ cov_bot_hat_d @ S_top.T
            b1 = S_top @ cov_bot_hat_d
            Q = l_shrink*t/(1-l_shrink)*b0
            B0 = np.linalg.solve(b0, b1)

            ### END SHRINKAGE PARAMETERS ###

            # Update beta
            tmp1, tmp2 = self.R + Q, self.h + Q @ B0
            self.beta = np.linalg.solve(tmp1, tmp2)

            # Predict
            self.predict(x_pred_i)



class Reconciler(c.ForecastModel):
    
    def __init__(self, S: np.ndarray, source = c.RawData, predictor = ShrinkagePredictor, predictor_init_params={}, predictor_params={}):
        # Assumes first m colums of data/source are Y, next n columns are Y_hat

        # TODO: consider allowing any predictor

        self.S = S

        if predictor is ShrinkagePredictor:
            predictor_init_params = predictor_init_params | {"S_top": self.S_top}

        # Subclass predictor to fit hierarchy

        class HierarchyPredictor(predictor):

            def update_fit(self, X, Y, X_pred, *args, **kwargs):
                n, m = S.shape
                X_h = X.iloc[:,:n-m] # GLM from hierarchy transform for fitting
                X_pred_h = X_pred.iloc[:,:n-m] # GLM from hierarchy transform for prediction
                pred_base_fc = X_pred.iloc[:,-n:] # Base forecasts for prediction

                # Fit model and generate error predictions
                res = super().update_fit(X_h, Y, X_pred_h, *args, **kwargs)

                if isinstance(res, tuple):
                    rec_err_pred, *other = res
                else:
                    rec_err_pred = res
                    other = []

                # Compute reconciled forecasts from error predictions
                rec_base_fc = (pred_base_fc.iloc[:,-m:] + rec_err_pred.values) # Reconciled bottom level forecasts

                rec_fc =  rec_base_fc @ S.T # Reconciled forecasts

                rec_fc.columns = pred_base_fc.columns

                if len(other) > 0:

                    # Assumes variance is second output, i.e. first element of other
                    var_bot_est, *other = other

                    var_est = var_bot_est.copy()

                    # Compute reconciled variance estimates
                    for i in range(len(var_est)):
                        var_est.at[i,"cov"] = (S @ var_bot_est.at[i, "cov"] @ S.T)                    

                    return c.ForecastTuple(rec_fc, var_est, *other)

                return rec_fc

        # Copy parameters
        HierarchyPredictor.params = predictor.params

        # Initialise superclass
        super().__init__(HierarchyPredictor, predictor_init_params, predictor_params, False, False)

        n, m = self.S.shape
            
        # Construct transforms
        hierarchy = HierarchyTransform(self.S_top, data = source)
        fc_error = c.IndexMap(*[i for i in range(m)], data = hierarchy)
        rec_err_and_base_fc = c.IndexMap(*[i for i in range(m, n + n)], data = hierarchy)

        # Add inputs and outputs
        self.add_inputs(X = rec_err_and_base_fc)
        self.add_outputs(Y = fc_error)

    def hierarchy_update(self, Y, Y_hat, **predictor_params):

        data = pd.concat([Y, Y_hat], axis = 1)

        return self.update(data, **predictor_params)

    @property
    def S_top(self):
        return self.S[:-self.S.shape[1]]

    @property
    def P(self):
        n, m = self.S.shape
        null = np.zeros_like((n-m, n-m))
        return (np.vstack((null, np.eye(m))) + np.vstack((np.eye(n-m), -self.S_top.T)) @ self.predictors[(None, None)].beta).T

    def hierarchy_fit(self, Y, Y_hat, **predictor_params):
        self.reset_state()
        return self.hierarchy_update(Y, Y_hat, **predictor_params)

class TemporalReconciler(Reconciler):


    def __init__(self, columns: pd.MultiIndex, predictor = ShrinkagePredictor, skip_k = True, predictor_init_params={}, predictor_params={}):
        # TODO: generalize to allow S as input. TemporalHierarchyTransform should be updated accordingly.

        # Infer hierarchy and check if columns adhere to expected format
        S = self.infer_hierarchy(columns)
       
        n, m = S.shape

        source = TemporalHierarchyTransform(S[:-m])

        super().__init__(S, source, predictor, predictor_init_params | {"skip_k": skip_k}, predictor_params)


    def infer_hierarchy(self, columns: pd.MultiIndex):
        # Infers hierarchy from data columns. The input data should be a forecast_matrix type. First level columns should either consider the observation (any name), or integers representing the hierarchy levels > 0.
        # Second level columns should be integers representing the forecast horizon.

        # Check that columns adhere to the expected format
        obs_col = None
        fc_levels = {}
        level = None

        for col in columns:
            if not isinstance(col[0], int):
                if obs_col is not None:
                    raise ValueError("Multiple observation columns")
                else:
                    obs_col = col
            else:

                if not col[0] in fc_levels:
                    fc_levels[col[0]] = [col[1]]

                    if level is not None and col[0] != level - 1:
                        if col[0] > level:
                            raise ValueError("Levels should be in descending order")
                        else:
                            raise ValueError("Levels should be contiguous")
                else:
                    # Check that forecast horizons are in ascending order
                    if col[1] < fc_levels[col[0]][-1]:
                        raise ValueError("Forecast horizons should be in ascending order")

                    # Check that horizons are multiple of first
                    if col[1] % fc_levels[col[0]][0] != 0:
                        raise ValueError("Forecast horizons should be multiples of the first horizon in the level")
                    
                    fc_levels[col[0]].append(col[1])

                level = col[0]
            
        if min(fc_levels.keys()) != 0:
            raise ValueError("Bottom level not found")

        if obs_col is None:
            raise ValueError("No observation column")

        if columns[-1] != obs_col:
            raise ValueError("Observation column should be the last column")

        # Extract first (minimum) horizon for each level
        aggr_levels = tuple(v[0] for v in fc_levels.values())

        S = construct_temporal_hierarchy(hierarchy_shape(*aggr_levels))
        
        return S


    def update(self, data, **predictor_params):

        # Check that data adheres to expected format
        res = super().update(data, **predictor_params)

        is_tuple = isinstance(res, tuple)

        if is_tuple:
            rec_fc = res[0]
        else:
            rec_fc = res

        rec_fc.columns = data.columns[:-1]

        if is_tuple:
            return c.ForecastTuple(rec_fc, *res[1:])
        else:
            return rec_fc
