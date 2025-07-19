# reconciliation.py 

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import py_online_forecast.core_main as c
from py_online_forecast.core_main import *
import py_online_forecast.hierarchies as h
from py_online_forecast.hierarchies import *

#
# RLS
res1_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res1/rls_res1h.csv")
res2_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res2/rls_res2h.csv")
res6_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res6/rls_res6h.csv")
res12_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res12/rls_res12h.csv")
res24_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res24/rls_res24h.csv")
res48_rls = c.read_forecast_csv("../7_results/Case_2/1_rls/res48/rls_res48h.csv")

rls_1_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res1/rls_res1h_with_obs.csv")
rls_2_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res2/rls_res2h_with_obs.csv")
rls_6_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res6/rls_res6h_with_obs.csv")
rls_12_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res12/rls_res12h_with_obs.csv")
rls_24_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res24/rls_res24h_with_obs.csv")
rls_48_obs = c.read_forecast_csv("../7_results/Case_2/1_rls/res48/rls_res48h_with_obs.csv")

res1_rls = res1_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24), ('', 25), ('', 26), ('', 27), ('', 28),
                ('', 29), ('', 30), ('', 31), ('', 32), ('', 33), ('', 34), ('', 35), ('', 36), ('', 37), ('', 38),
                ('', 39), ('', 40), ('', 41), ('', 42), ('', 43), ('', 44), ('', 45), ('', 46), ('', 47), ('', 48)]
res1_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res2_rls = res2_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24)]
res2_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res6_rls = res6_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8)]
res6_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res12_rls = res12_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4)]
res12_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res24_rls = res24_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('',2)]
res24_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res48_rls = res48_rls.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1)]
res48_rls.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res1_rls = res1_rls[121:]
res1_rls = res1_rls[:-48]
res1_rls = res1_rls.reset_index(drop=True)
res2_rls = res2_rls[120:]
res2_rls = res2_rls[:-46]
res2_rls = res2_rls.reset_index(drop=True)
res6_rls = res6_rls[116:]
res6_rls = res6_rls[:-42]
res6_rls = res6_rls.reset_index(drop=True)
res12_rls = res12_rls[110:]
res12_rls = res12_rls[:-36]
res12_rls = res12_rls.reset_index(drop=True)
res24_rls = res24_rls[98:]
res24_rls = res24_rls[:-24]
res24_rls = res24_rls.reset_index(drop=True)
res48_rls = res48_rls[74:]
res48_rls = res48_rls.reset_index(drop=True)

rls_1_obs = rls_1_obs[121:]
rls_1_obs = rls_1_obs[:-48]
rls_1_obs = rls_1_obs.reset_index(drop=True)
rls_2_obs = rls_2_obs[120:]
rls_2_obs = rls_2_obs[:-46]
rls_2_obs = rls_2_obs.reset_index(drop=True)
rls_6_obs = rls_6_obs[116:]
rls_6_obs = rls_6_obs[:-42]
rls_6_obs = rls_6_obs.reset_index(drop=True)
rls_12_obs = rls_12_obs[110:]
rls_12_obs = rls_12_obs[:-36]
rls_12_obs = rls_12_obs.reset_index(drop=True)
rls_24_obs = rls_24_obs[98:]
rls_24_obs = rls_24_obs[:-24]
rls_24_obs = rls_24_obs.reset_index(drop=True)
rls_48_obs = rls_48_obs[74:]
rls_48_obs = rls_48_obs.reset_index(drop=True)





# ARMAX
res1_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res1/armax_res1h.csv")
res2_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res2/armax_res2h.csv")
res6_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res6/armax_res6h.csv")
res12_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res12/armax_res12h.csv")
res24_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res24/armax_res24h.csv")
res48_armax = c.read_forecast_csv("../7_results/Case_2/2_armax/res48/armax_res48h.csv")

armax_1_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res1/armax_res1h_with_obs.csv")
armax_2_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res2/armax_res2h_with_obs.csv")
armax_6_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res6/armax_res6h_with_obs.csv")
armax_12_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res12/armax_res12h_with_obs.csv")
armax_24_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res24/armax_res24h_with_obs.csv")
armax_48_obs = c.read_forecast_csv("../7_results/Case_2/2_armax/res48/armax_res48h_with_obs.csv")

res1_armax = res1_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10), 
               ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18), 
               ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24), ('', 25), ('', 26), ('', 27), ('', 28),
               ('', 29), ('', 30), ('', 31), ('', 32), ('', 33), ('', 34), ('', 35), ('', 36), ('', 37), ('', 38),
               ('', 39), ('', 40), ('', 41), ('', 42), ('', 43), ('', 44), ('', 45), ('', 46), ('', 47), ('', 48)]
res1_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res2_armax = res2_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10), 
               ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18), 
               ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24)]
res2_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res6_armax = res6_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8)]
res6_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res12_armax = res12_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4)]
res12_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res24_armax = res24_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2)]
res24_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res48_armax = res48_armax.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1)]
res48_armax.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res1_armax = res1_armax[121:]
res1_armax = res1_armax[:-48]
res1_armax = res1_armax.reset_index(drop=True)
res2_armax = res2_armax[120:]
res2_armax = res2_armax[:-46]
res2_armax = res2_armax.reset_index(drop=True)
res6_armax = res6_armax[116:]
res6_armax = res6_armax[:-42]
res6_armax = res6_armax.reset_index(drop=True)
res12_armax = res12_armax[110:]
res12_armax = res12_armax[:-36]
res12_armax = res12_armax.reset_index(drop=True)
res24_armax = res24_armax[98:]
res24_armax = res24_armax[:-24]
res24_armax = res24_armax.reset_index(drop=True)
res48_armax = res48_armax[74:]
res48_armax = res48_armax.reset_index(drop=True)

armax_1_obs = armax_1_obs[121:]
armax_1_obs = armax_1_obs[:-48]
armax_1_obs = armax_1_obs.reset_index(drop=True)
armax_2_obs = armax_2_obs[120:]
armax_2_obs = armax_2_obs[:-46]
armax_2_obs = armax_2_obs.reset_index(drop=True)
armax_6_obs = armax_6_obs[116:]
armax_6_obs = armax_6_obs[:-42]
armax_6_obs = armax_6_obs.reset_index(drop=True)
armax_12_obs = armax_12_obs[110:]
armax_12_obs = armax_12_obs[:-36]
armax_12_obs = armax_12_obs.reset_index(drop=True)
armax_24_obs = armax_24_obs[98:]
armax_24_obs = armax_24_obs[:-24]
armax_24_obs = armax_24_obs.reset_index(drop=True)
armax_48_obs = armax_48_obs[74:]
armax_48_obs = armax_48_obs.reset_index(drop=True)





# Decision Trees
res1_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res1/arfr_res1h.csv")
res2_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res2/arfr_res2h.csv")
res6_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res6/arfr_res6h.csv")
res12_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res12/arfr_res12h.csv")
res24_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res24/arfr_res24h.csv")
res48_arfr = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res48/arfr_res48h.csv")
arfr_1_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res1/arfr_res1h_with_obs.csv")
arfr_2_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res2/arfr_res2h_with_obs.csv")
arfr_6_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res6/arfr_res6h_with_obs.csv")
arfr_12_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res12/arfr_res12h_with_obs.csv")
arfr_24_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res24/arfr_res24h_with_obs.csv")
arfr_48_obs = c.read_forecast_csv("../7_results/Case_2/4_ohdt/res48/arfr_res48h_with_obs.csv")

res1_arfr = res1_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24), ('', 25), ('', 26), ('', 27), ('', 28),
                ('', 29), ('', 30), ('', 31), ('', 32), ('', 33), ('', 34), ('', 35), ('', 36), ('', 37), ('', 38),
                ('', 39), ('', 40), ('', 41), ('', 42), ('', 43), ('', 44), ('', 45), ('', 46), ('', 47), ('', 48)]
res1_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res2_arfr = res2_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24)]
res2_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res6_arfr = res6_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8)]
res6_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res12_arfr = res12_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4)]
res12_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res24_arfr = res24_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1), ('', 2)]
res24_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res48_arfr = res48_arfr.drop(columns=[('Variable', 'NA')])
new_columns = [('HC.f', 1)]
res48_arfr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res1_arfr = res1_arfr[121:]
res1_arfr = res1_arfr[:-48]
res1_arfr = res1_arfr.reset_index(drop=True)    
res2_arfr = res2_arfr[120:]
res2_arfr = res2_arfr[:-46]
res2_arfr = res2_arfr.reset_index(drop=True)    
res6_arfr = res6_arfr[116:]
res6_arfr = res6_arfr[:-42]
res6_arfr = res6_arfr.reset_index(drop=True)
res12_arfr = res12_arfr[110:]
res12_arfr = res12_arfr[:-36]
res12_arfr = res12_arfr.reset_index(drop=True)
res24_arfr = res24_arfr[98:]
res24_arfr = res24_arfr[:-24]
res24_arfr = res24_arfr.reset_index(drop=True)
res48_arfr = res48_arfr[74:]
res48_arfr = res48_arfr.reset_index(drop=True)

arfr_1_obs = arfr_1_obs[121:]
arfr_1_obs = arfr_1_obs[:-48]
arfr_1_obs = arfr_1_obs.reset_index(drop=True)
arfr_2_obs = arfr_2_obs[120:]
arfr_2_obs = arfr_2_obs[:-46]
arfr_2_obs = arfr_2_obs.reset_index(drop=True)
arfr_6_obs = arfr_6_obs[116:]
arfr_6_obs = arfr_6_obs[:-42]
arfr_6_obs = arfr_6_obs.reset_index(drop=True)
arfr_12_obs = arfr_12_obs[110:]
arfr_12_obs = arfr_12_obs[:-36]
arfr_12_obs = arfr_12_obs.reset_index(drop=True)
arfr_24_obs = arfr_24_obs[98:]
arfr_24_obs = arfr_24_obs[:-24]
arfr_24_obs = arfr_24_obs.reset_index(drop=True)
arfr_48_obs = arfr_48_obs[74:]
arfr_48_obs = arfr_48_obs.reset_index(drop=True)





# Online SVR
res1_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res1/svr_res1h.csv")
res2_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res2/svr_res2h.csv")
res6_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res6/svr_res6h.csv")
res12_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res12/svr_res12h.csv")
res24_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res24/svr_res24h.csv")
res48_svr = c.read_forecast_csv("../7_results/Case_2/3_svr/res48/svr_res48h.csv")
svr_1_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res1/svr_res1h_with_obs.csv")
svr_2_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res2/svr_res2h_with_obs.csv")
svr_6_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res6/svr_res6h_with_obs.csv")
svr_12_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res12/svr_res12h_with_obs.csv")
svr_24_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res24/svr_res24h_with_obs.csv")
svr_48_obs = c.read_forecast_csv("../7_results/Case_2/3_svr/res48/svr_res48h_with_obs.csv")

res1_svr = svr_1_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f.48', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24), ('', 25), ('', 26), ('', 27), ('', 28),
                ('', 29), ('', 30), ('', 31), ('', 32), ('', 33), ('', 34), ('', 35), ('', 36), ('', 37), ('', 38),
                ('', 39), ('', 40), ('', 41), ('', 42), ('', 43), ('', 44), ('', 45), ('', 46), ('', 47), ('', 48)]
res1_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res2_svr = svr_2_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f_future2h.23', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8), ('', 9), ('', 10),
                ('', 11), ('', 12), ('', 13), ('', 14), ('', 15), ('', 16), ('', 17), ('', 18),
                ('', 19), ('', 20), ('', 21), ('', 22), ('', 23), ('', 24)]
res2_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res6_svr = svr_6_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f_future6h.7', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4), ('', 5), ('', 6), ('', 7), ('', 8)]
res6_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res12_svr = svr_12_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f_future12h.3', 'NA')])
new_columns = [('HC.f', 1), ('', 2), ('', 3), ('', 4)]
res12_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res24_svr = svr_24_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f_future24h.1', 'NA')])
new_columns = [('HC.f', 1), ('', 2)]
res24_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res48_svr = svr_48_obs.drop(columns=[('Variable', 'NA'), ('Unnamed: 1', 'NA'), ('HC.f_future48h', 'NA')])
new_columns = [('HC.f', 1)]
res48_svr.columns = pd.MultiIndex.from_tuples(new_columns, names=['Variable', 'Horizon'])

res1_svr = res1_svr[121:]
res1_svr = res1_svr[:-96]
res1_svr = res1_svr.reset_index(drop=True)
res2_svr = res2_svr[120:]
res2_svr = res2_svr[:-46]
res2_svr = res2_svr.reset_index(drop=True)
res6_svr = res6_svr[116:]
res6_svr = res6_svr[:-42]
res6_svr = res6_svr.reset_index(drop=True)
res12_svr = res12_svr[110:]
res12_svr = res12_svr[:-36]
res12_svr = res12_svr.reset_index(drop=True)
res24_svr = res24_svr[98:]
res24_svr = res24_svr[:-24]
res24_svr = res24_svr.reset_index(drop=True)
res48_svr = res48_svr[74:]
res48_svr = res48_svr.reset_index(drop=True)

svr_1_obs = svr_1_obs[121:]
svr_1_obs = svr_1_obs[:-96]
svr_1_obs = svr_1_obs.reset_index(drop=True)
svr_2_obs = svr_2_obs[120:]
svr_2_obs = svr_2_obs[:-46]
svr_2_obs = svr_2_obs.reset_index(drop=True)
svr_6_obs = svr_6_obs[116:]
svr_6_obs = svr_6_obs[:-42]
svr_6_obs = svr_6_obs.reset_index(drop=True)
svr_12_obs = svr_12_obs[110:]
svr_12_obs = svr_12_obs[:-36]
svr_12_obs = svr_12_obs.reset_index(drop=True)
svr_24_obs = svr_24_obs[98:]
svr_24_obs = svr_24_obs[:-24]
svr_24_obs = svr_24_obs.reset_index(drop=True)
svr_48_obs = svr_48_obs[74:]
svr_48_obs = svr_48_obs.reset_index(drop=True)










# Final check shapes of predictions 
print(res1_rls.shape, res1_armax.shape, res1_arfr.shape, res1_svr.shape)
print(res2_rls.shape, res2_armax.shape, res2_arfr.shape, res2_svr.shape)
print(res6_rls.shape, res6_armax.shape, res6_arfr.shape, res6_svr.shape)
print(res12_rls.shape, res12_armax.shape, res12_arfr.shape, res12_svr.shape)
print(res24_rls.shape, res24_armax.shape, res24_arfr.shape, res24_svr.shape)
print(res48_rls.shape, res48_armax.shape, res48_arfr.shape, res48_svr.shape)

print(rls_1_obs.shape, armax_1_obs.shape, arfr_1_obs.shape, svr_1_obs.shape)
print(rls_2_obs.shape, armax_2_obs.shape, arfr_2_obs.shape, svr_2_obs.shape)
print(rls_6_obs.shape, armax_6_obs.shape, arfr_6_obs.shape, svr_6_obs.shape)
print(rls_12_obs.shape, armax_12_obs.shape, arfr_12_obs.shape, svr_12_obs.shape)
print(rls_24_obs.shape, armax_24_obs.shape, arfr_24_obs.shape, svr_24_obs.shape)
print(rls_48_obs.shape, armax_48_obs.shape, arfr_48_obs.shape, svr_48_obs.shape)


# print sum of nan 
print("Sum of NaNs in res1_rls:", res1_rls.isna().sum().sum())
print("Sum of NaNs in res1_armax:", res1_armax.isna().sum().sum())
print("Sum of NaNs in res1_arfr:", res1_arfr.isna().sum().sum())
print("Sum of NaNs in res1_svr:", res1_svr.isna().sum().sum())

print("Sum of NaNs in res2_rls:", res2_rls.isna().sum().sum())
print("Sum of NaNs in res2_armax:", res2_armax.isna().sum().sum())
print("Sum of NaNs in res2_arfr:", res2_arfr.isna().sum().sum())
print("Sum of NaNs in res2_svr:", res2_svr.isna().sum().sum())

print("Sum of NaNs in res6_rls:", res6_rls.isna().sum().sum())
print("Sum of NaNs in res6_armax:", res6_armax.isna().sum().sum())
print("Sum of NaNs in res6_arfr:", res6_arfr.isna().sum().sum())
print("Sum of NaNs in res6_svr:", res6_svr.isna().sum().sum())

print("Sum of NaNs in res12_rls:", res12_rls.isna().sum().sum())
print("Sum of NaNs in res12_armax:", res12_armax.isna().sum().sum())
print("Sum of NaNs in res12_arfr:", res12_arfr.isna().sum().sum())
print("Sum of NaNs in res12_svr:", res12_svr.isna().sum().sum())

print("Sum of NaNs in res24_rls:", res24_rls.isna().sum().sum())
print("Sum of NaNs in res24_armax:", res24_armax.isna().sum().sum())
print("Sum of NaNs in res24_arfr:", res24_arfr.isna().sum().sum())
print("Sum of NaNs in res24_svr:", res24_svr.isna().sum().sum())

print("Sum of NaNs in res48_rls:", res48_rls.isna().sum().sum())
print("Sum of NaNs in res48_armax:", res48_armax.isna().sum().sum())
print("Sum of NaNs in res48_arfr:", res48_arfr.isna().sum().sum())
print("Sum of NaNs in res48_svr:", res48_svr.isna().sum().sum())



obs = rls_1_obs[("HC.f", "NA")]
print(obs.shape)


obs_2 = rls_2_obs[("HC.f_future2h", "NA")]
obs_6 = rls_6_obs[("HC.f_future6h", "NA")]
obs_12 = rls_12_obs[("HC.f_future12h", "NA")]
obs_24 = rls_24_obs[("HC.f_future24h", "NA")]
obs_48 = rls_48_obs[("HC.f_future48h", "NA")]






#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 1h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res1_rls,
    'ARMAX': res1_armax,
    'HRT':  res1_arfr,
    'SVR':   res1_svr
}

# define a unique marker and color for each model
markers = {
    'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()





#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 2h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res2_rls,
    'ARMAX': res2_armax,
    'HRT':  res2_arfr,
    'SVR':   res2_svr
}

# define a unique marker and color for each model
markers = {
        'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs_2.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()









#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 6h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res6_rls,
    'ARMAX': res6_armax,
    'HRT':  res6_arfr,
    'SVR':   res6_svr
}

# define a unique marker and color for each model
markers = {
    'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs_6.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()






#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 12h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res12_rls,
    'ARMAX': res12_armax,
    'HRT':  res12_arfr,
    'SVR':   res12_svr
}

# define a unique marker and color for each model
markers = {
    'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs_12.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()






#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 24h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res24_rls,
    'ARMAX': res24_armax,
    'HRT':  res24_arfr,
    'SVR':   res24_svr
}

# define a unique marker and color for each model
markers = {
    'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs_24.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()






#%%
# ------------------------------------------------------------
# RMSE plot for each model across all horizons - 48h
# ------------------------------------------------------------

print("--- Calculating RMSE for Each Horizon Across All Models ---")

models = {
    'RLS':   res48_rls,
    'ARMAX': res48_armax,
    'HRT':  res48_arfr,
    'SVR':   res48_svr
}

# define a unique marker and color for each model
markers = {
    'RLS':   'o',
    'ARMAX': 's',
    'HRT':  '^',
    'SVR':   'D'
}
colors = {
    'RLS':   'orange',
    'ARMAX': 'darkseagreen',
    'HRT':  'indianred',
    'SVR':   'cornflowerblue'
}

rmse_list = []

for model_name, df in models.items():
    for i in range(df.shape[1]):
        h = i + 1
        forecasts = df.iloc[:, i]

        valid_fc = forecasts.iloc[:-h]
        obs_h    = obs_48.iloc[h:]

        err  = obs_h.values - valid_fc.values
        rmse = np.sqrt(np.mean(err**2))

        rmse_list.append({
            'Model':   model_name,
            'Horizon': h,
            'RMSE':    rmse
        })

# assemble results and pivot
rmse_all   = pd.DataFrame(rmse_list)
rmse_table = rmse_all.pivot(index='Horizon', columns='Model', values='RMSE').round(4)
rmse_table['RLS'] = 1512.64

# plotting
plt.figure(figsize=(12, 4))

for model in rmse_table.columns:
    plt.plot(
        rmse_table.index,
        rmse_table[model],
        marker=markers[model],
        color=colors[model],
        linestyle='-',
        linewidth=0.9,
        label=model
    )

plt.title('RMSE by Forecast Horizon for Each Model', fontsize=16)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_table.index)
plt.legend(title='Model')

# remove grid entirely
plt.grid(False)

# show only bottom & left spines
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()






































#%%
# ------------------------------------------------------------
# Diagnostic plots for h=1
# ------------------------------------------------------------

h = 1

for model_name, df in models.items():
    # 1) get h-step-ahead residuals
    fc    = df.iloc[:, h - 1].iloc[:-h]
    obs_h = obs.iloc[h:]
    resid = obs_h.values - fc.values
    n     = len(resid)
    flat_conf = 1.96 / np.sqrt(n)

    # 2) set up GridSpec: top row spans two columns, bottom has 2 panels
    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3)
    ax_ts   = fig.add_subplot(gs[0, :])
    ax_acf  = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"{model_name} – Residual Diagnostics", fontsize=16)

    # ----------------------------------------------------------------------------
    # 3) Residuals over time (top, spanning both columns)
    ax_ts.plot(
        resid,
        marker='o', linestyle='-', color='k',
        markerfacecolor='k', markeredgecolor='k'
    )
    ax_ts.axhline(0, color='grey', linestyle='--')
    ax_ts.set_title("Residuals over Time")
    ax_ts.set_xlabel("t")
    ax_ts.set_ylabel("Residual")

    # ----------------------------------------------------------------------------
    # 4) ACF with flat ±1.96/√n bands (bottom‐left)
    lags    = np.arange(0, 41)
    acf_vals= acf(resid, nlags=40, fft=False)
    ax_acf.stem(lags, acf_vals, linefmt='k-', markerfmt='ko', basefmt='k-')
    ax_acf.fill_between(lags, -flat_conf, flat_conf, color='lightblue', alpha=0.5)
    ax_acf.set_title("Autocorrelation")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylim(-1, 1)

    # ----------------------------------------------------------------------------
    # 5) Histogram + Normal density (bottom‐right)
    # histogram (gray, frequency → density)
    ax_hist.hist(resid, bins=20, density=True, color='darkgray', edgecolor='black', alpha=0.8)
    # overlay normal PDF
    mu, sigma = resid.mean(), resid.std(ddof=0)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax_hist.plot(xs, stats.norm.pdf(xs, mu, sigma), linestyle='-', color='tomato', linewidth=2)
    # optional: small rug
    ax_hist.plot(resid, np.full_like(resid, -0.002), '|k', markeredgewidth=1)
    ax_hist.set_title("Histogram of Residuals")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Density")

    # finalize
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()









#%% Reconcilied forecasts RMSE ml

# read csv files

print("--- Plotting RMSE_Reconciled vs RMSE_Base ---")

# load the results
rmse_df = pd.read_csv('rmse_case2_ml_2.csv')

# define markers and colors
markers = {
    'RMSE_Reconciled': 'o',
    'RMSE_Base':      's'
}
colors = {
    'RMSE_Reconciled': 'cadetblue',
    'RMSE_Base':       'indianred'
}

label_map = {
    'RMSE_Reconciled': 'Reconciled',
    'RMSE_Base':       'HRT'
}

# plotting
plt.figure(figsize=(12, 4))

for col in ['RMSE_Reconciled', 'RMSE_Base']:
    plt.plot(
        rmse_df['Horizon'],
        rmse_df[col],
        marker=markers[col],
        color=colors[col],
        linestyle='-',
        linewidth=0.9,
        label=label_map[col]
    )

plt.title('RMSE by Forecast Horizon: Machine Learning Models Case 2', fontsize=13)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_df['Horizon'])
plt.legend(title='Series')

# remove grid, show only bottom & left spines
plt.grid(False)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()




#%% Reconcilied forecasts RMSE st

print("--- Plotting RMSE_Reconciled vs RMSE_Base ---")

# load the results
rmse_df = pd.read_csv('rmse_case2_st_2.csv')

# define markers and colors
markers = {
    'RMSE_Reconciled': 'o',
    'RMSE_Base':      's'
}
colors = {
    'RMSE_Reconciled': 'slateblue',
    'RMSE_Base':       'orange'
}

label_map = {
    'RMSE_Reconciled': 'Reconciled',
    'RMSE_Base':       'RLS'
}

# plotting
plt.figure(figsize=(12, 4))

for col in ['RMSE_Reconciled', 'RMSE_Base']:
    plt.plot(
        rmse_df['Horizon'],
        rmse_df[col],
        marker=markers[col],
        color=colors[col],
        linestyle='-',
        linewidth=0.9,
        label=label_map[col]
    )

plt.title('RMSE by Forecast Horizon: Statistical Models Case 2', fontsize=13)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rmse_df['Horizon'])
plt.legend(title='Series')

# remove grid, show only bottom & left spines
plt.grid(False)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()





#%% Reconcilied forecasts RMSE ml-st



# load the two RMSE files
ml_df = pd.read_csv('rmse_case2_ml_2.csv')
st_df = pd.read_csv('rmse_case2_st_2.csv')

plt.figure(figsize=(12, 4))

# plot reconciled ML
plt.plot(
    ml_df['Horizon'],
    ml_df['RMSE_Reconciled'],
    marker='o',
    color='cadetblue',
    linestyle='-',
    linewidth=0.9,
    label='Reconciled ML'
)

# plot reconciled ST
plt.plot(
    st_df['Horizon'],
    st_df['RMSE_Reconciled'],
    marker='s',
    color='slateblue',
    linestyle='-',
    linewidth=0.9,
    label='Reconciled ST'
)

plt.title('Reconciled RMSE: ML vs ST Case 2', fontsize=13)
plt.xlabel('Horizon', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(ml_df['Horizon'])
plt.legend(title='Series')

# clean up spines
plt.grid(False)
ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
plt.show()












#%%

# plot rls_1_obs, res1_rls, rec_rls1