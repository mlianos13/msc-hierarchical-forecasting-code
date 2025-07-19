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
res1_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res1/rls_res1h.csv")
res2_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res2/rls_res2h.csv")
res6_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res6/rls_res6h.csv")
res12_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res12/rls_res12h.csv")
res24_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res24/rls_res24h.csv")
res48_rls = c.read_forecast_csv("../7_results/Case_1/1_rls/res48/rls_res48h.csv")

rls_1_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res1/rls_res1h_with_obs.csv")
rls_2_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res2/rls_res2h_with_obs.csv")
rls_6_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res6/rls_res6h_with_obs.csv")
rls_12_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res12/rls_res12h_with_obs.csv")
rls_24_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res24/rls_res24h_with_obs.csv")
rls_48_obs = c.read_forecast_csv("../7_results/Case_1/1_rls/res48/rls_res48h_with_obs.csv")

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
res1_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res1/armax_res1h.csv")
res2_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res2/armax_res2h.csv")
res6_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res6/armax_res6h.csv")
res12_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res12/armax_res12h.csv")
res24_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res24/armax_res24h.csv")
res48_armax = c.read_forecast_csv("../7_results/Case_1/2_armax/res48/armax_res48h.csv")

armax_1_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res1/armax_res1h_with_obs.csv")
armax_2_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res2/armax_res2h_with_obs.csv")
armax_6_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res6/armax_res6h_with_obs.csv")
armax_12_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res12/armax_res12h_with_obs.csv")
armax_24_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res24/armax_res24h_with_obs.csv")
armax_48_obs = c.read_forecast_csv("../7_results/Case_1/2_armax/res48/armax_res48h_with_obs.csv")

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
res1_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res1/arfr_res1h.csv")
res2_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res2/arfr_res2h.csv")
res6_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res6/arfr_res6h.csv")
res12_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res12/arfr_res12h.csv")
res24_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res24/arfr_res24h.csv")
res48_arfr = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res48/arfr_res48h.csv")
arfr_1_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res1/arfr_res1h_with_obs.csv")
arfr_2_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res2/arfr_res2h_with_obs.csv")
arfr_6_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res6/arfr_res6h_with_obs.csv")
arfr_12_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res12/arfr_res12h_with_obs.csv")
arfr_24_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res24/arfr_res24h_with_obs.csv")
arfr_48_obs = c.read_forecast_csv("../7_results/Case_1/4_ohdt/res48/arfr_res48h_with_obs.csv")

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
res1_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res1/svr_res1h.csv")
res2_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res2/svr_res2h.csv")
res6_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res6/svr_res6h.csv")
res12_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res12/svr_res12h.csv")
res24_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res24/svr_res24h.csv")
res48_svr = c.read_forecast_csv("../7_results/Case_1/3_svr/res48/svr_res48h.csv")
svr_1_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res1/svr_res1h_with_obs.csv")
svr_2_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res2/svr_res2h_with_obs.csv")
svr_6_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res6/svr_res6h_with_obs.csv")
svr_12_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res12/svr_res12h_with_obs.csv")
svr_24_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res24/svr_res24h_with_obs.csv")
svr_48_obs = c.read_forecast_csv("../7_results/Case_1/3_svr/res48/svr_res48h_with_obs.csv")

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









# Data preparation for hierarchical reconciliation

# Create an empty ForecastMatrix without pre-defining an index
reconciliation_data = c.new_fc()

# For daily (48-hourly) forecasts (level 5) - keep at original frequency
reconciliation_data[(5, 48)] = res48_svr[("HC.f", 1)]  # No reindexing/padding

# For 24-hourly forecasts (level 4) - keep at original frequency
for k in range(1, 3):  # k = 1, 2
    horizon = 24 * k  # horizons: 24 and 48
    if k == 1:
        col_name = ("HC.f", k)
    else:
        col_name = ("", k)
    reconciliation_data[(4, horizon)] = res24_svr[col_name]  # No reindexing

# For 12-hourly forecasts (level 3) - keep at original frequency
for k in range(1, 5):  # k = 1, 2, 3, 4
    horizon = 12 * k
    if k == 1:
        col_name = ("HC.f", k)
    else:
        col_name = ("", k)
    reconciliation_data[(3, horizon)] = res12_arfr[col_name]  # No reindexing

# For 6-hourly forecasts (level 2) - keep at original frequency
for k in range(1, 9):  # k = 1,...,8
    horizon = 6 * k
    if k == 1:
        col_name = ("HC.f", k)
    else:
        col_name = ("", k)
    reconciliation_data[(2, horizon)] = res6_svr[col_name]  # No reindexing

# For 2-hourly forecasts (level 1) - keep at original frequency
for k in range(1, 25):
    horizon = 2 * k
    if k == 1:
        col_name = ("HC.f", k)
    else:
        col_name = ("", k)
    reconciliation_data[(1, horizon)] = res2_svr[col_name]  # No reindexing

# For 1-hourly forecasts (level 0) - keep at original frequency
for k in range(1, 49):
    horizon = k  # 1 to 48
    if k == 1:
        col_name = ("HC.f", k)
    else:
        col_name = ("", k)
    reconciliation_data[(0, horizon)] = res1_arfr[col_name]  # No reindexing

# Add observations at their natural frequency
reconciliation_data[('Y_obs', 'NA')] = obs

# Convert string indices to integers if needed (this part is fine)
obs_col = ("Y_obs", "NA")
new_columns = [col for col in reconciliation_data.columns if col != obs_col]
for col in new_columns:
    if isinstance(col[0], str) and col[0].isdigit():
        new_columns[new_columns.index(col)] = (int(col[0]), col[1])

# Create final data structure
data = c.new_fc(reconciliation_data.values, 
                index=reconciliation_data.index, 
                columns=pd.MultiIndex.from_tuples(new_columns + [obs_col]))















#%% Out-of-Sample MinT Reconciliation: Train/Test Split

l_shrink = 0.1

# 0. Split your full 'data' into a 2/3 train set and 1/3 test set (by rows)
n_total    = len(data)
n_train    = int(n_total * 2/3)
data_train = data.iloc[:n_train]
data_test  = data.iloc[n_train:]

# 1. TRAINING PHASE: Estimate P on data_train
tr_train      = TemporalReconciler(data_train.columns)
S             = tr_train.S
S_top         = tr_train.S_top

# Transform to (Y, Y_hat) format
tdata_train, _ = TemporalHierarchyTransform(S_top).apply(data_train)
tY_train       = tdata_train.iloc[:, :48]
tYhat_train    = tdata_train.iloc[:, 48:]
# Lag and clean exactly as before
Yhat_lagged    = tYhat_train.fc.get_lagged_subset()
Y_train        = tY_train.iloc[48:].iloc[::-2].iloc[::-1]
Yhat_train     = Yhat_lagged.iloc[48:].iloc[::-2].iloc[::-1]

# Compute projection matrix P
P, _ = minT(S, Y_train, Yhat_train, l_shrink=l_shrink)

# 2. TEST PHASE: Reconcile the last 1/3 purely out-of-sample
#    (no Y_obs) → pick only the base‐forecast columns
forecast_cols = [col for col in data_test.columns if col[0] != 'Y_obs']
Yhat_test     = data_test[forecast_cols]

# Apply MinT: rec_out = (S @ (P @ Yhat_test.T)).T
rec_np  = S @ (P @ Yhat_test.values.T)    # <-- note: no .values on S
rec_out = pd.DataFrame(
    data    = rec_np.T,
    index   = data_test.index,
    columns = forecast_cols
)

rec_out.columns = pd.MultiIndex.from_tuples(
    rec_out.columns,
    names=data_test.columns.names  # or ['level','horizon'] if you prefer
)


# 3. Extract reconciled forecasts by aggregation level
#    (these rec_* will now be purely out-of-sample)
# (a) Hourly forecasts (level = 0)
rec_res1 = rec_out.xs(0, level=0, axis=1)
rec_res1.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res1.columns],
    names=['Variable', 'Horizon']
)

# (b) 2-hourly forecasts (level = 1)
rec_res2 = rec_out.xs(1, level=0, axis=1)
rec_res2.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res2.columns],
    names=['Variable', 'Horizon']
)

# (c) 6-hourly forecasts (level = 2)
rec_res6 = rec_out.xs(2, level=0, axis=1)
rec_res6.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res6.columns],
    names=['Variable', 'Horizon']
)

# (d) 12-hourly forecasts (level = 3)
rec_res12 = rec_out.xs(3, level=0, axis=1)
rec_res12.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res12.columns],
    names=['Variable', 'Horizon']
)

# (e) 24-hourly forecasts (level = 4)
rec_res24 = rec_out.xs(4, level=0, axis=1)
rec_res24.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res24.columns],
    names=['Variable', 'Horizon']
)

# (f) 48-hourly forecasts (level = 5)
rec_res48 = rec_out.xs(5, level=0, axis=1)
rec_res48.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in rec_res48.columns],
    names=['Variable', 'Horizon']
)

# Verify everything landed where it should
for name, df in [
    ('rec_res1', rec_res1), ('rec_res2', rec_res2),
    ('rec_res6', rec_res6), ('rec_res12', rec_res12),
    ('rec_res24', rec_res24), ('rec_res48', rec_res48)
]:
    print(f"{name}: shape={df.shape}, NaNs={df.isna().sum().sum()}")





# Prepare out‐of‐sample reconciled data for each frequency level
# Using rec_out from the test‐phase reconciliation

# Hourly forecasts (level 0)
rec_res1 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 0]
rec_res1.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in range(1, 49)],
    names=['Variable', 'Horizon']
)

# 2‐hourly forecasts (level 1)
rec_res2 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 1]
rec_res2.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in range(1, 25)],
    names=['Variable', 'Horizon']
)

# 6‐hourly forecasts (level 2)
rec_res6 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 2]
rec_res6.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in range(1, 9)],
    names=['Variable', 'Horizon']
)

# 12‐hourly forecasts (level 3)
rec_res12 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 3]
rec_res12.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in range(1, 5)],
    names=['Variable', 'Horizon']
)

# 24‐hourly forecasts (level 4)
rec_res24 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 4]
rec_res24.columns = pd.MultiIndex.from_tuples(
    [('HC.f', h) for h in range(1, 3)],
    names=['Variable', 'Horizon']
)

# 48‐hourly forecasts (level 5)
rec_res48 = rec_out.loc[:, rec_out.columns.get_level_values(0) == 5]
rec_res48.columns = pd.MultiIndex.from_tuples(
    [('HC.f', 1)],
    names=['Variable', 'Horizon']
)

# Verify shapes and confirm no NaNs
for name, df in [('rec_res1', rec_res1), ('rec_res2', rec_res2),
                 ('rec_res6', rec_res6), ('rec_res12', rec_res12),
                 ('rec_res24', rec_res24), ('rec_res48', rec_res48)]:
    print(f"{name} shape: {df.shape}, NaNs: {df.isna().sum().sum()}")



# Align data for RMSE comparison

# Extract the out‐of‐sample observations
obs_test = data_test[('Y_obs', 'NA')].reset_index(drop=True)
obs_test.name = 'observations'

# Assume res1_arfr holds your base (e.g. ARFR) forecasts for the test period
# and has the same MultiIndex structure as rec_res1
res1_arfr_test = res1_arfr.iloc[len(res1_arfr) - len(obs_test):].reset_index(drop=True)

# Trim reconciled forecasts to match obs length and reset index
rec_res1_test = rec_res1.reset_index(drop=True).iloc[: len(obs_test)]

print(f"obs_test shape: {obs_test.shape}")
print(f"res1_arfr_test shape: {res1_arfr_test.shape}")
print(f"rec_res1_test shape: {rec_res1_test.shape}")



# Calculate RMSE by horizon and plot
print("--- Generating Plots and Calculating RMSE for Each Horizon ---")
rmse_results = []
N = len(obs_test)

for h in range(1, rec_res1_test.shape[1] + 1):
    idx = h - 1  # zero‐based column index

    # Extract the forecast series for the current horizon `h`.
    forecasts_rec  = rec_res1_test.iloc[:, idx]
    forecasts_base = res1_arfr_test.iloc[:, idx]

    # --- Correct Alignment for RMSE ---
    valid_rec   = forecasts_rec.iloc[: N - h]
    valid_base  = forecasts_base.iloc[: N - h]
    obs_shifted = obs_test.iloc[h:]

    # Calculate errors
    error_rec  = obs_shifted.values - valid_rec.values
    error_base = obs_shifted.values - valid_base.values

    # Calculate the RMSE for the current horizon.
    rmse_rec  = np.sqrt(np.mean(error_rec**2))
    rmse_base = np.sqrt(np.mean(error_base**2))

    rmse_results.append({
        'Horizon':           h,
        'RMSE_Reconciled':   rmse_rec,
        'RMSE_Base':         rmse_base
    })

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 4))
    obs_time_axis = np.arange(N)
    valid_len     = N - h
    forecast_time_axis = np.arange(valid_len) + h

    # Plot observed series
    ax.plot(obs_time_axis, obs_test,
            label='Observed',   color='black',      linewidth=1.8, zorder=1)

    # Plot only valid reconciled forecasts
    ax.plot(forecast_time_axis, valid_rec.values,
            label='Reconciled', color='cadetblue',  linewidth=1.8, zorder=1)

    # Plot only valid base forecasts (HRT)
    ax.plot(forecast_time_axis, valid_base.values,
            label='HRT',        color='indianred',  linewidth=1.8, zorder=2)

    ax.set_title(f'Forecast Comparison at Horizon {h}', fontsize=12)
    ax.set_xlabel('Time Step (t)',        fontsize=12)
    ax.set_ylabel('Heat Load - Case 1',   fontsize=12)
    ax.legend(loc='best')
    ax.grid(False)

    # Hide top/right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Limit x-axis to observed span
    ax.set_xlim(0, N - 1)

    plt.tight_layout()
    plt.show()


print("--- RMSE Comparison by Horizon ---")
comparison_df = pd.DataFrame(rmse_results)
comparison_df['Absolute_Improvement']    = comparison_df['RMSE_Base'] - comparison_df['RMSE_Reconciled']
comparison_df['Percentage_Improvement'] = (comparison_df['Absolute_Improvement'] / comparison_df['RMSE_Base']) * 100

print(comparison_df.round(4))

# Save results
comparison_df.to_csv('rmse_case1_ml_2.csv')