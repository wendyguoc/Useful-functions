import pandas as pd
import numpy as np
import math
from colorama import Fore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif_(X, thresh=10.0):


    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def ks(data=None,target=None, prob=None):

    data['target0'] = 1 - data[target]

    data['bucket'] = pd.qcut(data[prob], 10)

    grouped = data.groupby('bucket', as_index = False)

    kstable = pd.DataFrame()

    kstable['min_score'] = grouped.min()[prob]

    kstable['max_score'] = grouped.max()[prob]

    kstable['events']   = grouped.sum()[target]

    kstable['nonevents'] = grouped.sum()['target0']
    
    kstable = kstable.sort_values(by="min_score", ascending=False).reset_index(drop = True)
    
#     kstable['cumevents']=  kstable.events.cumsum()
#     kstable['cumnonevents']= kstable.nonevents.cumsum()
#     kstable['totevents']= kstable.events.sum()
    kstable['total']= kstable.events+kstable.nonevents
    
    kstable['event_rate']= (kstable.events/kstable.total).apply('{:.2%}'.format)
    kstable['nonevent_rate']= (kstable.nonevents/kstable.total).apply('{:.2%}'.format)
    
    kstable['% events'] = (kstable.events / kstable.events.sum()).apply('{:.2%}'.format)

    kstable['% nonevents'] = (kstable.nonevents / kstable.nonevents.sum()).apply('{:.2%}'.format)

    
    kstable['% cum_events']= (kstable.events / kstable.events.sum()).cumsum()

    kstable['% cum_nonevents']= (kstable.nonevents / kstable.nonevents.sum()).cumsum()
    
    
    kstable['KS'] = np.round(kstable['% cum_events']-kstable['% cum_nonevents'], 3) * 100
    
    kstable['KS']=np.abs(kstable['KS'])

    #Formating

    kstable['% cum_events']= kstable['% cum_events'].apply('{:.2%}'.format)

    kstable['% cum_nonevents']= kstable['% cum_nonevents'].apply('{:.2%}'.format)

    kstable.index = range(1,11)

    kstable.index.rename('Decile', inplace=True)

    pd.set_option('display.max_columns', None)
    
    #last entry
    events=sum(kstable["events"])
    nonevents=sum(kstable["nonevents"])
    grandtot=sum(kstable["total"])
    event_rate="{:.2%}".format(events/grandtot)
    nonevent_rate="{:.2%}".format(nonevents/grandtot)
    percent_event="{:.2%}".format(1)
    percent_nonevent="{:.2%}".format(1)
    
    finalks=max(kstable['KS'])
    lastentry = {'events': events, 'nonevents': nonevents, 'total': grandtot, 'event_rate': event_rate, 
                 'nonevent_rate':nonevent_rate, '% events': percent_event, '% nonevents': percent_nonevent,  'KS': finalks}
    
    kstable = kstable.append(lastentry, ignore_index=True)

    kstable.replace(np.NaN, ' ', inplace=True)
    print(kstable)

    

     #Display KS

    from colorama import Fore

    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))

    return(kstable)

def psi(bench, comp, group):
    ben_len=len(bench)
    comp_len=len(comp)
    bench.sort()
    comp.sort()
 
    psitable=pd.DataFrame(columns = ['lowercut', 'uppercut', 'ben_cnt','ben_pct','comp_cnt','comp_pct','pct_diff','info_odds','psi'])
    ten_split = np.array_split(bench, group)
    for i in range(1,group+1):
        n = len(ten_split[i-1])
        lowercut=bench[(i-1)*n]
        ben_cnt=n
        
        if i<group:
            uppercut=bench[(i*n-1)]
        else:
            uppercut=max(bench[-1],comp[-1])
        comp_cnt= len([i for i in comp if i >= lowercut and i<=uppercut])
        ben_pct=(ben_cnt+0.0)/ben_len
        comp_pct=(comp_cnt+0.0)/comp_len
        pct_diff=ben_pct-comp_pct
        info_odds=math.log(ben_pct/comp_pct)
        psi = pct_diff * info_odds
        groupentry = {'lowercut':lowercut,'uppercut':uppercut,'ben_cnt':ben_cnt,'ben_pct':ben_pct ,\
                      'comp_cnt':comp_cnt,'comp_pct':comp_pct,'pct_diff':pct_diff,'info_odds':info_odds ,'psi':psi}
        psitable = psitable.append( groupentry, ignore_index=True)
         
    psis=sum(psitable["psi"])
    lastentry = {'psi':psis}
    psitable = psitable.append(lastentry, ignore_index=True)
    print (psitable.iloc[0:10,:])

    print(Fore.RED + "PSI is " + str(psis))
    return (psitable)

def test():
    print ('a')
    
# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)