import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import scipy
import copy
import time
import datetime
from PIL import Image
import requests
from io import BytesIO

print("alpha funcs imported")

#==================================================================#
# Alpha infrastructure implementation Ver 2
#==================================================================#

#==================================================================#
# Elementary transformers
#==================================================================#
# fast neutralize
def fnn(alpha):
    return normalize(neutralize(alpha))

# Task 1 : Neutralize
# returns alpha-vector with zero mean
def neutralize(alpha):
    # create copy to avoid rewrite the original alpha
    c_alpha = alpha - alpha.mean()
    assert sum(c_alpha) < 10**(-6), f"Couldn't neutralize the alpha : {sum(c_alpha)}!"
    return c_alpha

# Task 2 : Normalize
# returns alpha-vector such sum( |ai| ) = 1
def normalize(alpha):
    # create copy to avoid rewrite the original alpha
    c_alpha = alpha / sum(abs(alpha))
    assert (sum(abs(c_alpha)) - 1) < 10**(-8), f"Couldn't neutralize the alpha : {(sum(abs(c_alpha)) - 1)}!"
    return c_alpha

# Task 3 : Truncate
# It makes boundary to alpha components that  abs(ai) < weight:
# XXX : truncate operation can rebalance the alpha - what should we do???
def truncate(alpha, max_weight, n_ops = 10, drop_size=0.9):
    c_alpha = copy.copy(alpha)
    for i in range(n_ops):
        for (i, a) in enumerate(c_alpha):
            # if ai is bigger that max_weight 
            if abs(a) > max_weight * drop_size:
                #print('trunc', a)
                c_alpha[i] = max_weight * drop_size
                #print('trunced', alpha[i])

        c_alpha = neutralize(c_alpha)
        c_alpha = normalize(c_alpha)
        #print(f"Processed alpha = {alpha}")
    # Make sure we alpha is still alpha
    s = [abs(a) > max_weight * drop_size for a in c_alpha] 
    #print(s)
    assert sum(s) == 0, "Couldn't truncate the alpha!"
    return c_alpha

# Task 5: Rank
def rank(alpha):
    assert len(alpha) > 1, "Alpha cannot be a scalar!"
    # create a copy to avoid corrupting the orig alpha
    c_alpha = copy.copy(alpha)
    return (rankdata(c_alpha) - 1) / (len(c_alpha) - 1)

# Task 6: CutOutliers
def cut_outliers(alpha, out_n):
    # check that a number for removal is correct
    assert out_n >= alpha.any(), "Attempt to cut more elements than alpha consists of!"
    c_alpha = copy.copy(alpha)
    # for each n make max|min equal zero 
    for i in range(out_n):
        argmax, argmin = c_alpha.argmax(), c_alpha.argmin()
        c_alpha[argmax], c_alpha[argmin] = 0, 0
    return c_alpha

# Task 7: CutMiddle
def cut_middle(alpha, mid_n):
    # check that a number for removal is correct
    assert mid_n >= alpha.any(), "Error in cut_middle operation"
    c_alpha = copy.copy(alpha)
     # for each n make mid equal zero 
    for i in range(mid_n):
        argmid = abs(c_alpha).argmin()
        # set the middle values 1 so we can use conditional transform 
        c_alpha[argmid] = 1 
    return np.where((c_alpha == 1), 0, c_alpha)

# Task 8: ApplyFunction
def apply_func(alpha, func):
    # vectorize wasn't effective & numpy allows us to apply func directly to
    # each component of the vector. just make it as an np.array
    c_alpha = np.asarray(alpha)
    return func(alpha)

# Task 13: Decay
def decay(alphas, k):
    assert len(alphas) >= k, "Error in decay operation"
    c_alpha = copy.copy(alphas[0])
    for i in range(1, k):
        c_alpha += (k - i) / k * alphas[i]
    c_alpha = normalize(neutralize(c_alpha))
    return c_alpha
    

# Task 4 : Return. Assume we have pandas dataframe table with assets i:
# Actually it isn't being used
def get_return(table, ticker, day, shift=1, dividends = False):
    # if we have dividends then add dividends to return
    if dividends:
        return (table.iloc[day, ticker] - table.iloc[day - shift, ticker] +\
               get_dividends(table, ticker, day)) / table.iloc[day - shift, ticker]
    else:
        return (table.iloc[day, ticker] - table.iloc[day - shift, ticker]) /\
            table.iloc[day - shift, ticker]
    
# Since we define the function earlier. Just return 0 for now
def get_dividends(table, ticker, day):
    return 0

# the simpliest implementation since we actually could use instead of former
def simple_return(price_today, price_yesterday):
    return (price_today - price_yesterday) / price_yesterday

#==================================================================#

# Task 9: Turnover
def turnover(alphas, aggregated_by_year=True):
    ''' Takes alpha matrix with dates and returns 
    '''
    diff_table = alphas.drop(columns='date').diff().reset_index()
    diff_table.columns = alphas.columns
    diff_table['date'] = alphas['date']
    turnover_table = pd.DataFrame(data=np.array([diff_table.iloc[i,1:].abs().sum() for i in range(1, len(diff_table['date']))]))
    turnover_table = turnover_table.reset_index()
    turnover_table.columns = ['date', 'turnover']
    acc = alphas['date'][1:].reset_index()
    turnover_table['date'] = acc['date']
    turnover_table['year'] = turnover_table['date'].apply(lambda x: x.year)
    if aggregated_by_year :
        return turnover_table.drop(columns='date').groupby(by='year').mean()
    else :
        return turnover_table.drop(columns='year')
        
 #==================================================================#
# Complex transformers for tickers dataframe
#==================================================================#

# Task 11: Return
def get_returns_table(table, aggregated_by_year=False):
    ''' Takes ticker table with dates
        Shifts the table to 1 then calc difference and divide pointwise
    '''
    a, b = table.drop(columns='date').diff(), table.drop(columns='date')
    c = a.div(b).reset_index()
    
    # return date column to returns table
    c.columns = np.append('date', c.columns[1:])
    c['date'] = table['date']
    c = c.iloc[1:]                            # drop NaN line with start date
    c = c.reset_index().drop(columns='index') # reset index
    if aggregated_by_year:
        c['year'] = c['date'].apply(lambda x : x.year)
        return c.drop(columns='date').groupby(by='year').cumsum()
    return c

def get_pnls(table, alphas, aggregated_by_year=False):
    ''' Takes N+1 x T table and N x T-1 alphas 
        since we have extra col date and calculate with shift on time 
        Returns pandas table
    '''
    # check that we have eq on tickers and +1 on time shift
    #print(alphas.shape, table.shape)
    assert alphas.shape[0] == (table.shape[0]-1) and\
        alphas.shape[1] == (table.shape[1]-1), "Not compatible dimensions!"
    # get transformed table with returns:
    
    ret_table = get_returns_table(table)
    # OK, we have ret_table N+1 x T-1
    # get scalar multiplication of alpha and take diagonal 
    # since we need pnl of all tickers over time (=> (1 x time) vector)
    pnl_table = np.dot(alphas, ret_table.drop(columns='date').T).diagonal()
    # create dataframe and drop index to create first date column
    pnl_table = pd.DataFrame(data=pnl_table).reset_index()
    pnl_table.columns = ['date', 'pnl']
    pnl_table['date'] = ret_table.reset_index()['date']
    
    if aggregated_by_year:
        ag_pnl_table = pnl_table
        ag_pnl_table['year'] = pnl_table['date'].apply(lambda x : x.year)
        return ag_pnl_table.drop(columns='date').groupby(by='year').sum(), pnl_table
    return pnl_table

# Task 10: Drawdown
def drawdown(table, alphas, update_on_year = True, plot=False):
    '''Takes table and alphas and returns the dictionary that contains
        info about max|min pnl for alphas during the each year of period
        Return dictionary with year -> stats
    '''
    # check that we have the same alphas and table sizes
    assert alphas.shape[0] == (table.shape[0]) and\
        alphas.shape[1] == (table.shape[1]), "Not compatible dimensions!"
    # calculate cumulative pnls, drop data and last alpha (we dont need the last alpha)
    cum_pnls = get_pnls(table, alphas.drop(columns='date').iloc[:-1]) 
    cum_pnls['cum_pnl'] = cum_pnls['pnl'].cumsum()
    # set max cumulative PnL, min drawdown, pnls over time
    max_drawdown, max_pnl, min_pnl = 0, cum_pnls['cum_pnl'][0], cum_pnls['cum_pnl'][0]
    
    # create the dictionary to store years, max/min values and dates
    info_dict = {}
    # Rabbit
    get_trolled = False
    years = np.append(cum_pnls.date[0].year, [(cum_pnls.date[0].year + year) for year in\
             range(1 + cum_pnls.date[len(cum_pnls.date)-1].year - cum_pnls.date[0].year)])
    for year in years:
        info_dict.update( { year : {} } )
        info_dict[year].update({'date':cum_pnls.date[0], 'max_pnl': cum_pnls['cum_pnl'][0],\
                                'min_pnl': cum_pnls['cum_pnl'][0], 'max_pnl_drd': cum_pnls['cum_pnl'][0],\
                                'max_drawdown': 0})
    if plot: plt.figure(figsize=(8,6)); plt.xlabel('Year', fontsize=14); plt.ylabel('cumulative PnL', fontsize=14)
    # go for loop over timeline: start from 0-day and go forward till the end
    for year in info_dict:
        if update_on_year:
            # If update on year -> get updated pnls and retaken alphas
            y_pnls = table[table.date >= pd.to_datetime(f'{year}-01-01')]
            y_pnls = y_pnls[y_pnls.date <= pd.to_datetime(f'{year}-12-31')]
            y_pnls = y_pnls.reset_index().drop(columns='index')
            
            new_alphas = alphas[alphas.date >= pd.to_datetime(f'{year}-01-01')]
            new_alphas = new_alphas[new_alphas.date <= pd.to_datetime(f'{year}-12-31')]
            new_alphas = new_alphas.reset_index().drop(columns='index')

            #print(new_alphas.shape, y_pnls.shape)
            date_before = y_pnls['date'][0]
            y_pnls = get_pnls(y_pnls, new_alphas.drop(columns='date').iloc[:-1])
            y_pnls['cum_pnl'] = y_pnls['pnl'].cumsum()
            
            info_dict[year]['date'] = y_pnls['date'][0]
            info_dict[year]['max_pnl'] = y_pnls['cum_pnl'][0]
            info_dict[year]['min_pnl'] = y_pnls['cum_pnl'][0]
            info_dict[year]['max_pnl_drd'] = y_pnls['cum_pnl'][0]
            info_dict[year]['max_drawdown'] = 0
            
        else:
            # update annual pnl table and reset index to avoid errors
            y_pnls = cum_pnls[cum_pnls.date >= pd.to_datetime(f'{year}-01-01')]
            y_pnls = y_pnls[y_pnls.date <= pd.to_datetime(f'{year}-12-31')]
            y_pnls = y_pnls.reset_index().drop(columns='index')
        # update stats for each year
            
        for day_info in y_pnls.values:
            if day_info[2] < info_dict[year]['min_pnl']:
                info_dict[year]['min_pnl'] = day_info[2]
            if day_info[2] > info_dict[year]['max_pnl']:
                info_dict[year]['max_pnl'] = day_info[2]
            elif (info_dict[year]['max_pnl'] - day_info[2]) / info_dict[year]['max_pnl'] > info_dict[year]['max_drawdown']:
                info_dict[year]['max_drawdown'] = (info_dict[year]['max_pnl'] - day_info[2]) / info_dict[year]['max_pnl']
                info_dict[year]['max_pnl_drd'] = info_dict[year]['max_pnl']
                info_dict[year]['date'] = day_info[0]            
        if info_dict[year]['max_pnl'] <= 0 : 
            info_dict[year]['max_pnl'] = 0
            info_dict[year]['max_pnl_drd'] = 0 
            info_dict[year]['max_drawdown'] = info_dict[year]['min_pnl']
            info_dict[year]['date'] = y_pnls[y_pnls['cum_pnl'] == info_dict[year]['min_pnl']]['date'].item()
            get_trolled = True
             
        if plot:
            f_line = pd.DataFrame([[date_before, 0, 0]], columns=y_pnls.columns)
            y_pnls = pd.concat([f_line, y_pnls])
             
            plt.plot(y_pnls['date'], y_pnls['cum_pnl'], c='b')#, label='historic alphas pnl')
            plt.scatter(info_dict[year]['date'], y_pnls[y_pnls['date']==info_dict[year]['date']]['cum_pnl'], c='r')
            #plt.scatter(y_pnls[y_pnls['cum_pnl']==info_dict[year]['max_pnl']]['date'], info_dict[year]['max_pnl'], c='b')
            plt.scatter(y_pnls[y_pnls['cum_pnl']==info_dict[year]['max_pnl_drd']]['date'], info_dict[year]['max_pnl_drd'], c='g')
            
            strt = list(y_pnls[y_pnls['cum_pnl']==info_dict[year]['max_pnl_drd']]['date'])
            while strt[len(strt)-1] != info_dict[year]['date']:
                strt.append(strt[len(strt)-1] + datetime.timedelta(days=1))
            x = np.asarray(strt)
            y = np.linspace(y_pnls[y_pnls['date']==info_dict[year]['date']]['cum_pnl'], info_dict[year]['max_pnl_drd'], int(((info_dict[year]['max_pnl_drd']) - (y_pnls[y_pnls['date']==info_dict[year]['date']]['cum_pnl']))*10000))
            xy = np.meshgrid(x,y)
            plt.scatter(xy[0], xy[1], alpha=0.005,s=10, c='r')
            plt.title('Drawdown for each year (PnL starts from 0%)')
            plt.grid()
            
    
    if get_trolled:
        plt.show()
        plt.figure(figsize=(8,6))
        response = requests.get('https://sun9-60.userapi.com/impf/PYWv4FxU1yPYu2i25AcKAWlvXfIt_N9fZUypQA/gilgSNvLTeM.jpg?size=1207x447&quality=96&sign=0f17f09f0e2162387e60372c94f53da6&c_uniq_tag=moLtPV_jastzaRH_g5Tqc-53FFcQh21FfhEQGevg0Rs&type=album')
        img = Image.open(BytesIO(response.content))
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.show()

    return info_dict
    
# Task 12 : Sharpe
def sharpe(table, alphas):
    '''Takes 2 pandas dt and return sharpe ratio aggr.by year
    '''
    pnls = get_pnls(table, alphas)
    pnls['year'] = pnls['date'].apply(lambda x : x.year)
    agg_ret = pnls.drop(columns='date').groupby(by='year').\
        apply(lambda x : x.mean() / x.std()).drop(columns='year')
    return agg_ret

# Task 15 Returns Correlation
def get_corr(*returns):
    '''Takes 2 or more vectors to calc correlation'''
    returns = np.array(returns)
    return pd.DataFrame(data=returns).T.corr(method='pearson')

# Task 14 : Alpha Stats
def alpha_stats(table, alphas, plot_drd = False):
    '''Takes initial ticker table, alphas 
        Returns nothing - print stats and plot'''
    # Get stats: sharpe, pnls, turnover, drawdown
    srp = sharpe(table, alphas.drop(columns='date')[:-1])
    ag_pnls, pnls = get_pnls(table, alphas.drop(columns='date')[:-1], aggregated_by_year=True)
    trn = turnover(alphas, aggregated_by_year=True)
    drd = drawdown(table, alphas, update_on_year = True, plot=plot_drd)
    for i in range(len(srp)):
        print(f'{srp.iloc[i].name:} : sharpe={srp.iloc[i].pnl:.5f}, turnover={trn.iloc[i].turnover:.5f}, cum_pnl={ag_pnls.iloc[i].pnl:.5f}, drawdown={drd[srp.iloc[i].name]["max_drawdown"]*100:.3f}%')
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.plot(pnls['date'], pnls['pnl'].cumsum())
    plt.title('Alphas historic cumulative pnl', fontsize=16)
    plt.xlabel('Year', fontsize=14); plt.ylabel('cumulative PnL', fontsize=14)
    plt.show()
    
