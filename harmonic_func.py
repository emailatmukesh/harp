import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

X_time=[]
A_time=[]
B_time=[]
C_time=[]
D_time=[]

timer=[]

def peak_detect(price,Low1,High1,zz):
    # Find our relative extrema
    # Return the max indexes of the extrema
    max_idx = list(argrelextrema(High1, np.greater, order=zz)[0])
    # Return the min indexes of the extrema
    min_idx = list(argrelextrema(Low1, np.less, order=zz)[0])
    idx = max_idx + min_idx + [len(price) - 1]
    idx.sort()
    current_idx = idx[-5:]

    start = min(current_idx)
    end = max(current_idx)

    current_pat = price[current_idx]
    return current_idx, current_pat, start, end

def is_Gartley(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)

    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           return 1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    elif XA<0 and AB>0 and BC<0 and CD>0:
        # AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        # BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        # CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            return -1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    else:
        return np.NaN

def is_Butterfly(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)

    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           return 1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    elif XA<0 and AB>0 and BC<0 and CD>0:
        # AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        # BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        # CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            return -1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    else:
        return np.NaN

def is_Bat(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_range = np.array([0.382 - err_allowed, 0.5 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)

    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           return 1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    elif XA<0 and AB>0 and BC<0 and CD>0:
        # AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        # BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        # CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            return -1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    else:
        return np.NaN

def is_Crab(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    AB_range = np.array([0.382 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([2.24 - err_allowed, 3.618 + err_allowed]) * abs(BC)

    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
           return 1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    elif XA<0 and AB>0 and BC<0 and CD>0:
        # AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        # BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        # CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            return -1
            # plt.plot(np.arange(start, i+15), price.values[start:i+15])
            # plt.scatter(idx, current_pat, c='r')
            # plt.show()
        else:
            return np.NaN
    else:
        return np.NaN
    
    
    
def walk_forward(price,sign,slippage=4,stop=10):
    
    slippage=float(slippage)/float(10000)
    stop_amount=float(stop)/float(10000)
    
    
    if sign== 1:
        #X1_time=data.iloc[np.array(current_idx)[0]].name
        #X_time=np.append(X_time,X1_time)
       # print(X_time)
        initial_stop_loss=price[0]-stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            if move >0  and (price[i]-stop_amount)> initial_stop_loss:
                stop_loss=price[i]- stop_amount
            elif price[i]< stop_loss:
                return stop_loss -price[0]-slippage
            
            
    elif sign== -1:
        initial_stop_loss=price[0]+stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            if move <0  and (price[i] + stop_amount) < initial_stop_loss:
                stop_loss=price[i]+ stop_amount
            elif price[i]> stop_loss:
                return price[0]- stop_loss - slippage
                
            
def time_exit(current_idx, data):

       # dateend1=data.iloc[end].name
        #dateend=np.append(dateend,dateend1)
        
        #dateend2=data.iloc[start].name
        #datestart=np.append(datestart,dateend2)
        
        global X_time
        global D_time
        X1_time=data.iloc[np.array(current_idx)[0]].name
        X_time=np.append(X_time,X1_time)
        
        A1_time=data.iloc[np.array(current_idx)[1]].name
        A_time=np.append(X_time,A1_time)
        
        B1_time=data.iloc[np.array(current_idx)[2]].name
        B_time=np.append(X_time,B1_time)
        
        C1_time=data.iloc[np.array(current_idx)[3]].name
        C_time=np.append(X_time,C1_time)
        
        D1_time=data.iloc[np.array(current_idx)[4]].name
        D_time=np.append(D_time,D1_time)
        
        #print(C1_time-B1_time)
        ## time difference
        
        seconds = B1_time.timestamp()
        
        #print(seconds)
       
        XA_td= (A1_time-X1_time)
        AB_td= B1_time-A1_time
        
        BC_td= C1_time-B1_time
        CD_td= D1_time-C1_time
    
        final= XA_td + D1_time
        #print(final)
        trade_kill=data.loc[data['time']==final]
        money=trade_kill.Close[0]
        
        trade_init=data.loc[data['time']==D1_time]
        ini_money=trade_init.Close[0]
        
        
        profit=  money - ini_money
        
        return profit
    
    
    
    
def candle_exit(current_idx):

       x1=current_idx[0]
       a1=current_idx[1]
       d1=current_idx[4]
       diff=a1-x1
       for tm in range(1,16):                   ## time multipliar 
           predicted_d= a1+ float(tm/4)*diff
           if predicted_d -3 < d1 < predicted_d + 3:
               return True
               break
           
            
def action_reaction(current_idx):

       x1=current_idx[0]
       print(x1, 'index')
       a1=current_idx[1]
       d1=current_idx[4]
       diff=a1-x1
       for tm in range(1,4):                   ## time multipliar 
           predicted_d= a1+ tm*diff
           if predicted_d -2 < d1 < predicted_d + 2:
               return True
               break
           
       

       
def last_candle(df):
    if df['Close'].values[-1] > df['Open'].values[-1]:
        return 'green'
    elif df['Close'].values[-1] <= df['Open'].values[-1]:
        return 'red'
    
def sma(df,a):
    df['sma']=df["Close"].rolling(a,min_periods=1).mean()
    da= df['sma'].values[-1]
    if da < df['Close'].values[-1]:
        return 'long'
    elif da >= df['Close'].values[-1]:
        return 'sell'
   
def lines(sign,price_high,price_low,current_pat):
    
    
    price_diff= abs(current_pat[3] - current_pat[4])*0.3
    entry_price=current_pat[4]
    target1= current_pat[3]
    
    
    if sign== 1:
        stop_amount= current_pat[4] - price_diff
        
        #print()
        #entered_price= current_pat[4]
        for q in range(1,len(price_low)):
           
            if  price_high[q] >= target1:
                exit1= price_high[q]
                amount= exit1 - entry_price
                #print(entry_price,'entry price')
                #print('exit=',exit1)
                return float(amount)
              #  return  exit1
                break
            
            
            elif  price_low[q]<= stop_amount:
                  exit1= price_low[q]
                  amount= exit1 - entry_price
                  print(entry_price,'entry price')
                  print('exit=',exit1)
                  return float(amount)
               # return  exit1
                  break
              
    
            
           
                
        
    elif sign== -1:
        stop_amount= current_pat[4] + price_diff
        
        for q in range(1,len(price_low)):
           
            if  price_low[q] <= target1:
                exit1= price_low[q]
                amount= entry_price-exit1
                #print(entry_price,'entry price')
                #print('exit=',exit1)
                return float(amount)
                break
               # return  exit1
                
            
            
            elif  price_high[q] >= stop_amount:
                exit1= price_high[q]
                amount= entry_price-exit1
                #print(entry_price,'=entry price')
                #print('exit=',exit1)
                return float(amount)
                break
                #return  exit1
                



    #slope=0 y=mx+c  y=c  
    

def walk_forward_dy(price,sign,current_pat):
    
    #slippage=float(slippage)/float(10000)
    #stop_amount=float(stop)/float(10000)
    
    stop_amount= abs(current_pat[1] - current_pat[4])*0.4
    slippage= abs(current_pat[1] - current_pat[4])*0.2
    
    #entry_price=current_pat[4]
    #target1= current_pat[1]
    
    
    if sign== 1:
        #X1_time=data.iloc[np.array(current_idx)[0]].name
        #X_time=np.append(X_time,X1_time)
       # print(X_time)
        
        initial_stop_loss=price[0]-stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            if move >0  and (price[i]-stop_amount)> initial_stop_loss:
                stop_loss=price[i]- stop_amount
            elif price[i]< stop_loss:
                return stop_loss -price[0]-slippage
            
            
    elif sign== -1:
        initial_stop_loss=price[0]+stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            if move <0  and (price[i] + stop_amount) < initial_stop_loss:
                stop_loss=price[i]+ stop_amount
            elif price[i]> stop_loss:
                return price[0]- stop_loss - slippage       
      
    
def walk_forward_target(price,sign,current_pat):

#slippage=float(slippage)/float(10000)
#stop_amount=float(stop)/float(10000)
    
    stop_amount= abs(current_pat[1] - current_pat[4])*0.4
    slippage= abs(current_pat[1] - current_pat[4])*0.2
    
    #entry_price=current_pat[4]
    target1= current_pat[1]
    
    
    if sign== 1:
        #X1_time=data.iloc[np.array(current_idx)[0]].name
        #X_time=np.append(X_time,X1_time)
       # print(X_time)
        
        initial_stop_loss=price[0]-stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            
            if price[i] >= target1:
                exit1= price[i]
                return exit1 - price[0]
            
            elif move >0  and (price[i]-stop_amount)> initial_stop_loss:
                stop_loss=price[i]- stop_amount
                
            elif price[i]< stop_loss:
                return stop_loss -price[0]-slippage
            
            
            
            
    elif sign== -1:
        initial_stop_loss=price[0]+stop_amount
        stop_loss=initial_stop_loss
        for i in range(1,len(price)):
            move= price[i]-price[i-1]
            
            if  price[i] <= target1:
                exit1= price[i]
                return price[0]-exit1
                
            elif move <0  and (price[i] + stop_amount) < initial_stop_loss:
                stop_loss=price[i]+ stop_amount
                
            elif price[i]> stop_loss:
                return price[0]- stop_loss - slippage    
    

def percentage_move(High,Low,current_idx):

#slippage=float(slippage)/float(10000)
#stop_amount=float(stop)/float(10000)

    Balance_point=High[current_idx[4]]-Low[current_idx[4]]
    if Balance_point >=0.04:
        return True
    
def Balance_point(price,current_idx):

#slippage=float(slippage)/float(10000)
#stop_amount=float(stop)/float(10000)
    c=current_idx[4]
    Bp_close= price[c-5]+price[c--4]+price[c-3]+price[c-2]+price[c-1]
    Bp_avg=float(Bp_close)/5
    
    if abs(Bp_avg-price[c]) >=0.04:
        return True
'''               
    
def trend_detect(ohlc_df,frequency,n):
    "function to assess the trend by analyzing each candle"
    
    df = ohlc_df.copy()
    df1 = pd.DataFrame()
    #df1["Close"] = df["Close"]
    df1["Open"] = df.groupby(pd.Grouper(level=0, freq=str(frequency)))["Open"].transform("first").dropna()
    df1["Low"] = (df.groupby(pd.Grouper(level=0, freq=str(frequency)))["Low"].expanding().min().droplevel(0)).dropna()
    df1["High"] = (df.groupby(pd.Grouper(level=0, freq=str(frequency)))["High"].expanding().max().droplevel(0).dropna()
    
    df1=df1
    #df=df1.copy()
    df1["up"] = np.where(df1["Low"]>=df1["Low"].shift(1),1,0)
    df1["dn"] = np.where(df1["High"]<=df1["High"].shift(1),1,0)
    if df1["Close"][-1] > df1["Open"][-1]:
        if df1["up"][-1*n:].sum() >= 0.7*n:
            return "uptrend"
    elif df1["Open"][-1] > df1["Close"][-1]:
        if df1["dn"][-1*n:].sum() >= 0.7*n:
            return "downtrend"
    else:
        return None
    
''' 
   
def trend(ohlc_df,frequency,n):
    "function to assess the trend by analyzing each candle"
    
    df = ohlc_df.copy()
    
    df1=df.resample(frequency).apply({'Low': lambda s: s.min(),'High': lambda s: s.max(),'Close':lambda s: s.mean()}).dropna() 
    df1["up"] = np.where(df1["Low"]>=df1["Low"].shift(1),1,0)
    df1["dn"] = np.where(df1["High"]<=df1["High"].shift(1),1,0)
    #if df1["Close"][-1] > df1["Open"][-1]:
    if df1["up"][-1*n:].sum() >= 0.65*n:
        return 1
   # elif df1["Open"][-1] > df1["Close"][-1]:
    elif df1["dn"][-1*n:].sum() >= 0.65*n:
        return -1
    
    
def trend_detect(ohlc_df,n):
    df = ohlc_df.copy()
    df1 = pd.DataFrame()
    df["up"] = np.where(df["Low"]>=df["Low"].shift(1),1,0)
    df["dn"] = np.where(df["High"]<=df["High"].shift(1),1,0)
    if df["Close"][-1] > df["Open"][-1]:
        if df["up"][-1*n:].sum() >= 0.7*n:
            return 1
    elif df["Open"][-1] > df["Close"][-1]:
        if df["dn"][-1*n:].sum() >= 0.7*n:
            return -1
    
def last_candle_detect(df):
    if df["Close"][-2] > df["Open"][-2]:
        if df["Close"][-1] > df["Open"][-1]:
            return 1
    elif df["Open"][-2] > df["Close"][-2]:
        if df["Open"][-1] > df["Close"][-1]:
            return -1
        
def last_candle_detect_2(df):
    if df["Close"][-1] >= df["Open"][-1]:
        return 1
    elif df["Open"][-1] >= df["Close"][-1]:
            return -1
    
def step_trend(price):

    Bp_close= price[-6]+price[--5]+price[-4]+price[-3]+price[-2]
    Bp_avg=float(Bp_close)/5
    
    if price[-1]>=Bp_avg:
        return 1
    
    elif price[-1]<Bp_avg:
        return -1
    
def data_re(ohlc_df,frequency):
    "function to assess the trend by analyzing each candle"
    
    df = ohlc_df.copy()
    
    df1=df.resample(frequency).apply({'Low': lambda s: s.min(),'High': lambda s: s.max(),'Close':lambda s: s.mean()}).dropna()
    return df1['Close']