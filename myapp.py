

import matplotlib.pyplot as plt
from harmonic_func import *
from tqdm import tqdm
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpdates
import streamlit as st
import csv
import requests

from alpha_vantage.timeseries import TimeSeries
import pandas as pd

import pandas as pd
import time




form = st.form(key='my-form')
tickerSymbol = form.text_input("Enter any ticker symbol", value='AMZN')
options = form.selectbox('Select the time-frame of candle',('Daily','1min', '5min', '15min', '30min', '60min'))
order = form.number_input('Enter minima/maxima',min_value=1, max_value=100, value=15)
err_allowed = form.number_input('Enter error % can be taken',min_value=1, max_value=100, value=15)

submit = form.form_submit_button('Submit')

st.write('Press submit to get results')

tickerSymbol.upper()
err_allowed=float(err_allowed)/100.00




if options=='1min' or '5min' or  '15min' or '30min' or '60min' :


    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol='+ tickerSymbol+'&interval=' + options+'&slice=year2month12&apikey=4KOS149K02Y0C1MC'
    
    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        df = pd.DataFrame(my_list,columns =['Date',"Open","High",'Low',"Close","Volume"])
        df=df.iloc[::-1]
        df.set_index(['Date'],inplace=True)
        data=df[:-1]
        
if options=='Daily':


    # extracting data for a single ticker
    ts = TimeSeries(key='4KOS149K02Y0C1MC', output_format='pandas')
    data = ts.get_daily(symbol=tickerSymbol, outputsize='full')[0]
    data.columns = ["Open","High",'Low',"Close","Volume"]
    data = data.iloc[::-1]
    
        
data = data.apply(pd.to_numeric, errors='coerce')
#data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S.%f')
st.write('Data from ',str(data['Date'][0])+' to ',str(data['Date'][-1]))
data = data.set_index(data['Date'])


data = data.drop_duplicates(keep=False)
price = data['Close'].copy()

Low1=data['Low'].copy()
High1=data['High'].copy()


pnl=[]
trade_dates=[]

pnl_time=[]

correct_pats=0
pats = 0

dateend=[]
datestart=[]
Low1=data['Low'].copy()
High1=data['High'].copy()
ttt=[]
plt.ion()
selling=[0]
buying=[1]
buy_prices=[]
sell_prices=[]
signal=[1]
stop1=[]
stop2=[]
t1=[]
cum_pips=[]


####df[['Date', 'Open', 'High',   'Low', 'Close']]

for i in tqdm(range(200,len(data))):
    
        data1=data[:i]
        
    
        if  signal[-1]==1:    
            current_idx, current_pat, start, end = peak_detect(price.values[:i], Low1.values[:i], High1.values[:i],order)
        
            XA = current_pat[1] - current_pat[0]
            AB = current_pat[2] - current_pat[1]
            BC = current_pat[3] - current_pat[2]
            CD = current_pat[4] - current_pat[3]
            
           
            
        
            moves = [XA, AB, BC, CD]
        
            gartley = is_Gartley(moves, err_allowed)
            butterfly = is_Butterfly(moves, err_allowed)
            bat = is_Bat(moves, err_allowed)
            crab = is_Crab(moves, err_allowed)
        
            harmonics = np.array([gartley, butterfly, bat, crab])
            labels = [
                'Gartley',
                'Butterfly',
                'Bat',
                'Crab'
            ]
            
            
            
                
            #if np.any(harmonics == 1)  or np.any(harmonics == -1):
                
                
                    #for j in range(0, len(harmonics)):
            j=0
            
            if harmonics[j] == 1 and last_candle(data1)=='green' : #and sma(data1,10)=='long':
                    pats+=1
                    
                    
                    
                    
                    
                    
                    start=np.array(current_idx).min()
                    end= np.array(current_idx).max()
                    #trend_day=trend(data[:end],'H',8)
                    
                    date=data.iloc[end].name
                    trade_dates=np.append(trade_dates,date)

                    
                    buying.append(-1)
                    signal.append(-1)
                    entry=data1["Close"].values[-1]
                    entry_date=data1["Date"].values[-1]
                    sl1b= (data1.Low.values[-3:,]).min()
                    stop1.append(sl1b)
                    sl2b= entry - abs(current_pat[1] - current_pat[4])*0.3
                    stop2.append(sl2b)
                    t1b= current_pat[3]
                    t1.append(t1b)
              
                
            elif harmonics[j] == -1 and last_candle(data1)=='red' : #and sma(data1,10)=='sell':
                    pats+=1
                    
                    
                  
                    start=np.array(current_idx).min()
                    end= np.array(current_idx).max()
                    #trend_day=trend(data[:end],'H',8)
                    
                    date=data.iloc[end].name
                    trade_dates=np.append(trade_dates,date)
                    

                    selling.append(-1)
                    signal.append(-1)
                    entry=data1["Close"].values[-1]
                    entry_date=data1["Date"].values[-1]
                    sl1s= (data1.High.values[-3:,]).min()
                    sl2s= abs(current_pat[1] - current_pat[4])*0.3 + entry 
                    t1s= current_pat[3]
                    stop1.append(sl1s)
                    stop2.append(sl2s)
                    t1.append(t1s)
                    
                   
                
        elif signal[-1]==-1:
            #print(stop1[-1])
            if buying[-1]==-1:
                if (stop1[-1]> data1["Close"].values[-1] or  stop2[-1] > data1["Close"].values[-1] or t1[-1] < data1["Close"].values[-1]):
                    profit=  data1["Close"].values[-1]- entry
                    signal.append(1)
                    buying.append(1)
                    
                    pnl=np.append(pnl,profit)
                
                    cum_pips = pnl.cumsum()
                    
                    if profit > 0:
                        correct_pats+=1
                    plt.style.use('ggplot') 
                    a_point=current_idx[0]-30
                    b_point= current_idx[-1]+70
                    exit_date= data1["Date"].values[-1]
                    exit_price= data1["Close"].values[-1]
                    
                    df=data.iloc[a_point:b_point,]    
                    df = df[['Date', 'Open', 'High',   'Low', 'Close']]
                      
                    # convert into datetime object
                    df['Date'] = pd.to_datetime(df['Date'])
                      
                    # apply map function
                    df['Date'] = df['Date'].map(mpdates.date2num)
                      
                    # creating Subplots
                    fig, ax = plt.subplots()
                      
                    # plotting the data
                    candlestick_ohlc(ax, df.values, width = 0.0,
                                     colorup = 'green', colordown = 'red', 
                                     alpha = 0.0)
                    
                    
                    xd_value=[data.Date.values[current_idx[0]], data.Date.values[current_idx[1]],data.Date.values[current_idx[2]], data.Date.values[current_idx[3]],data.Date.values[current_idx[4]] ]
                    ax.plot(xd_value,current_pat)
                    # allow grid
                    ax.grid(True)
                    ax.scatter(entry_date,entry,c='green', label='ENTRY LONG SIDE')
                    ax.scatter(exit_date,exit_price,c='red', label='EXIT LONG SIDE')
                    # Setting labels 
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                      
                    # setting title
                    plt.title(str(tickerSymbol) +'for'+ str(options) +'candles')                      
                    # Formatting Date
                    date_format = mpdates.DateFormatter('%d-%m-%Y')
                    ax.xaxis.set_major_formatter(date_format)
                    fig.autofmt_xdate()
                      
                    fig.tight_layout()
                    plt.legend()
                    st.pyplot(fig)
                      
                    # show the plot
                    plt.show()
                    
            
                    
            if selling[-1]==-1:
                    if stop1[-1]<data1["Close"].values[-1] or stop2[-1] < data1["Close"].values[-1] or  t1[-1] > data1["Close"].values[-1]:
                        profit= entry-  data1["Close"].values[-1]
                        print(profit)
                        signal.append(1)
                        selling.append(1)
                        pnl=np.append(pnl,profit)
                        exit_date= data1["Date"].values[-1]
                        exit_price= data1["Close"].values[-1]
                    
                        cum_pips = pnl.cumsum()
                        
                        if profit > 0:
                            correct_pats+=1
                        plt.style.use('ggplot') 
                        a_point=current_idx[0]-30
                        b_point= current_idx[-1]+70
                        
                        df=data.iloc[a_point:b_point,]    
                        df = df[['Date', 'Open', 'High',   'Low', 'Close']]
                          
                        # convert into datetime object
                        df['Date'] = pd.to_datetime(df['Date'])
                          
                        # apply map function
                        df['Date'] = df['Date'].map(mpdates.date2num)
                          
                        # creating Subplots
                        fig, ax = plt.subplots()
                          
                        # plotting the data
                        candlestick_ohlc(ax, df.values, width = 0.0,
                                         colorup = 'green', colordown = 'red', 
                                         alpha = 0.0)
                        
                        
                        xd_value=[data.Date.values[current_idx[0]], data.Date.values[current_idx[1]],data.Date.values[current_idx[2]], data.Date.values[current_idx[3]],data.Date.values[current_idx[4]] ]
                        ax.plot(xd_value,current_pat)
                        # allow grid
                        ax.grid(True)
                        
                        ax.scatter(entry_date,entry,  c='green', label='ENTRY SELL SIDE')
                        ax.scatter(exit_date,exit_price,  c='red', label='EXIT SELL SIDE')
                          
                        # Setting labels 
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price')
                          
                        # setting title
                        plt.title(str(tickerSymbol) +' for '+ str(options) +' candles')
                          
                        # Formatting Date
                        date_format = mpdates.DateFormatter('%d-%m-%Y')
                        ax.xaxis.set_major_formatter(date_format)
                        fig.autofmt_xdate()
                          
                        fig.tight_layout()
                        plt.legend()
                        st.pyplot(fig)
                        
                        # show the plot
                        plt.show()
                                
                     
                        
try:
    lastm=np.round(cum_pips[-1],2)
    jjjj= np.round(float(correct_pats)/float(pats),2)
    lbl= "[Accuracy = " +str(100*jjjj)+ '%]' + '[ Profit = '+ str(lastm) +' per quantity ] [Total trade = '+ str(pats)+ "] "
    #return lbl
except:
    lbl='No pattern found '
plt.clf()
plt.plot(cum_pips,label=lbl)
plt.legend()
plt.pause(0.05)

st.write('Profit and Loss')
st.write(pnl)
                     
                    
                    
                        
                        
                        
                        
                        
                        
