# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:46:34 2021

@author: ffan1
"""

#==============================================================================
# Initiating
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import mplfinance as mpf

#==============================================================================
# Tab 1
#==============================================================================

def tab1():

    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 1 - Summary')    

    # Divide into columns, add buttons
    Column1, Column2 = st.columns(2)

    a1, a2, a3, a4, a5, a6, a7, a8 = st.columns(8)
    
    M1 = a1.button("1M")
    M3 = a2.button("3M")
    M6 = a3.button("6M")
    YTD = a4.button("YTD")
    Y1 = a5.button("1Y")
    Y3 = a6.button("3Y")
    Y5 = a7.button("5Y")
    MAX = a8.button("MAX")


# 1M
    @st.cache
    def GetData1m(ticker, start_date=end_date - timedelta(days=30), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=30), end_date=end_date, index_as_date = False, interval = "1d")
    if M1:      
        if ticker != '-':
            stockprice = GetData1m(ticker, start_date=end_date - timedelta(days=30), end_date=end_date, index_as_date = False, interval = "1d")
   
            fig1m, ax1m = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax1m.plot(x, y, color='green')
            # Create labels
            ax1m.set_title("Historical 1 Month Close Price ")
            ax1m.set_ylabel("Close (USD)")
            Column2.pyplot(fig1m)
     

# 3M
    @st.cache
    def GetData3m(ticker, start_date=end_date - timedelta(days=90), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=90), end_date=end_date, index_as_date = False, interval = "1d")
    if M3:      
        if ticker != '-':
            stockprice = GetData3m(ticker, start_date=end_date - timedelta(days=90), end_date=end_date, index_as_date = False, interval = "1d")
   
            fig3m, ax3m = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax3m.plot(x, y, color='green')
            # Create labels
            ax3m.set_title("Historical 3 Month Close Price ")
            ax3m.set_ylabel("Close (USD)")
            Column2.pyplot(fig3m)   
        
# 6M
    @st.cache
    def GetData6m(ticker, start_date=end_date - timedelta(days=180), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=180), end_date=end_date, index_as_date = False, interval = "1d")
    if M6:      
        if ticker != '-':
            stockprice = GetData6m(ticker, start_date=end_date - timedelta(days=180), end_date=end_date, index_as_date = False, interval = "1d")
   
            fig6m, ax6m = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax6m.plot(x, y, color='green')
            # Create labels
            ax6m.set_title("Historical 6 Month Close Price ")
            ax6m.set_ylabel("Close (USD)")
            Column2.pyplot(fig6m) 

# YTD
    @st.cache
    def GetDataYTD(ticker, start_date=end_date.replace(month=1, day=1), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date.replace(month=1, day=1), end_date=end_date, index_as_date = False, interval = "1d")
    if YTD:    
        if ticker != '-':
            stockprice = GetDataYTD(ticker, start_date=end_date.replace(month=1, day=1), end_date=end_date, index_as_date = False, interval = "1d")
  
            figYTD, axYTD = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            axYTD.plot(x, y, color='green')
            # Create labels
            axYTD.set_title("Historical Year-To-Date Close Price")
            axYTD.set_ylabel("Close (USD)")
            Column2.pyplot(figYTD)  
        
# 1Y
    @st.cache
    def GetData1Y(ticker, start_date=end_date - timedelta(days=365), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=365), end_date=end_date, index_as_date = False, interval = "1d")
    if Y1:      
        if ticker != '-':
            stockprice = GetData1Y(ticker, start_date=end_date - timedelta(days=365), end_date=end_date, index_as_date = False, interval = "1d")
  
            fig1Y, ax1Y = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax1Y.plot(x, y, color='green')
            # Create labels
            ax1Y.set_title("Historical 1 Year Close Price")
            ax1Y.set_ylabel("Close (USD)")
            Column2.pyplot(fig1Y)  
        
# 3Y
    @st.cache
    def GetData3Y(ticker, start_date=end_date - timedelta(days=1095), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=1095), end_date=end_date, index_as_date = False, interval = "1d")
    if Y3:      
        if ticker != '-':
            stockprice = GetData3Y(ticker, start_date=end_date - timedelta(days=1095), end_date=end_date, index_as_date = False, interval = "1d")
   
            fig3Y, ax3Y = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax3Y.plot(x, y, color='green')
            # Create labels
            ax3Y.set_title("Historical 3 Year Close Price")
            ax3Y.set_ylabel("Close (USD)")
            Column2.pyplot(fig3Y)  

# 5Y
    @st.cache
    def GetData5Y(ticker, start_date=end_date - timedelta(days=1825), end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=end_date - timedelta(days=1825), end_date=end_date, index_as_date = False, interval = "1d")
    if Y5:       
        if ticker != '-':
            stockprice = GetData5Y(ticker, start_date=end_date - timedelta(days=1825), end_date=end_date, index_as_date = False, interval = "1d")
  
            fig5Y, ax5Y = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            ax5Y.plot(x, y, color='green')
            # Create labels
            ax5Y.set_title("Historical 5 Year Close Price")
            ax5Y.set_ylabel("Close (USD)")
            Column2.pyplot(fig5Y)  
        

# MAX
    @st.cache
    def GetDataMAX(ticker, end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, end_date=end_date, index_as_date = False, interval = "1d")
    if MAX:       
        if ticker != '-':
            stockprice = GetDataMAX(ticker, end_date=end_date, index_as_date = False, interval = "1d")
  
            figMAX, axMAX = plt.subplots()
            x = stockprice['date']  
            y = stockprice['close']
            axMAX.plot(x, y, color='green')
            # Create labels
            axMAX.set_title("Full Timeline of Historical Close Price")
            axMAX.set_ylabel("Close (USD)")
            Column2.pyplot(figMAX)  
        
# Display data from quote table
    def GetQuote(ticker):
        return si.get_quote_table(ticker, dict_result=False)
    
    if ticker != '-':
        infotab1 = GetQuote(ticker)
        infotab1['value'] = infotab1['value'].astype(str)
        Column1.dataframe(infotab1, height=500)  
    
 


#==============================================================================
# Tab 2
#==============================================================================

def tab2():
    
    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 2 - Chart')
    st.subheader("Close Price and Trading Volume")

    # Get data
    def GetDataT2(ticker, start_date, end_date, interval):
        return si.get_data(ticker, start_date = start_date, end_date = end_date, index_as_date = True, interval = interval)
    
    # Divide into columns, add buttons
    Column1, Column2 = st.columns(2)
    Column1_type = Column1.selectbox("Select the Graph Type",("Candle","Line"))
    Column2_type = Column2.selectbox("Select Interval",("Day", "Month"))
    
    a1, a2, a3, a4, a5, a6, a7, a8 = st.columns(8)
    
    M1 = a1.button("1M")
    M3 = a2.button("3M")
    M6 = a3.button("6M")
    YTD = a4.button("YTD")
    Y1 = a5.button("1Y")
    Y3 = a6.button("3Y")
    Y5 = a7.button("5Y")
    MAX = a8.button("MAX")
    
    if M1:
        end_day = end_date
        start_day = end_day - timedelta(days=30)
    elif M3:
        end_day = end_date
        start_day = end_day - timedelta(days=90)
    elif M6:
        end_day = end_date
        start_day = end_day - timedelta(days=180)
    elif YTD:
        end_day = end_date
        start_day = end_date.replace(month=1, day=1)
    elif Y1:
        end_day = end_date
        start_day = end_day - timedelta(days=365)
    elif Y3:
        end_day = end_date
        start_day = end_day - timedelta(days=1095)
    elif Y5:
        end_day = end_date
        start_day = end_day - timedelta(days=1825)
    elif MAX:
        end_day = end_date
        start_day = datetime.date(si.get_data(ticker).index.min())
    else:
        end_day = end_date
        start_day = end_day - timedelta(days=365)
    
    # Define the interval Type
    if Column2_type == "Day":
        interval = "1d"
    elif Column2_type == "Month":
        interval = "1mo"
    else:
        interval = "1d"
    
    # Calculate stockprice for close, up, and down
    if ticker != '-':
        stockprice = GetDataT2(ticker, start_day, end_day, interval)
        up = stockprice[stockprice["close"]>stockprice["open"]]
        down = stockprice[stockprice["close"]<stockprice["open"]]
        stockprice['moving_avg'] = stockprice['close'].rolling(window=50).mean()
    
    # Part 1: Line Plot
    
    # Create the figure
        figtab2, ax2line = plt.subplots(figsize=(20, 8))
        
    # Second axis
        ax2 = ax2line.twinx()
       
    # Plot close price
        ax2.plot(stockprice["close"], label="close", color='black')
    
    # Plot moving average
        ax2.plot(stockprice.index, stockprice.moving_avg, color="blue", label="Mov Avg 50 day")
      
    # Plot trading volume 
        ax2line.bar(up.index, up.volume, width = 0.4, color = "red", alpha=0.6, label="Up")
        ax2line.bar(down.index, down.volume, width = 0.4, color = "green", alpha=0.6, label="Down")
      
    # Adjust the axis
        ax2line.set_ylim(bottom=0,top=stockprice["volume"].max()*5)
        ax2line.set_xlim(left=stockprice.index.min(),right=stockprice.index.max())
      
    # Create labels
        ax2.set_title("Stock Close and Volume")
        ax2.set_ylabel("Close (USD)")
        ax2line.set_ylabel("Volume (million)")
       
    # Location of legends
        ax2.legend(loc=1)
        ax2line.legend(loc=2)
        
    # Part 2: Candle Plot
    
    # Create the figure
        fig2candles, ax2line = plt.subplots(figsize=(20, 8))
       
    # Second axis
        ax2 = ax2line.twinx()
      
    # Plot close price
        width = 0.6
        width2 = 0.02
       
    # Plot Up data 
        ax2.bar(up.index, up.close - up.open, width, bottom = up.open, color = "red")
        ax2.bar(up.index, up.high - up.close, width2, bottom= up.close, color="grey")
        ax2.bar(up.index, up.open - up.low, width2, bottom= up.low, color="grey")
      
    # Plot Down data
        ax2.bar(down.index, down.close - down.open, width, bottom = down.open, color = "green")
        ax2.bar(down.index, down.high - down.close, width2, bottom= down.close, color="grey")
        ax2.bar(down.index, down.open - down.low, width2, bottom= down.low, color="grey")
    
    # Plot moving average
        ax2.plot(stockprice.index, stockprice.moving_avg, color="blue", label="Mov Avg 50 day")
             
    # Plot trading volume 
        ax2line.bar(up.index, up.volume, width = 0.3, color = "red", alpha=0.6, label="Up")
        ax2line.bar(down.index, down.volume, width = 0.3, color = "green", alpha=0.6, label="Down")
            
    # Adjust the axis
        ax2line.set_ylim(bottom=0,top=stockprice["volume"].max()*5)
        ax2line.set_xlim(left=stockprice.index.min(),right=stockprice.index.max())

    # Create labels
        ax2.set_title("Close Price and Trading Volume")
        ax2.set_ylabel("Close Price (USD)")
        ax2line.set_ylabel("Trading Volume (Million)")
        
    # Location of legends
        ax2.legend(loc=1)
        ax2line.legend(loc=2)
            
    # Switch between candle plot and line plot
        if Column1_type == "Candle":
            st.pyplot(fig2candles)
        elif Column1_type == "Line":
            st.pyplot(figtab2)
        else:
            st.pyplot(figtab2)



#==============================================================================
# Tab 3
#==============================================================================

def tab3():
    
    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 3 - Statistics')
    
    
    # Divide into columns, add buttons
    Column1, Column2 = st.columns(2)
    
    # Add table to show stock data
    @st.cache
    def GetStats(ticker):
        return si.get_stats(ticker)
    
    if ticker != '-':
        info = GetStats(ticker)
        info['Value'] = info['Value'].astype(str)
        Column2.dataframe(info, height=1000)
        
      # Add table to show stock data
    @st.cache
    def GetVal(ticker):
        return si.get_stats_valuation(ticker)
    
    if ticker != '-':
        info3 = GetVal(ticker)
        #info3['value'] = info3['value'].astype(str)
        Column1.dataframe(info3, height=1000)
        

    
    
#==============================================================================
# Tab 4
#==============================================================================
      

def tab4():
    
    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 4 - Financials')
    
    # Add table to show stock data
    @st.cache
    def GetIncome(ticker, yearly = True):
        return si.get_income_statement(ticker, yearly = True) 
    
    # Add table to show stock data
    @st.cache
    def GetIncome1(ticker, yearly = False):
        return si.get_income_statement(ticker, yearly = False)
    
    # Add table to show stock data
    @st.cache
    def GetBalance(ticker, yearly = True):
        return si.get_balance_sheet(ticker, yearly = True)
    
    # Add table to show stock data
    @st.cache
    def GetBalance1(ticker, yearly = False):
        return si.get_balance_sheet(ticker, yearly = False)
    
    # Add table to show stock data
    @st.cache
    def GetCash(ticker, yearly = True):
        return si.get_cash_flow(ticker, yearly = True)
    
    # Add table to show stock data
    @st.cache
    def GetCash1(ticker, yearly = False):
        return si.get_cash_flow(ticker, yearly = False)
    
    
    # Add a radio box      
    Type = st.radio("Type", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
    Time = st.radio("Select tab", ['Annual', 'Quarterly'])
    
    # Show the selected tab
    if Type == 'Income Statement':
        
        
            if Time == 'Annual':
            # Run tab 1
                st.write(GetIncome(ticker, yearly = True))
            elif Time == 'Quarterly':
            # Run tab 2
                st.write(GetIncome1(ticker, yearly = False))
        
    elif Type == 'Balance Sheet':
        
        
            if Time == 'Annual':
            # Run tab 1
                st.write(GetBalance(ticker, yearly = True))
            elif Time == 'Quarterly':
            # Run tab 2
                st.write(GetBalance1(ticker, yearly = False))
    
    elif Type == 'Cash Flow':
        
        
            if Time == 'Annual':
            # Run tab 1
                st.write(GetCash(ticker, yearly = True))
            elif Time == 'Quarterly':
            # Run tab 2
                st.write(GetCash1(ticker, yearly = False))
    

   

#==============================================================================
# Tab 5
#==============================================================================

def tab5():
    
    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 5 - Analysis')
    
    
    # Add table to show stock data
    @st.cache
    def GetAnalystInfo(ticker):
        return si.get_analysts_info(ticker)
   
    df = pd.DataFrame.from_dict(GetAnalystInfo(ticker), columns = ['Earnings Estimate'], orient = 'index')

    Earnings_Estimate = df.iloc[0,0]
    Revenue_Estimate = df.iloc[1,0]
    Earnings_History = df.iloc[2,0]
    EPS_Trend = df.iloc[3,0]
    EPS_Revisions = df.iloc[4,0]
    Growth_Estimates = df.iloc[5,0]
    
    st.write(Earnings_Estimate)
    st.write(Revenue_Estimate)
    st.write(Earnings_History)
    st.write(EPS_Trend)
    st.write(EPS_Revisions)
    st.write(Growth_Estimates)
   

    
    


#==============================================================================
# Tab 6
#==============================================================================

def tab6():
    
    # Add dashboard title and description
    st.title("Fangda Fan Small Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 6 - Monte Carlo Simulation')
    
    def monte(ticker, start_date=start_date, end_date=end_date, index_as_date = False, interval = "1d"):
        return si.get_data(ticker, start_date=start_date, end_date=end_date, index_as_date = False, interval = "1d")
         
    if ticker != '-':
        stockprice = monte(ticker, start_date=start_date, end_date=end_date, index_as_date = False, interval = "1d")

        
        daily_return = stockprice['close'].pct_change()
        daily_volatility = np.std(daily_return)
        

        time_list = [30,60,90]   
        # Add selection box
        t = st.selectbox("Select a time horizon", time_list)

        simulation_list = [200,500,1000]   
        # Add selection box
        s = st.selectbox("Select a simulation", simulation_list)

        # Setup the Monte Carlo simulation
        np.random.seed(123)
        simulations = s
        time_horizone = t

        # Run the simulation
        simulation_df = pd.DataFrame()

        for i in range(simulations):
    
            # The list to store the next stock price
            next_price = []
    
            # Create the next stock price
            last_price = stockprice['close'].iloc[-1]
    
            for j in range(time_horizone):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
    
            # Store the result of the simulation
            simulation_df[i] = next_price
            
        # Plot the simulation stock price in the future
        fig6b, ax6b = plt.subplots()
        fig6b.set_size_inches(15, 10, forward=True)
            
        plt.plot(simulation_df)
        st.pyplot(fig6b)

        # Get the ending price of the 200th day
        ending_price = simulation_df.iloc[-1:, :].values[0, ]
        
        # Price at 95% confidence interval
        future_price_95ci = np.percentile(ending_price, 5)
        
        # Value at Risk
        # 95% of the time, the losses will not be more than 16.35 USD
        VaR = stockprice['close'].iloc[-1] - future_price_95ci
        st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')



#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
    # Add select begin-end date
    global start_date, end_date
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart', 'Statistics', 'Financials', 'Analysis', 'Monte Carlo Simulation'])
    
    # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1
        tab1()
    elif select_tab == 'Chart':
        # Run tab 2
        tab2()
    elif select_tab == 'Statistics':
        # Run tab 3
        tab3()
    elif select_tab == 'Financials':
        # Run tab 4
        tab4()
    elif select_tab == 'Analysis':
        # Run tab 5
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        # Run tab 6
        tab6()
    
if __name__ == "__main__":
    run()
    
###############################################################################
# END
###############################################################################