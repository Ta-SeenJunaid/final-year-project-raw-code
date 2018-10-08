import datetime as dt 
import pandas as pd 
import pandas_datareader.data as pdr 

date_entry_start = input('Enter start date in YYYY-MM-DD format: ')
year, month, day = map(int,date_entry_start.split('-'))
start = dt.date(year, month, day)

date_entry_end = input('Enter end date in YYYY-MM-DD format: ')
year, month, day = map(int,date_entry_end.split('-'))
end = dt.date(year, month, day)


ticker = input('Enter company ticker name: ')
df = pdr.DataReader(ticker, 'yahoo', start, end)
df.to_csv('testing.csv')