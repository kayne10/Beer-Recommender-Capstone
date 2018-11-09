import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load Data
    beers = pd.read_csv('craft-cans/beers.csv')
    breweries = pd.read_csv('craft-cans/breweries.csv')
    # Merge Data
    df = beers.merge(breweries, left_on=beers.brewery_id, right_on=breweries['Unnamed: 0'])
    # Clean Data
    df.rename(columns={
        'key_0':'breweryID',
        'id':'beerID',
        'name_x':'beer_name',
        'name_y':'brewery_name'
    }, inplace=True)
    df.drop(['brewery_id','Unnamed: 0_y','Unnamed: 0_x'], axis=1, inplace=True)
    df['state'] = df['state'].str.lstrip()
    df.sort_values('breweryID', inplace=True)
    # df.set_index(['breweryID','beerID'], inplace=True)
    # Create new csv of clean data
    df.to_csv('craft-cans/clean.csv')
