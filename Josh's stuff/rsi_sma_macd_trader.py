import pandas as pd


# Calculating price change
def price_change(days, close):
    df = close.pct_change(periods=days)
    df = df.to_numpy()
    return df

# Adding 1-14 day price change columns to df2
def profit_columns(df2, test_period):
    df2 = pd.DataFrame(df2)
    temp_dict = {}
    for day in range(test_period):
        day += 1
        temp = price_change(day, df2)
        temp_dict[str(day) + '_day'] = temp
    for k, v in temp_dict.items():
        df2[k] = v

    return df2


# Adding ROI columns
def roi_calc(df2, test_period):
    roi_array = []
    for index, row in df2.iterrows():
        total = 0
        for day in range(test_period):
            day += 1
            str_day = str(day) + '_day'
            total += row[str_day]
        initial_inv = row['Close']
        mean = total/test_period
        avg_return = mean
        roi = avg_return/initial_inv
        roi_array.append(roi)
    df2['roi'] = roi_array
    return df2

# Adding upper and lower barriers to df2
def barrier_columns(df2):
    upper = 0.1
    lower = -0.1
    df2['Upper Barrier'] = upper
    df2['Lower Barrier'] = lower
    return df2

# Assigning buy, sell and hold to each row. Also adding rsi, macd's and sma's to dataframe
# Returns dataframe containing all values and buy/sell labels
def final_dataframe(close, indicator_list): # finds action points for stock
    test_period = 14
    # action_array = []  # New array that can be added to the final dataframe
    df2 = profit_columns(close, test_period)
    # df2 = barrier_columns(df2)
    for indicator in indicator_list:
        df2[indicator.name] = indicator
    df2 = roi_calc(df2, test_period)
    df2 = df2.dropna()

    # for index, row in df2.iterrows():
    #     action = 0
    #     for i in range(test_period):
    #         day = test_period - i
    #         str_day = str(day) + '_day'
    #         if row[str_day] > row['Upper Barrier']:
    #             action = 1
    #         elif row[str_day] < row['Lower Barrier']:
    #             action = -1
    #     action_array.append(action)

    for day in range(test_period):
        day += 1
        str_day = str(day) + '_day'
        df2 = df2.drop([str_day], axis=1)

    # df2 = df2.drop(['Upper Barrier', 'Lower Barrier'], axis=1)
    # df2['action'] = action_array

    return df2
