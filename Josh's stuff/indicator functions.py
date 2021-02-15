
# Returns zero_crossings array
def macd_crossing(macd_signal, macd): # identifying where macd shows buy/sell signal
    macd_diff = macd_signal - macd
    macd_diff = macd_diff.to_numpy()
    zero_crossings = np.where(np.diff(np.sign(macd_diff)))[0]

    return zero_crossings


# Returns action_index_dict
def macd_actions(zero_crossings, macd, macd_signal): # determines buy and sell positions
    action_index_dict = {}
    for k in zero_crossings:
        if macd[k] > macd_signal[k]:
            action_index_dict[k] = 'sell'
        else:
            action_index_dict[k] = 'buy'

    return action_index_dict


# Returns action_index_dict (possibilty of previous buy/sell being overridden)
def rsi_actions(rsi, action_index_dict): # identifying where rsi shows buy/sell signal
    rsi_buy_lim = 70
    rsi_sell_lim = 30
    for i in range(len(rsi)):
        if rsi[i] > rsi_buy_lim: # recording a buy location
            action_index_dict[i] = 'buy'
        elif rsi[i] < rsi_sell_lim: # recording a sell location
            action_index_dict[i] = 'sell'

    return action_index_dict


# Returns zero_crossings array
def sma_crossing(slow_sma, fast_sma): # identifying where macd shows buy/sell signal
    sma_diff = slow_sma - fast_sma
    sma_diff = sma_diff.to_numpy()
    zero_crossings = np.where(np.diff(np.sign(sma_diff)))[0]

    return zero_crossings


# Returns action_index_dict (possibilty of previous buy/sell being overridden)
def sma_actions(slow_sma, fast_sma, action_index_dict, zero_crossings):
    for k in zero_crossings:
        if slow_sma[k] > fast_sma[k]:
            action_index_dict[k] = 'sell'
        else:
            action_index_dict[k] = 'buy'

    return action_index_dict