import numpy as np

# Returns zero_crossings
def macd_crossing(macd_signal, macd): # identifying where macd shows buy/sell signal
    macd_diff = macd_signal - macd

    zero_crossings = np.where(np.diff(np.sign(macd_diff)))[0]
    zero_crossings = zero_crossings

    return zero_crossings


# Returns action_index_dict
def locate_action(zero_crossings, macd, macd_signal): # determines buy and sell positions
    action_index_dict = {}
    for k in zero_crossings:
        if macd[k] > macd_signal[k]:
            action_index_dict[k] ='sell'
        else:
            action_index_dict[k] = 'buy'

    return action_index_dict

# Returns action_index_dict
def basic_trader(macd, macd_signal): # finds action points for stock
    zero_crossings = macd_crossing(macd_signal, macd)
    action_index_dict = locate_action(zero_crossings, macd, macd_signal)

    return action_index_dict