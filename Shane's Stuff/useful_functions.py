import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def locate_action(pred_prices):
  action_dict={}
  for i in range(0,len(pred_prices)-1):
    if pred_prices[i+1]>=pred_prices[i]:
      action_dict[i]= 1
    else:
      action_dict[i]= -1
  return action_dict

def correct_action_list(action_index_dict): # Processes the actions to remove: consecutive actions; starting with a sell; ending with a buy.
    prev = -1
    delete = []
    last_a = ''
    last_i = ''
    for k, v in action_index_dict.items():
        if prev == v:
            delete.append(k)
        elif k == 0:
            delete.append(k)
        else:
            prev = v
    for i in delete:
        del action_index_dict[i]
    if len(action_index_dict) !=0:
        for k, v in action_index_dict.items():
            last_a = v
            last_i = k
        if last_a == 1:
            del action_index_dict[last_i]

    return action_index_dict

def sell(old_profit, loss_counter, close_column, key, num_shares, sell_counter):
    prev_act = -1
    sell_counter += 1
    profit = close_column[key] * num_shares
    if old_profit > profit:
        loss_counter += 1
    return profit, prev_act, loss_counter, sell_counter


def buy(profit, close_column, num_shares,key):
    num_shares = profit/close_column[key]  # using share price and max budget to calculate num of shares brought
    prev_act = 1
    return num_shares, prev_act


def profit_calc(budget, action_dict,close_column):  # calculates profit using action dictionary
    profit = budget
    prev_act = -1
    sell_counter, loss_counter, num_shares = 0, 0, 0
    for key in action_dict:
        if action_dict[key] == -1 and prev_act != -1:
            profit, prev_act, loss_counter, sell_counter = sell(profit, loss_counter, close_column, key, num_shares, sell_counter)
        elif action_dict[key] == 1 and prev_act != 1:
            num_shares, prev_act = buy(profit, close_column, num_shares,key)
    nb_shares = budget/ close_column[0]
    passive_income = nb_shares*close_column[-1]
    profit -= budget
    print('The algorithm made a loss ' + str(loss_counter) + ' times.')
    print('The algorithm sold stock ' + str(sell_counter) + ' times.')
    print('The algorithm made a profit of ' + str(profit))
    print('Passive income would have been ' + str(passive_income))
    return profit

def max_profit(close_column,budget):
    actions =  correct_action_list(locate_action(close_column))
    return profit_calc(budget, actions, close_column)

def allocative_efficiency(predictor_profit, max_profit):
    return predictor_profit/max_profit * 100

#returns RSI values for each date, as a dataframe
def rsi_calc(close):
    #calculate difference in close price in 1 day
    diff = close.diff(1)
    
    # this preserves dimensions of diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=14 - 1, min_periods=14).mean()
    down_chg_avg = down_chg.ewm(com=14 - 1, min_periods=14).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# returns the macd and macd signal as data frames for a ticker
def macd_calc(close):  
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal


def scale_list(l, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

    if len(set(l)) == 1:
        return [np.floor((to_max + to_min)/2)] * len(l)
    else:
        return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]


def format_data(stock_data_column):
    stock_new = stock_data_column
    stock_new = stock_new.fillna(method='bfill')  
    stock_new =  list(stock_new.values)
    return stock_new

def clean_predictions(predictions,threshold):
    new_preds = {}
    for i in range(0,len(predictions)):
        if predictions[i] > threshold:
            new_preds[i] = 1
        else:
            new_preds[i] = -1 
    return new_preds
    
