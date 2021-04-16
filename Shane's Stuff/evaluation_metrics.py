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
    profit -= budget
    print('The algorithm made a loss ' + str(loss_counter) + ' times.')
    print('The algorithm sold stock ' + str(sell_counter) + ' times.')
    print('The algorithm made a profit of ' + str(profit))
    return profit

def max_profit(close_column,budget):
    actions =  correct_action_list(locate_action(close_column))
    return profit_calc(budget, actions, close_column)

def allocative_efficiency(predictor_profit, max_profit):
    return predictor_profit/max_profit * 100

#Import the test data set and transform it into a NumPy array
test_data = pd.read_csv('FB_test_data.csv')
test_data = test_data.iloc[:, 1].values


maximum_profit =  max_profit(test_data,1000)
