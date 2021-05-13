To see the RNN working, run RNN_bt_h.py
- This stands for 'RNN Backtest Hour'

If you wish to experiment with the vision windows epoch number, or test size, you can edit them at the top of the main function.
The output will be a comparitive Net Worth Graph.

You can also adapt the strategy:
 - Strat = -1 means the simulation shorts and longs
 - Strat = 0 means the simulation only longs
 - Risk = 0.9 means the simulation will only risk 90% of its net worth on each trade

You can also adapt the starting balance if you like.

To adapt the time periods trained and tested on, change the value of the 'offset' variable, which will move the data one month back in the past.