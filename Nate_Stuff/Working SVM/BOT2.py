# This version will use data from the CryptoCompare API and will therefore
# work exclusively on cryptoassets
# it will make calls every minute... might have to change that eventually

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.svm import SVC
import ta
from colorama import Fore, Style

# import other work
from BOT_utils import *
from CryptoCompare import *

#set up the API:
KEY = "b14d953abbd87596c1c1c75e5c2318eea939bc1ddb17a113355b8a4cca312488"
cc.cryptocompare._set_api_key_parameter(KEY)

def loop():
    pass

@logger.catch
def main():

    # First, we import the training data:
    df = get_cc_minute_prices('BTC')




if __name__ == '__main__':
    main()
