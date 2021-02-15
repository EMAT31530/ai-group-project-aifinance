import random
from operator import attrgetter
from math import log10, floor
import pandas as pd


def two_sig(x):
    if x == 0:
        return x
    else:
        return round(x, 2-int(floor(log10(abs(x))))-1)


class Chromosome:
    def __init__(self, fitness, indicator_list):
        self.genes_val = []
        self.genes_name = []
        self.fitness = fitness
        for indicator in indicator_list:
            if indicator.max() < 15:
                self.genes_val.append(round(random.uniform(indicator.min(), indicator.max()), 1))
            else:
                self.genes_val.append(random.randint(indicator.min(), indicator.max()))
            self.genes_name.append(indicator.name)


class Init_Pop:
    def __init__(self, pop_size, indicator_list):
        self.chromosomes = []
        for i in range(pop_size):
            self.chromosomes.append(Chromosome(0, indicator_list))


def fit_calc(pop, df2, action):
    for chromo in pop.chromosomes:
        total = 0
        result_counter = 0
        df = df2.loc[df2[chromo.genes_name[0]] == chromo.genes_val[0]]
        for i in range(len(chromo.genes_name)-1):
            i += 1
            df = df.loc[df2[chromo.genes_name[i]] == chromo.genes_val[i]]
        for index, row in df.iterrows():
            if action == 'buy':
                total += row['roi']
            elif action == 'sell':
                total -= row['roi']
            result_counter += 1
        if result_counter != 0:
            roi = total/result_counter
        else:
            roi = total
        chromo.fitness = roi


def selection(pop, pop_size):
    chromos = pop.chromosomes
    sorted_chromos = sorted(chromos, key=attrgetter('fitness'))
    while len(sorted_chromos) > pop_size:
        sorted_chromos.remove(sorted_chromos[0])  # Removing the weakest chromosome
    pop.chromosomes = sorted_chromos
    return pop


def order(pop):
    chromos = pop.chromosomes
    sorted_chromos = sorted(chromos, key=attrgetter('fitness'))
    pop.chromosomes = sorted_chromos
    return pop


def remove_double(pop):
    chromos = pop.chromosomes
    done = 0
    removal = 0
    while done == 0:
        done = 1
        for i in range(len(chromos)-1):
            if chromos[i].genes_val == chromos[i+1].genes_val and done == 1:
                removal = chromos[i]
                done = 0
        if done == 0:
            chromos.remove(removal)
    return pop


def mutation(pop, indicator_list):
    mutate_prob = 0.05
    chromos = pop.chromosomes
    up_range = round(1 / mutate_prob)
    for chromo in chromos:
        for i in range(len(chromo.genes_val)):
            guess = random.randint(1, up_range)  # Create a range giving the probability of 'mutate_prob' of getting 1
            if guess == 1:  # If guess is 1 then probability is met
                temp = Chromosome(0, indicator_list)  # Create a new chromosome to donate a new gene
                for it in range(len(chromo.genes_val)):  # Loop through all genes changing all to the original chromosomes
                    if it != i:                      # apart from the mutated gene
                        temp.genes_val[it] = chromo.genes_val[it]  # Change the gene in the new chromosome to the donor gene
                pop.chromosomes.append(temp)  # Append mutated chromosome
    return pop


def crossover(pop, indicator_list):
    cross_prob = 0.4
    chromos = pop.chromosomes
    random.shuffle(chromos)
    num_cross = cross_prob * len(chromos)
    num_cross = round(num_cross)
    if num_cross % 2 != 0:  # Ensuring number of chromosomes used is a multiple of 2
        num_cross -= 1
    for i in range(int(num_cross/2)):
        one = Chromosome(0, indicator_list)  # Create new offspring chromosome
        rand_ind = random.randint(0, len(one.genes_val)-1)  # Picks a random index for gene swap
        for it in range(len(one.genes_val)):
            one.genes_val[it] = chromos[i*2].genes_val[it]
        one.genes_val[rand_ind] = chromos[i*2 + 1].genes_val[rand_ind]
        pop.chromosomes.append(one)  # both new chromosomes are added to pop
    return pop


def cycle(pop, it_num, indicator_list, df2, pop_size, action):
    for i in range(it_num):
        pop = crossover(pop, indicator_list)
        pop = mutation(pop, indicator_list)
        fit_calc(pop, df2, action)
        order(pop)
        pop = remove_double(pop)
        pop = selection(pop, pop_size)
    return pop


def create_pops(pop_size, it_num, indicator_list, df2):
    pop_buy = Init_Pop(pop_size, indicator_list)
    pop_sell = Init_Pop(pop_size, indicator_list)
    pop_hold = Init_Pop(pop_size, indicator_list)
    pop_buy = cycle(pop_buy, it_num, indicator_list, df2, pop_size, 'buy')
    pop_sell = cycle(pop_sell, it_num, indicator_list, df2, pop_size, 'sell')
    for i in range(len(pop_hold.chromosomes)):
        pop_hold.chromosomes[i].genes_val = [two_sig((a + b) / 2) for a, b in zip(pop_buy.chromosomes[i].genes_val,
                                                                                  pop_sell.chromosomes[i].genes_val)]
    return pop_buy, pop_sell, pop_hold


def create_train_df(pop_buy, pop_sell, pop_hold, headers):
    buy_train = []
    for chromo in pop_buy.chromosomes:
        buy_train.append(chromo.genes_val)
    sell_train = []
    for chromo in pop_sell.chromosomes:
        sell_train.append(chromo.genes_val)
    hold_train = []
    for chromo in pop_hold.chromosomes:
        hold_train.append(chromo.genes_val)

    df_buy = pd.DataFrame(buy_train, columns=headers)
    df_buy['action'] = 1
    df_sell = pd.DataFrame(sell_train, columns=headers)
    df_sell['action'] = -1
    df_hold = pd.DataFrame(hold_train, columns=headers)
    df_hold['action'] = 0
    dfs = [df_buy, df_sell, df_hold]
    return dfs
