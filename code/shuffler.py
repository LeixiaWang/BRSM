from cmath import exp
from http.client import ImproperConnectionState
import parameters
import tool
import math
import numpy as np
from user import *

class Shuffler():
    def __init__(self, Y_hist:list, n:int, local_epsilon: float, randomizer:str, pa_mode:str, byzt_ratio_upper_bound:float):
        self.Y_hist = Y_hist.copy()
        self.d = len(Y_hist)
        self.n = n
        self.local_epsilon = local_epsilon
        self.randomizer = randomizer
        self.pa_mode = pa_mode
        self.byzt_ratio_upper_bound = byzt_ratio_upper_bound


    def cal_padding_length(self):
        if self.pa_mode == 'grr':
            m = self.n * self.byzt_ratio_upper_bound * self.d / (math.exp(self.local_epsilon) + self.d - 1)
        elif self.pa_mode == 'general':
            m = self.n * self.byzt_ratio_upper_bound * self.d / (2 * math.exp(self.local_epsilon) + self.d - 2)
        elif self.pa_mode == 'None':
            m = 0
        return math.ceil(m)


    def padding_shuffling(self):
        # we omit the shuffling step since we pass the aggregate message directly and it will not influence the accuracy
        # in actual, we should shuffle all messages
        rng = np.random.default_rng()
        self.m = self.cal_padding_length()
        noises = rng.integers(0, self.d, self.m)
        perturb_tool = Users(noises, None, self.d, self.local_epsilon)
        perturbed_noises = perturb_tool.randomize(self.randomizer)
        Y_hist = self.Y_hist + perturbed_noises
        return Y_hist, self.m