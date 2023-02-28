import parameters
import math
import numpy as np
from user import *
from shuffler import *
from analyzer import *
import sympy as sp
import tool

class BRSM():
    def __init__(self, data:list, d:int, n:int, perturbation:str, central_epsilon:float,
     byzt_ratio = parameters.default_byzt_ratio, byzt_ratio_upper_bound = parameters.byzt_ratio_upper_bound, delta = parameters.delta):
        self.data = data
        self.d = d
        self.n = n
        self.epsilon = central_epsilon
        self.delta = delta
        self.byzt_number = int(self.n * byzt_ratio)
        self.byzt_ratio_upper_bound = byzt_ratio_upper_bound
        self.randomizer = perturbation
        self.honest_data, self.byzt_data = self._data_partition()
        


    def collect_n_aggregate(self, attack = None, attack_target_ratio = None):
        # the parameter 'attack' is given when existing Byzantine users
        # the parameter 'attck_target_ratio' is give when the attack is 'MGA'
        if attack == 'ideal':
            n = len(self.honest_data)
        else:
            n = self.n
        # initialization
        self.local_epsilon, self.pa_mode = self.PA(self.epsilon, self.delta, n=n)
        # randomize
        self.users = Users(self.honest_data, self.byzt_data, self.d, self.local_epsilon)
        Y_hist_version = self.users.randomize(self.randomizer, attack, attack_target_ratio)
        # padding and shuffling
        self.shuffler = Shuffler(Y_hist_version, n, self.local_epsilon, self.randomizer, self.pa_mode, self.byzt_ratio_upper_bound)
        shuffled_Y_hist, m = self.shuffler.padding_shuffling()
        # aggregate and rectification
        self.analyzer = Analyzer(shuffled_Y_hist, n, m, self.local_epsilon, self.randomizer)
        f_hat = self.analyzer.aggregate()
        return f_hat


    def rectification(self, f_hat, rect_method, analyse_iter = False):
        if rect_method == 'normalization':
            f_tilde = self.analyzer.normalization_Cao(f_hat)
            iter_dict = None
        elif rect_method == 'basic':
            f_tilde, iter_dict = self.analyzer.basic_rectification(f_hat)
        elif rect_method == 'optimal':
            f_tilde, iter_dict = self.analyzer.optimal_rectification(f_hat)
        if analyse_iter:
            return f_tilde, iter_dict
        else:
            return f_tilde
    
    # -------------------------- pre-processing --------------------------------------

    
    def _data_partition(self):
        rng =  np.random.default_rng()
        byzt_indx = rng.choice(self.n, self.byzt_number, replace = False)
        byzt_data = np.array(self.data)[byzt_indx]
        honest_data = np.delete(self.data, byzt_indx)
        return honest_data, byzt_data


    # --------------------------- privacy amplification -------------------------------

    
    def PA(self, central_epsilon, delta, n = 'default'):
        if n == 'default':
            n = self.n
        if self.randomizer == 'grr':
            pa_mode = 'grr'
            local_epsilon = self.pa_clones_GRR_c2l(n, self.d, central_epsilon, delta) 
            if local_epsilon is not None:
                return local_epsilon, pa_mode
        pa_mode = 'general'
        local_epsilon = self.pa_clones_general_c2l(n, central_epsilon, delta)
        if local_epsilon is None:
            print("\033[33mWarning: the privacy amplification is not work.\033[0m")
            pa_mode = 'None'
            local_epsilon = central_epsilon
        return local_epsilon, pa_mode


    def pa_clones_GRR_c2l(self, n, d, epsilon, delta):
        try:
            thresh_c = self.pa_clones_GRR_threshold_c(n,d,delta)
            if epsilon >= thresh_c:
                thresh_l = math.log(n / (16*math.log(4/delta)))
                epsilon = max(epsilon, thresh_l)
                return epsilon
            y = math.exp(epsilon) - 1
            theta = math.log(4/delta)
            alpha = (d+1) / (d*n)
            a = 64*(alpha**2)
            b = 64*(alpha**2)*d - 64*alpha*theta - 16*alpha*y
            c = y**2 - 16*alpha*y
            h = y**2 * d
            x = sp.Symbol('x')
            f = a * x ** 3 + b * x ** 2 + c * x + h
            x = sp.solveset(f,x)
            threshold = n / (16*math.log(4/delta)) - 1
            x = [i for i in x if i > 0 and i <= threshold][0]
            epsilon_l = math.log(x + 1)
            return epsilon_l
        except Exception as ee:
            # print('\033[34m{0}\033[0m'.format(ee))
            return None


    def pa_clones_general_c2l(self, n, epsilon, delta):
        try:
            thresh_c = self.pa_clones_general_threshold_c(n,delta)
            if epsilon >= thresh_c:
                thresh_l = math.log(n / (16*math.log(4/delta)))
                epsilon = max(epsilon, thresh_l)
                return epsilon
            y = (math.exp(epsilon) - 1) / 8
            theta = math.sqrt(math.log(4 / delta) / n)
            x = sp.Symbol('x')
            f = x ** 4 / n + theta * x ** 3 - (1/n + y) * x ** 2 - theta * x - y
            x = sp.solveset(f,x)
            thresh = n / (16*math.log(4/delta))
            y = []
            for i in x:
                t = i ** 2
                if t.is_real and t > 1 and t <= thresh:
                    y.append(t)
            epsilon_l = math.log(y[0])
            return epsilon_l
        except Exception as ee:
            # print('\033[34m{0}\033[0m'.format(ee))
            return None


    def pa_clones_general_threshold_c(self, n, delta):
        eps_l = math.log(n / (16*math.log(4/delta)))
        a = (math.exp(eps_l) - 1)/(math.exp(eps_l) + 1)
        b = 8 * (math.sqrt(math.exp(eps_l) * math.log(4/delta)/n) + math.exp(eps_l)/n)
        thresh = math.log(1 + a * b)
        return thresh


    def pa_clones_GRR_threshold_c(self, n, d, delta):
        eps_l = math.log(n / (16*math.log(4/delta)))
        a = math.exp(eps_l) - 1
        b = 4 * math.sqrt(2 * (d+1) * math.log(4/delta) / ((math.exp(eps_l) + d -1) * d * n))
        c = 4 * (d+1) / (d * n)
        thresh = math.log(1 + a * (b+c))
        return thresh

    # --------------------------- theoretical analysis --------------------------------


    def theoretical_var_without_attak(self, f:list):
        f = np.array(f)
        p = self.users.p
        q = self.users.q
        m = self.shuffler.m
        term1 = (self.n * f + m / self.d) * (1 - p - q) / (self.n ** 2 * (p - q))
        term2 = (self.n + m) * (1 - q) * q / (self.n ** 2 * (p - q) ** 2)
        return np.average(term1 + term2)