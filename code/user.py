from operator import methodcaller
import parameters
import math
import numpy as np
import tool
import scipy.stats as stats

class Users():
    def __init__(self, honest_data:list, byzt_data:list, d:int, local_epsilon:float):
        self.honest_data = honest_data.copy()
        self.byzt_data = byzt_data.copy() if byzt_data is not None else None
        self.d = d
        self.honest_n = len(honest_data)
        self.byzt_n = len(byzt_data) if byzt_data is not None else 0
        self.eps_l = local_epsilon

    
    def randomize(self, randomizer:str, attack = 'None', attack_target_ratio = None):
        self.randomizer_name = randomizer
        self._privacy_prob() # generate p and q for specfic randomizer
        hist = methodcaller('randomizer_' + randomizer, self.honest_data)(self)
        if attack == 'ideal':
            return hist
        if attack != 'None':
            self.attack_name = attack
            if attack_target_ratio is None:
                hist += methodcaller('attack_' + attack + '_' + randomizer)(self)
            else:
                hist += methodcaller('attack_' + attack + '_' + randomizer, attack_target_ratio)(self)
        elif self.byzt_data is not None:
            hist += methodcaller('randomizer_' + randomizer, self.byzt_data)(self)
        return hist


    def _privacy_prob(self):
        if self.randomizer_name == 'ue':
            self.p = math.exp(self.eps_l / 2) / (math.exp(self.eps_l / 2) + 1)
            self.q = 1 / (math.exp(self.eps_l / 2) + 1)
        if self.randomizer_name == 'grr':
            self.p = math.exp(self.eps_l) / (math.exp(self.eps_l) + self.d - 1)
            self.q = 1 / (math.exp(self.eps_l) + self.d - 1)


    def randomizer_ue(self, X = None):
        # in simulation, we can output the aggregated result to save memory.
        if X is None:
            X = self.honest_data
            n = self.honest_n
        else:
            n = len(X)

        block_size = 10 ** 5
        Y_hist = np.zeros(self.d)
        for i in range(math.ceil(n/block_size)):
            left = i * block_size
            right = (i+1)*block_size if (i+1)*block_size <= n else n
            X_block = X[left:right]
            n_block = len(X_block)

            rng = np.random.default_rng()
            Y = np.zeros([n_block,self.d])
            Y[np.arange(n_block),X_block] = 1

            randoms = rng.random((n_block,self.d))
            Y[randoms > self.p-self.q] = rng.integers(0, 2, len(randoms[randoms>self.p-self.q]))

            Y_hist += np.sum(Y, axis=0)

        return Y_hist


    def randomizer_grr(self, X = None):
        # in simulation, we can output the aggregated result to save memory.
        if X is None:
            X = self.honest_data
            n = self.honest_n
        else:
            n = len(X)
        rng = np.random.default_rng()
        randoms = rng.random(size=n)

        Y = X.copy()
        Y[randoms > self.p-self.q] = rng.integers(0, self.d, len(randoms[randoms>self.p-self.q]))

        Y_hist = tool.cal_hist_c(Y, self.d)
        return Y_hist

    
    def attack_MLA_grr(self):
        # Maximal loss attack with grr randomizer
        # in simulation, we can output the aggregated result to save memory.
        rng = np.random.default_rng()
        target = rng.choice(self.d)
        Y_hist = np.zeros(self.d)
        Y_hist[target] = self.byzt_n
        return Y_hist


    def attack_MLA_ue(self):
        # Maximal loss attack with ue randomizer
        # in simulation, we can output the aggregated result to save memory.
        rng = np.random.default_rng()
        c = self.p + (self.d - 1) * self.q
        targets = rng.choice(range(self.d), math.ceil(c))
        Y_hist = np.zeros(self.d)
        Y_hist[targets] = self.byzt_n
        prob_plus = math.ceil(c) - math.floor(c)
        if prob_plus != 0:
            target_plus = rng.choice(targets)
            Y_hist[target_plus] = np.random.binomial(self.byzt_n, prob_plus)
        return Y_hist


    def attack_ASA_grr(self):
        rng = np.random.default_rng()
        Y = rng.choice(self.d, self.byzt_n)
        Y_hist = tool.cal_hist_c(Y, self.d)
        return Y_hist

    
    def attack_ASA_ue(self):
        c = self.p + (self.d - 1) * self.q
        Y_hist = np.random.binomial(self.byzt_n, c/self.d, self.d)
        return Y_hist
        

    def attack_RDA_grr(self):
        rng = np.random.default_rng()
        bern_probs = rng.random((self.d,))
        bern_probs = bern_probs / bern_probs.sum()
        # bern_probs = np.random.dirichlet(bern_probs) # bern_probs值很大时和归一化的结果相似
        Y_hist = [np.random.binomial(self.byzt_n, bern_probs[i]) for i in range(self.d)]
        # tool.draw_bar(Y_hist, color='kb')
        return Y_hist


    def attack_RDA_ue(self):
        rng = np.random.default_rng()
        c = self.p + (self.d - 1) * self.q
        bern_probs = rng.random((self.d,))
        bern_probs = bern_probs / bern_probs.sum() * c
        Y_hist = [np.random.binomial(self.byzt_n, bern_probs[i]) for i in range(self.d)]
        # tool.draw_bar(Y_hist, 'byzt_distr', 'kb')
        return Y_hist

    
    def attack_RGA_grr(self):
        rng = np.random.default_rng()
        mu = rng.random() * self.d
        sigma = rng.random() * self.d
        distr = np.zeros(self.d)
        # print(mu, sigma)
        for i in range(self.d):
            distr[i] = stats.norm.cdf(i+1, mu, sigma) - stats.norm.cdf(i, mu, sigma)
        distr /= distr.sum()
        Y = rng.choice(self.d, self.byzt_n, p=distr, shuffle=False).tolist()
        Y_hist = tool.cal_hist_c(Y, self.d)
        # tool.draw_bar(Y_hist)
        return Y_hist

    
    def attack_RGA_ue(self):
        rng = np.random.default_rng()
        c = self.p + (self.d - 1) * self.q
        mu = rng.random() * self.d
        sigma = rng.random() * self.d
        distr = np.zeros(self.d)
        for i in range(self.d):
            distr[i] = stats.norm.cdf(i+1, mu, sigma) - stats.norm.cdf(i, mu, sigma)
        distr = distr / distr.sum() * c
        Y_hist = [np.random.binomial(self.byzt_n, distr[i]) for i in range(self.d)]
        # tool.draw_bar(Y_hist)
        return Y_hist


    def attack_RIA_grr(self):
        rng = np.random.default_rng()
        Y = rng.choice(self.d, self.byzt_n)
        Y_hist = self.randomizer_grr(Y)
        return Y_hist


    def attack_RIA_ue(self):
        rng = np.random.default_rng()
        Y = rng.choice(self.d, self.byzt_n)
        Y_hist = self.randomizer_ue(Y)
        return Y_hist


    def attack_MGA_grr(self, target_ratio = 0.02):
        rng = np.random.default_rng()
        if target_ratio != 1:
            targets = rng.choice(self.d, math.ceil(self.d * target_ratio), replace = False)
        else:
            targets = np.arange(self.d)
        Y = rng.choice(targets, self.byzt_n)
        Y_hist = tool.cal_hist_c(Y, self.d)
        return Y_hist


    def attack_MGA_ue(self, target_ratio = 0.02):
        rng = np.random.default_rng()
        c = round(self.p + (self.d - 1) * self.q)
        target_num = round(self.d * target_ratio)
        targets = rng.choice(self.d, round(self.d * target_ratio), replace = False)
        Y_hist = np.zeros(self.d)
        if c >= target_num:
            l = c - target_num
            Y_hist[targets] = 1 * self.byzt_n
            non_targets = np.delete(np.arange(self.d), targets)
            for i in range(self.byzt_n):
                Y = rng.choice(non_targets, l, replace = False)
                Y_hist[Y] += 1
        else:
            for i in range(self.byzt_n):
                Y = rng.choice(targets, c, replace = False)
                Y_hist[Y] += 1
        # tool.draw_bar(Y_hist, 'kb')
        return Y_hist