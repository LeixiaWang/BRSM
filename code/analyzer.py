from code import interact
import parameters
import tool
import math
import hdbscan
import numpy as np
from collections import Counter


class Analyzer():
    def __init__(self, Y_hist:list, n:int = -1, m:int = -1, local_epsilon:float = -1, randomizer:str = 'none'):
        self.Y_hist = Y_hist.copy()
        if n != -1:
            self.n = n
            self.d = len(Y_hist)
            self.m = m
            self.local_epsilon = local_epsilon
            self.randomizer = randomizer
            self._privacy_prob()


    def _privacy_prob(self):
        if self.randomizer == 'ue':
            self.p = math.exp(self.local_epsilon / 2) / (math.exp(self.local_epsilon/ 2) + 1)
            self.q = 1 / (math.exp(self.local_epsilon / 2) + 1)
        if self.randomizer == 'grr':
            self.p = math.exp(self.local_epsilon) / (math.exp(self.local_epsilon) + self.d - 1)
            self.q = 1 / (math.exp(self.local_epsilon) + self.d - 1)
        ept_var = self.q * (1-self.q) / (self.n* ((self.p-self.q) ** 2))
        self.sigma = math.sqrt(ept_var)


    def aggregate(self):
        nume = (self.n + self.m) * self.q
        deno = self.n * (self.p - self.q)
        diff = self.m / (self.n * self.d)
        self.f_hat = (self.Y_hist - nume) / deno - diff
        return self.f_hat


    def normalization_Cao(self, f):
        f = f.copy()
        f_min = min(f)
        f_tilde = (f-f_min) / (f-f_min).sum()
        return f_tilde


    def _normalization(self, f):
        f = f.copy()
        f_min = min(f)
        if f_min < 0:
            f -= f_min
        if f.sum() != 0:
            f_tilde = f / f.sum()
        else:
            f_tilde = np.ones(self.d) / self.d
        return f_tilde


    def cluster_align(self, f = None):
        labels = self._hdbscan(f)
        hist = self._scale2align(f, labels)

        while True:
            labels = self._hdbscan(hist)
            new_hist = self._scale2align(hist, labels)
            if (new_hist == hist).all():
                break
            else:
                hist = new_hist.copy()
        return hist


    def _hdbscan(self, f):
        ## 2-D clustering, which is better than 1-D case. In most cases, we derive one or two clusters
        f = f.copy()
        x = np.arange(len(f))
        x = x / (len(f))
        f = f / f.sum()
        points = np.vstack((np.array(f), np.array(x))).T
        # clustr = hdbscan.HDBSCAN(cluster_selection_method='eom', allow_single_cluster=True)
        cluster_epsilon = float(math.sqrt((x[1]-x[0])**2 + (self.sigma * 1.96 * 2/ f.sum())**2))
        clustr = hdbscan.HDBSCAN(cluster_selection_method='eom', allow_single_cluster=True, cluster_selection_epsilon=cluster_epsilon)
        clustr.fit(points)
        labels = clustr.labels_
        return labels


    def _scale2align(self, hist, label):
        # there're a lot of clusters 
        # choose the one with the largest number of items as the basic item
        # align other clusters to the basic one
        hist = hist.copy()
        label = label.copy()
        d = len(hist)
        n_c = label.max() + 1
        diff = [np.inf for _ in range(n_c)]
        basic = Counter(label).most_common(1)[0][0]

        if np.all(label == -1):
            return hist
        if np.any(label != -1) and basic == -1: # the outlier cannot be the basic item
            basic = Counter(label).most_common(2)[1][0]
            
        # store the diff of '-1' on the basic one
        for j in range(d):
            if label[j] == basic:
                if j-1 >= 0 and label[j-1] != basic:
                    update = label[j-1] if label[j-1] != -1 else basic
                    diff[update] = (hist[j-1] - hist[j]) if abs(hist[j-1] - hist[j]) < abs(diff[update]) else diff[update]
                if j+1 <= d-1 and label[j+1] != basic:
                    update = label[j+1] if label[j+1] != -1 else basic
                    diff[update] = (hist[j+1] - hist[j]) if abs(hist[j+1] - hist[j]) < abs(diff[update]) else diff[update]
        for i in range(n_c):
            if diff[i] != np.inf:
                if i != basic:
                    hist[label == i] -= diff[i]
                else: # this step processes the outlier. if you don't want to align the outliers, comment it out.
                    hist[label == -1] -= diff[basic]
        return hist


    def _smooth(self, f):
        d = len(f)
        f_bar = np.zeros(d)
        for j in range(d):
            l = f[j-1] if j-1 >= 0 else f[0]
            r = f[j+1] if j+1 <= d-1 else f[d-1]
            f_bar[j] = (l + f[j] + r) / 3
        return f_bar


    def _indicator(self, f_bar, f_hat, threshold):
        # the benign items are labeled as 1
        # the corrupted items are labeled as 0
        indi = np.ones(self.d)
        bias = np.abs(f_hat-f_bar)
        indi[bias >= threshold] = 0
        return indi


    def _fill(self, f, inds):
        f = f.copy()
        l = -1
        if np.all(inds == 0):
            return np.zeros(len(f))
        for i in range(len(inds)):
            if inds[i] == 1:
                if i - l > 1:
                    if l == -1:
                        f[:i] = f[i]
                    else:
                        f[l + 1:i] = (f[i] + f[l]) / 2
                l = i
        if l < len(inds) - 1:
            f[l:] = f[l]
        return f
        

    def basic_rectification(self, f:list, T_xi = None, return_index = False, cluster_filter = True):
        if T_xi is None:
            T_xi = self.sigma * 2 * 1.96

        if cluster_filter is True:
            f_hat = self.cluster_align(f)
        else:
            f_hat = f.copy()

        last_inds = None
        inds = np.ones(self.d)
        f_bar = f_hat.copy()
        iter_i = 0
        while not np.array_equal(inds, last_inds) and iter_i < self.d:
            last_inds = inds.copy()
            # print(inds)
            f_bar = self._smooth(f_bar)
            # print(np.round(f_bar * 400))
            inds = self._indicator(f_bar, f_hat, T_xi)
            f_bar = self._fill(f_hat, inds)
            # print(np.round(f_bar * 400))
            iter_i += 1
            # tool.draw_bar(f_bar)
        f_bar = self._smooth(f_bar)
        f_bar = self._normalization(f_bar)
        if return_index:
            return f_bar, inds
        else:
            return f_bar, {'iterat_4_dect':iter_i}


    def _find_threshold_range(self, f_hat, T_min, T_max, indx_min, indx_max):
        # 二分搜索停止阈值
        stop_thresh = max(T_min, 2 * 1.96 * self.sigma)
        # 找到包含所需区间的最小(T_min, T_max)
        while True:
            T_middle = (T_min + T_max) / 2
            f_bar, T_middle_indx = self.basic_rectification(f_hat, T_middle, return_index=True, cluster_filter=False)
            T_middle_num = len(T_middle_indx[T_middle_indx == 1])
            if T_max - T_min <= stop_thresh:
                break
            if T_middle_num > indx_min and T_middle_num < indx_max:
                break
            elif T_middle_num <= indx_min:
                T_min = T_middle
                continue
            elif T_middle_num >=  indx_max:
                T_max = T_middle
                continue
        # 分别对最小区间的两个端点l和r进行二分搜索
        # binary search for l
        l_min, l_max = self._binary_search(T_min, T_middle, indx_min, f_hat, stop_thresh)
        T_min = l_min
        # binary search for r
        r_min, r_max = self._binary_search(T_middle, T_max, indx_max, f_hat, stop_thresh)
        T_max = r_max
        # 有可能找到的该区间无限小，此时，返回T_min=T_max，该值取距离该区间最近的两个benign item number较大的那个，即选更可信的那个
        if T_max - T_min <= stop_thresh:
            T_min = T_max
        return T_min, T_max


    def _binary_search(self, l, r, num_cons, f_hat, stop_thresh):
        if (r-l) > stop_thresh:
            m = (l + r) / 2
            f_bar, m_indx = self.basic_rectification(f_hat, m, return_index=True, cluster_filter=False)
            m_num = len(m_indx[m_indx == 1])
            if m_num <= num_cons:
                l = m
            if m_num >= num_cons:
                r = m
            l,r = self._binary_search(l, r, num_cons, f_hat, stop_thresh)
        return l,r
        

    def optimal_rectification(self, f, T_set = None):
        f_hat = self.cluster_align(f)

        if T_set is None:
            # Delta 从小到大 
            f_s = f_hat - self._smooth(f_hat)
            Delta = np.abs(f_s)
            Delta = np.sort(Delta) 
            # search the appropriate
            min_trimmed_ratio = 0.6
            max_trimmed_ratio = 0.9
            T_min, T_max = self._find_threshold_range(f_hat, Delta[0], Delta[-1], self.d * min_trimmed_ratio, self.d * max_trimmed_ratio)
            # enumerate thresholds
            num = 100
            T_set = np.linspace(T_min, T_max, num = num, endpoint = True)

        F = []
        i = 0
        for T_xi in T_set:
            f_bar, indx = self.basic_rectification(f_hat, T_xi, return_index=True, cluster_filter=False)
            benign_item_ratio = len(indx[indx == 1]) /self.d
            if benign_item_ratio >= min_trimmed_ratio and benign_item_ratio != 1:
                i += 1
                F.append(f_bar)  
        F = np.array(F)

        stop_thresh = 1 / (self.n**2)
        stop_round = 1000 # we set this value but haven't use it. All updates can degrade the objective function step by step util convergence. The procedure is very quickly.
        F_tilde, iters_i, iter_j = self._optimize(F, stop_thresh, stop_round)
        return F_tilde, {'iterate_4_f':iters_i, 'iterate_4_opt':iter_j}

    def _optimize(self, F, stop_threshold, stop_round):
        weights = np.ones(len(F)) * math.log(len(F))
        old_obj = np.inf
        alpha = self._compute_penalty_coeff(weights)
        iter_j = 0
        iters_i = []
        while True:
            iter_j += 1
            F_tilde, F_smooth, iter_i = self._update_f_tilde(F, weights, alpha, stop_threshold, stop_round)
            iters_i.append(iter_i)
            weights = self._update_weights(F_tilde, F)
            alpha = self._compute_penalty_coeff(weights)
            new_obj = self._objective_func(F_tilde, F, F_smooth, weights, alpha)
            # print('iterate for optimizing', np.abs(old_obj - new_obj),stop_threshold,iter_j)
            if np.abs(old_obj - new_obj) < stop_threshold:
                break
            old_obj = new_obj
        return F_tilde, iters_i, iter_j

    
    def _compute_penalty_coeff(self, weights):
        return weights.sum()


    def _update_weights(self, F_tilde, F):
        F_smooth = self._smooth(F_tilde)
        Diff = np.square(F_tilde - F).sum(axis = 1) + np.square(F_tilde - F_smooth).sum()
        weights = - np.log(Diff / Diff.sum())
        return weights


    def _objective_func(self, F_tilde, F, F_smooth, weights, alpha):
        return (weights * np.square(F_tilde - F).sum(axis = 1)).sum() + alpha * np.square(F_tilde - F_smooth).sum()

    
    def _update_f_tilde(self, F, weights, alpha:float, stop_threshold:float, stop_round:int):
        # initialization
        F_smooth = np.zeros(self.d)
        old_obj = np.inf
        new_obj = - np.inf
        iter_i = 0
        # iteration for approximate solutions
        while abs(old_obj - new_obj) > stop_threshold:
            # print('iterate for approximate f', abs(old_obj - new_obj), stop_threshold, iter_i)
            wf = ((weights * F.T).T).sum(axis = 0) + alpha * F_smooth
            ww = weights.sum() + alpha
            F_tilde_init = wf / ww
            F_tilde = F_tilde_init + (1 - F_tilde_init.sum()) / self.d
            # iteration for searching non-zero f_tilde
            zero_F_indx = set()
            while (F_tilde < 0).any():
                zero_F_indx.update(np.where(F_tilde < 0)[0])
                F_tilde_init[list(zero_F_indx)] = 0
                F_tilde = F_tilde_init + (1 - F_tilde_init.sum()) / (self.d - len(zero_F_indx))
                F_tilde[list(zero_F_indx)] = 0
            F_smooth = self._smooth(F_tilde)
            iter_i += 1
            old_obj = new_obj
            new_obj = self._objective_func(F_tilde, F, F_smooth, weights, alpha)
        return F_tilde, F_smooth, iter_i