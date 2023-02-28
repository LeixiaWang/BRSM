import parameters
import tool
from brsdp import *
from user import *
from multiprocessing import Pool
import time



if __name__ == '__main__':
    X, d, n = tool.read_data(parameters.default_data)
    model = BRSM(X, d, n, parameters.default_randomizer, parameters.central_epsilon, parameters.default_byzt_ratio, parameters.byzt_ratio_upper_bound)
    poisoned = model.collect_n_aggregate(parameters.default_attack)
    rectified = model.rectification(poisoned, parameters.default_rectify)
    
    f_real = tool.cal_hist_f(X, d)
    tool.draw_bar(f_real, 'original histogram')
    tool.draw_bar(poisoned, 'poisoned histogram')
    tool.draw_bar(rectified, 'rectified histogram')