import numpy as np

delta = 10 ** (-8)

central_epsilon = 0.8

default_attack = 'MLA'

default_randomizer = 'grr'

default_data = 'Taxi'

default_rectify = 'optimal'
rectify_methods = ['normalization', 'basic', 'optimal']
# refer to Norm, MDR, MDR*, respectively

byzt_ratio_upper_bound = 0.5
default_byzt_ratio = 0.1

data_path = ''
