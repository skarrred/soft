# Practical 09: Implement the following.

# A. Find ratios using fuzzy logic.

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

s1 = "I like softcomputing"
s2 = "I like hardcomputing"

print("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2))
print("FuzzyWuzzy PartialRatio:", fuzz.partial_ratio(s1, s2))
print("FuzzyWuzzy TokenSortRatio:", fuzz.token_sort_ratio(s1, s2))
print("FuzzyWuzzy TokenSetRatio:", fuzz.token_set_ratio(s1, s2))
print("FuzzyWuzzy WRatio:", fuzz.WRatio(s1, s2), '\n\n')



# B. Solve Tipping problem using fuzzy logic.

import numpy as np
import skfuzzy as fuzz
# pip install scikit-fuzzy
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create fuzzy variables and their ranges
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Define membership functions for quality and service
quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 5])
quality['average'] = fuzz.trimf(quality.universe, [0, 5, 10])
quality['excellent'] = fuzz.trimf(quality.universe, [5, 10, 10])

service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['average'] = fuzz.trimf(service.universe, [0, 5, 10])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

# Define membership functions for the tip
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# Define fuzzy rules
rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(quality['excellent'] | service['excellent'], tip['high'])

# Create the fuzzy control system
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Create a simulation
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Input values for quality and service
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

# Compute the tipping result
tipping.compute()

# Print the output tip value
print("Recommended tip:", tipping.output['tip'])

# Visualize the membership functions and the output
quality.view()
service.view()
tip.view()

# Show the plots
plt.show()