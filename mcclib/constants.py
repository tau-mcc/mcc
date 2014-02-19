"""
Defines constants used in the entire project.

In particular, defines a constant :py:const:`currentsim` that refers to the sim-file to be used by :py:mod:`statistics` and :py:mod:`graphics`.
"""

import string
import numpy as np

useWeave = True
#compiler = "msvc"

#Avoid changing this if you can. If you do change it, be aware that it can make the *.npy files from previous simulations unusable.
floattype = np.float_

#standard values are
wallconst = 100.0
wallgradconst = 0.0

TYPE_MESENCHYMAL = 1
TYPE_AMOEBOID = 2
safety_factor = 1.2

DIM = 2
E_dim = 3
line_resolution_step = 100

c_ = "sim"
simdir = c_ + "/"
confpackage = c_ + "."

resultspath = "results/"
resourcespath = "resources/"

framesdir = "frames/"

logfilename = "sim.log"
videoshellscript = "anim.sh"

graphics_ending = ".png"
#graphics_ending = ".tiff"

simpickle_filename = "simdata.pickle"
finalstats_pickle = "finalstats.pickle"
finalstats_text = "finalstats.txt"
saved_constants_filename = "constants.txt"
individual_CI_filename = "individualCI.csv"
individual_ttt_filename = "individualTTT.csv"
individual_success_filename = "individualSuccess.csv"

avg_en_vel_filename = "avg_en_vel" + graphics_ending
avg_dist_filename = "avg_dist_from_goal" + graphics_ending
succ_rate_filename = "success_rate" + graphics_ending
finalmaze_filename = "finalmaze.npy"
pathlengths_filename = "pathlengths" + graphics_ending
pathlengths_filename_m = "pathlengths_m" + graphics_ending
pathlengths_filename_a = "pathlengths_a" + graphics_ending
pathlengths_stacked = "pathlengths_stacked" + graphics_ending
scattervel_filename = "vel_scattered" + graphics_ending
ci_filename = "ci" + graphics_ending
ci_filename_a = "ci_a" + graphics_ending
ci_filename_m = "ci_m" + graphics_ending
cisr_filename = "cisr" + graphics_ending

symbols = {
           "percentage" : "% of mesenchymal agents",
           "distance_from_goal" : "distance from goal",
           "distance_from_goal_a" : "distance from goal of amoeboid agents",
           "distance_from_goal_m" : "distance from goal of mesenchymal agents",
           "avg_path_length" : "average path length",
           "avg_path_length_a" : "average path length for amoeboid agents",
           "avg_path_length_m" : "average path length for mesenchymals agents",
           "avg_path_length_err" : "SD in average path length",
           "avg_path_length_err_a" : "SD in average path length for amoeboid agents",
           "avg_path_length_err_m" : "SD in average path length for mesenchymal agents",
           "success_ratio" : "success rate",
           "success_ratio_a" : "success rate of amoeboid agents",
           "success_ratio_m" : "success rate of mesenchymal agents",
           "fitness" : " success ratio * average energy",
           "fitness_a" : " success ratio * average energy (for amoeboid agents)",
           "fitness_m" : " success ratio * average energy (for mesenchymal agents)",
           "avg_en" : "average energy",
           "avg_en_a" : "average energy of amoeboid agents",
           "avg_en_m" : "average energy of mesenchymal agents",
           "avg_vel" : "average velocity",
           "avg_vel_a" : "average velocity of amoeboid agents",
           "avg_vel_m" : "average velocity of mesenchymal agents",
           "avg_ci_a" : "average CI for amoeboid agents",
           "resources/density_1.png" : "low density ECM",
           "resources/density_2.png" : "medium density ECM",
           "resources/density_3.png" : "high density ECM",           
           }

def escape(s):
    if type(s)==type(""):
        return string.replace(s, "_", " ")
    else:
        return s

def symbol(name):
    return symbols[name] if name in symbols else escape(name)

#If you have files in the sim/ directory that you want to ignore, you can add them to the ignore list.
#Alternatively, you can just add a single _ in front of the filename.
ignore = []
#ignore.append("maze-easy-ar.py")
#ignore.append("with-without-i.py")

#currentsim = "with-without-i.py"
#currentsim = "maze-easy-ar.py"
#currentsim = "maze-hard-ar.py"
#currentsim = "prototype.py"
#currentsim = "prototype-single.py"
currentsim = "walls-test2.py"
#currentsim = "maze-medium-test.py"
#currentsim = "wexplore.py"
#currentsim = "free-ci.py"
#currentsim = "varying-w.py"
