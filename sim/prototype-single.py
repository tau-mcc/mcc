"""This is an example of how you can describe simulations you want ``mcc`` to run."""
import numpy as np
import string

def getConst(vartuple, exclude=None):
    newconst = const.copy()
    
    perc = newconst["percentage"]
    N = newconst["N"]
    N_a = int(round(N*(1-perc)))
    newconst["N_amoeboid"] = N_a
    newconst["N_mesenchymal"] = N - N_a
    return newconst

#parameters that control the simulation in general
const = {
#name of the function that returns const in the right form
"get" : getConst,
#name of folder including %s for variables to be replaced according to ``factors``
"name" : "prototype-single",
#Which of the variables in const should be interpreted as a range of values to simulate?
"factors" : [],
#What to do when calculating values (from finalstats) from many repetitions: mean or median?
"handle_repetitions_with" : "mean",
#How many times this simulation will be repeated (should be >=1)
"repetitions" : 0,
#maximum simulation time
"max_time" : 1500.0,
#timestep
"dt" : 0.1,
"N" : 50,
#number of amoeboid agents
"N_amoeboid" : 10,
#number of mesenchymal agents
"N_mesenchymal" : 10,
#percentage of mesenchymal agents (of the total population)
"percentage" : 0.4,

#initial conditions
#the cercle inside of which agents are considered successful has radius:
"success_radius" : 100,
#the agents' initial positions follow a Gaussian distribution around:
"initial_position" : [110, 40],
#with standard deviation:
"initial_position_stray" : 10,

#parameters that apply to every agent
#energy intake
"q" : 0.5,
#energy dissipation
"delta" : 0.3,
"mass" : 1,#5e-4,
#Stokes' friction = gamma * v
#for amoeboids
"gamma_a" : 0.1,
#for mesenchymals                  
"gamma_m" : 1.0,
#eta is a constant characterizing propulsion
"eta" : 0.5,
#radius of agents
"radius" : 5,
#period and standard deviation of the two steps of chemotaxis
"orientationperiod" : 200,      #each #? time units the agents reorient themselves according to the gradient
"periodsigma" : 20,             #describes the variation in the above quantity
"orientationdelay" : 10,        #it takes them #? time units to do so
"delaysigma" : 2,               #describes the variation in the above quantity
"compass_noise_a" : round(0.225*np.pi, 5),    #when the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value
"compass_noise_m" : 2.0/3 * round(0.225*np.pi, 5),    #when the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value

#parameters that govern forces and interactions
#factor of the stochastic force
"r" : 0.0 * np.sqrt(2),
#enable interaction, True or False
"enable_interaction" : True,
#for distances smaller than this, agents experience repulsion
"interaction_radius" : 3,
#for distances between interaction_radius and this, there is alignment
"alignment_radius" : 6,
#constant governing repulsion
"repulsion_coupling" : 10,
#the weight that describes the alignment weight (but alignment happens at each step, so it is multiplied by dt later to compensate)
"w" : 0.05,

#parameters that apply to the maze and the environment
"fieldlimits" : (0, 800, 0, 800),
"border": 10,
"nodegradationlimit" : 13,
"gradientcenter" : [800.0, 800.0],
"maze" : "resources/easy_1600.png",
"wall" : 1.0,
"wall_limit" : 0.3,

#parameters that apply specifically to mesenchymal agents
"eatshape" : "eat_15.png",      #eatshape is a stupid name for the file that describes how surrounding pixels are affected by degradation
"degradation_radius" : 50,
"zeta" : 1.0,
"safety_factor" : 1.2,          #The safety factor describes what the sides of the rectangle containing the degradation imprints are multiplied with when updating the gradient.
"aura" : "aura.png",            #The aura blabla
"y" : 100.0,

#parameters that control what happens after the simulation
"numframes" : 170,
"fps" : 0.05, #0.4
"create_path_plot" : True,
"create_video_directly" : False,

"simulations_with_complete_dataset" : 1,
}
