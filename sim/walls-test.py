'''
Created on 09.10.2012

@author: frederic
'''
import scipy as sp
import string

#same as eom8 but half the orientation time

N = 50
percentage = [0.5]

def getConst(vartuple, exclude=None):
    factors = const["factors"]
    assert len(vartuple)==len(factors)
    newconst = const.copy()
    if exclude=="repetitions" and "repetitions" in factors:
        newconst["name"] = string.replace(newconst["name"], "_r%s", "")
        vartuple = tuple([vartuple[i] for i in range(len(vartuple)-1)])
    newname = newconst["name"] % (vartuple)
    for i, n in enumerate(vartuple):
        newconst[factors[i]] = vartuple[i]
    
    perc = vartuple[0]    
    N_a = int(N*(1-perc))
    newconst["N_amoeboid"] = N_a
    newconst["N_mesenchymal"] = N - N_a
    newconst["name"] = newname
    return newconst

const = {
#parameters that control the simulation in general
"get" : getConst,
"name" : "walls-test_pM%s_r%s", #name of folder including %s for variables to be replaced according to "factors"
"factors" : ["percentage", "repetitions"],
"handle_repetitions_with" : "mean",
"repetitions" : range(1),
"max_time" : 2500.0,
"dt" : 0.1,
"N_amoeboid" : 1,
"N_mesenchymal" : 1,
"percentage" : percentage, 

#initial conditions
"success_radius" : 100,
"initial_position" : [110, 40],
"initial_position_stray" : 10,

#parameters that apply to every agent
"q" : [1.0],
#"q" : [0.8],
"delta" : 0.3,
"mass" : 1,#5e-4,
"gamma_a" : 0.1,                  #Stokes' friction = gamma * v
"gamma_m" : 1.0,
"eta" : 0.5,                      #d is a constant characterizing propulsion
"radius" : 5,
"orientationperiod" : 200,      #each #? time units the agents reorient themselves according to the gradient
"periodsigma" : 20,             #describes the variation in the above quantity
"orientationdelay" : 10,        #it takes them #? time units to do so
"delaysigma" : 2,               #describes the variation in the above quantity
"compass_noise_a" : round(0.225*sp.pi, 5),    #when the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value
"compass_noise_m" : 2.0/3 * round(0.225*sp.pi, 5),    #when the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value

#parameters that govern forces and interactions
"r" : 0.0 * sp.sqrt(2),         #randomness scaling factor (when normal: sigma)
"enable_interaction" : False,
"interaction_radius" : 3, #for distances smaller than this, agents experience repulsion
"alignment_radius" : 6, #for distances between interaction_radius and this, there is alignment
"repulsion_coupling" : 10,
"w" : 0.05, # the weight that describes the alignment weight (but alignment happens at each step, so it is multiplied by dt later to compensate)

#parameters that apply to the maze and the environment
"fieldlimits" : (0, 800, 0, 800),
"border": 10,
"nodegradationlimit" : 13,
"gradientcenter" : [800.0, 800.0],
#"gradientcenter" : [400.0, 650.0],
"maze" : "resources/medium_1600.new.png",
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
"fps" : 0.2, #0.4
"create_path_plot" : True,
"create_video_directly" : False,

"simulations_with_complete_dataset" : 5,
}
