import json
import xtrack as xt
import xpart as xp
import xobjects as xo 
import numpy as np
import scipy as sp
import pandas as pd
import sys
import math
import json
import xdeps as xd
import conda


with open('/home/elamb/W_22_06_21/lhcmask/python_examples/run3_collisions_python/xsuite_lines/line_bb_for_tracking.json', 'r') as fid: # from sampple data
    loaded_dct = json.load(fid)
line = xt.Line.from_dict(loaded_dct)

line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=7460.522473522263)
part_ref = line.particle_ref 

context = xo.ContextCupy()         # For GPU
tracker = xt.Tracker(line=line, _context=context,)


sig1 = 1.09108937e-05*1000 # beam sigma at the IP

tracker.vars['on_sep1'] = sig1*20
tracker.vars['on_sep5'] = sig1*20

bunch_intensity = 1.8e11
sigma_z = 0 
n_part = int(2e4) # GPU optimisation for tracking 
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

particles = xp.generate_matched_gaussian_bunch(_context=context,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=part_ref,
         tracker=tracker)


# 10 sigma to 0 sigma in adiabatic steps
sigmas = np.linspace(sig20,0,100) # 10 sigma to 0 sigma in adiabatic steps

x_s=[]
px_s=[]
y_s=[]
py_s=[]


for i in range(len(sigmas)): 
    tracker.vars['on_sep1'] = sigmas[i]
    tracker.vars['on_sep5'] = sigmas[i]
    tracker.track(particles, num_turns=3000,turn_by_turn_monitor=False) 
    
    
    x_s.append(particles.x.get().copy()) 
    px_s.append(particles.px.get().copy())
    y_s.append(particles.y.get().copy())
    py_s.append(particles.py.get().copy())
    
pd.DataFrame(x_s).to_csv("tracking/x_track.csv")
pd.DataFrame(y_s).to_csv("tracking/y_track.csv")
pd.DataFrame(px_s).to_csv("tracking/px_track.csv")
pd.DataFrame(py_s).to_csv("tracking/py_track.csv")
