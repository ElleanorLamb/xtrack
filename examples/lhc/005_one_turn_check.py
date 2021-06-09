import pickle
import numpy as np

import xtrack as xt
import xobjects as xo
import pysixtrack

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

fname_line_particles = './lhc_no_bb/line_and_particle.pkl'
fname_line_particles = './lhc_with_bb/line_and_particle.pkl'

####################
# Choose a context #
####################

context = xo.ContextCpu()
context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

#############
# Load file #
#############

with open(fname_line_particles, 'rb') as fid:
    input_data = pickle.load(fid)


## DEBUUUUUG:
#line_dict = input_data['line']
#for ii, ee in enumerate(line_dict['elements']):
#    if ee['__class__'] == 'BeamBeam6D':
#        line_dict['elements'][ii] = {}
#        ee = line_dict['elements'][ii]
#        ee['__class__'] = 'Drift'
#        ee['length'] = 0.


##################
# Get a sequence #
##################

sequence = pysixtrack.Line.from_dict(input_data['line'])
if short_test:
    sequence = make_short_line(sequence)

##################
# Build TrackJob #
##################
print('Build tracker...')
tracker = xt.Tracker(context=context,
            sequence=sequence,
            particles_class=xt.Particles,
            local_particle_src=None,
            save_source_as='source.c')

######################
# Get some particles #
######################
part_pyst = pysixtrack.Particles.from_dict(input_data['particle'])

pysixtrack_particles = [part_pyst, part_pyst] # Track twice the same particle

particles = xt.Particles(pysixtrack_particles=pysixtrack_particles,
                         _context=context)
#########
# Track #
#########
print('Track a few turns...')
n_turns = 10
tracker.track(particles, num_turns=n_turns)

############################
# Check against pysixtrack #
############################
print('Check against pysixtrack...')
ip_check = 1
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
pyst_part = pysixtrack_particles[ip_check].copy()
#for _ in range(n_turns):
#    sequence.track(pyst_part)
#
#for vv in vars_to_check:
#    pyst_value = getattr(pyst_part, vv)
#    xt_value = context.nparray_from_context_array(getattr(particles, vv))[ip_check]
#    passed = np.isclose(xt_value, pyst_value, rtol=1e-9, atol=1e-11)
#    if not passed:
#        print(f'Not passend on var {vv}!\n'
#              f'    pyst:   {pyst_value: .7e}\n'
#              f'    xtrack: {xt_value: .7e}\n')
#        raise ValueError

##############
# Check  ebe #
##############
print('Check element-by-element against pysixtrack...')
pyst_part = pysixtrack_particles[ip_check].copy()
vars_to_check = ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']
problem_found = False
for ii, (eepyst, nn) in enumerate(zip(sequence.elements, sequence.element_names)):
    print(f'\nelement {nn}')
    vars_before = {vv :getattr(pyst_part, vv) for vv in vars_to_check}
    particles.set_particles_from_pysixtrack(ip_check, pyst_part)

    tracker.track(particles, ele_start=ii, num_elements=1)

    eepyst.track(pyst_part)
    for vv in vars_to_check:
        pyst_change = getattr(pyst_part, vv) - vars_before[vv]
        xt_change = context.nparray_from_context_array(
                getattr(particles, vv))[ip_check] -vars_before[vv]
        passed = np.isclose(xt_change, pyst_change, rtol=1e-10, atol=1e-14)
        if not passed:
            problem_found = True
            print(f'Not passend on var {vv}!\n'
                  f'    pyst:   {pyst_change: .7e}\n'
                  f'    xtrack: {xt_change: .7e}\n')
            break

    if not passed:
        break
    else:
        print("Check passed!")


if not problem_found:
    print('All passed on context:')
    print(context)

