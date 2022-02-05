from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import brian2cuda
set_device('cpp_standalone', build_on_run=False)

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

# Parameters
simulation_duration = 6 * second

## Neurons
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms

## STDP
taupre  = 20*ms
taupost = taupre
gmax    = .01
dApre   = .01
dApost  = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre  *= gmax

## Dopamine signaling
tauc    = 1000*ms
taud    = 200*ms
taus    = 1*ms
epsilon_dopa = 5e-3

# Setting the stage

## Stimuli section
input_indices = array([0, 1, 0, 1, 1, 0,
                       0, 1, 0, 1, 1, 0])
input_times   = array([ 500,  550, 1000, 1010, 1500, 1510,
                     3500, 3550, 4000, 4010, 4500, 4510])*ms

spike_input = SpikeGeneratorGroup(2, input_indices, input_times)

neurons = NeuronGroup(2, '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                            dge/dt = -ge / taue : 1''',
                      threshold='v>vt', reset='v = vr',
                      method='exact')
neurons.v = vr
neurons_monitor = SpikeMonitor(neurons)

synapse = Synapses(spike_input, neurons,
                   model='''s: volt''',
                   on_pre='v += s')
synapse.connect(i=[0, 1], j=[0, 1])
synapse.s = 100. * mV

## STDP section
synapse_stdp = Synapses(neurons, neurons,
                   model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                   on_pre='''ge += s
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                   on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                   method='euler'
                   )
synapse_stdp.connect(i=0, j=1)
#synapse_stdp.mode = 0
synapse_stdp.mode = 1
synapse_stdp.s = 1e-10
synapse_stdp.c = 1e-10
synapse_stdp.d = 0
synapse_stdp_monitor = StateMonitor(synapse_stdp, ['s', 'c', 'd', 'Apre', 'Apost'], record=[0])

## Dopamine signaling section
dopamine_indices = array([0, 0, 0])
dopamine_times = array([3520, 4020, 4520])*ms
dopamine = SpikeGeneratorGroup(1, dopamine_indices, dopamine_times)
dopamine_monitor = SpikeMonitor(dopamine)
reward = Synapses(dopamine, synapse_stdp, model='''''',
                            on_pre='''d_post += epsilon_dopa''',
                            method='exact')

array_i = [0]
array_j = [0]

reward.connect(i=array_i, j=array_j)


# Simulation
## Classical STDP
#synapse_stdp.mode = 0
#run(simulation_duration/2)
## Dopamine modulated STDP
#synapse_stdp.mode = 1
#run(simulation_duration/2)

run(simulation_duration)

device.build( directory='outputcpp', compile = True, run = True, debug=False, clean = True)


# Visualisation

visualise_connectivity(reward)

dopamine_indices, dopamine_times = dopamine_monitor.it
neurons_indices, neurons_times = neurons_monitor.it
plt.figure(figsize=(12, 6))
plt.subplot(511)
plt.plot([0.05, 2.95], [2.7, 2.7], linewidth=5, color='k')
plt.text(1.5, 3, 'Classical STDP', horizontalalignment='center', fontsize=20)
plt.plot([3.05, 5.95], [2.7, 2.7], linewidth=5, color='k')
plt.text(4.5, 3, 'Dopamine modulated STDP', horizontalalignment='center', fontsize=20)
plt.plot(neurons_times, neurons_indices, 'ob')
plt.plot(dopamine_times, dopamine_indices + 2, 'or')
plt.xlim([0, simulation_duration/second])
plt.ylim([-0.5, 4])
plt.yticks([0, 1, 2], ['Pre-neuron', 'Post-neuron', 'Reward'])
plt.xticks([])
plt.subplot(512)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.d.T/gmax, 'r-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Extracellular\ndopamine d(t)')
plt.xticks([])
plt.subplot(513)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.c.T/gmax, 'b-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Eligibility\ntrace c(t)')
plt.xticks([])
plt.subplot(514)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.s.T/gmax, 'g-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Synaptic\nstrength s(t)')
plt.xlabel('Time (s)')
plt.subplot(515)
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.Apre.T, 'k-')
plt.plot(synapse_stdp_monitor.t/second, synapse_stdp_monitor.Apost.T, 'b-')
plt.xlim([0, simulation_duration/second])
plt.ylabel('Traces Apre post(t)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()