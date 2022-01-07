from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

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




start_scope()

np.random.seed(2021)



(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X_flat = np.reshape(train_X,(train_X.shape[0], 28*28))
test_X_flat = np.reshape(test_X,(test_X.shape[0], 28*28))

plt_example_mnist = False

X_size          = 28*28
X_Train_Samples = train_X.shape[0]

if(plt_example_mnist):
    plt.figure()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(10):
    print('Class ' + str(i) + ' has number of samples ' + str(len(train_y[train_y==i])))

if(plt_example_mnist):
    #print the first 9 values
    for k in range(10):
        plt.figure()
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow( (train_X[train_y==k])[i], cmap=plt.get_cmap('gray'))


if(plt_example_mnist):
    plt.show()

'''
we want to have 60FPS
thus, each frame must take 16.6ms, here we use 15ms
10ms for the frame
and 5ms for the resting
'''

single_example_time   = 25*ms
space_between_samples = 25 #included the previous 10ms

taupre = 2 * ms
taupost = 5 * ms


eqs = '''
dv/dt = -v/tau + a/taus: 1 (unless refractory)
da/dt = -a/taus : 1
ddynThreshold/dt = (0.8-dynThreshold)/taut : 1
tau : second
taus : second
taut : second
'''
eqs_reset = '''
                v = -1
                a = +1
                dynThreshold = dynThreshold+0.2
            '''

Threshold = 0.8


n0_s_list        = []
n0_t_list        = []
n0_mon_list      = []
n1_mon_list      = []
n1_state_list    = []
s1_ex_state_list = []

ts_time = 0
Train_Samples = X_Train_Samples
Train_Samples = 3


n0_debug = False

my_train_X_flat  = train_X_flat

my_train_X_flat1 = train_X_flat[train_y==1]
my_train_X_flat2 = train_X_flat[train_y==2]
my_train_X_flat1 = my_train_X_flat1[0:3]
my_train_X_flat2 = my_train_X_flat2[0:3]
my_train_X_flat  = np.concatenate((my_train_X_flat1, my_train_X_flat2))

net     = Network()
N0      = SpikeGeneratorGroup(X_size, [0], [0]*ms)
N0mon   = SpikeMonitor(N0)
net.add(N0)
net.add(N0mon)

N1      = NeuronGroup(2, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=10*ms, method='exact')
N1mon   = SpikeMonitor(N1)
N1state = StateMonitor(N1, ['v', 'dynThreshold'], record=True)

net.add(N1)
net.add(N1mon)
net.add(N1state)


N1.tau  = [5, 5]*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                     #you can also increase the spacex0x1 and keep tau to 10ms for example

N1.taus = [30, 30]*ms
N1.taut = [50, 50]*ms
N1.v    = [0]
N1.a    = [0]
N1.dynThreshold = [Threshold]

S = Synapses(N0, N1,
                    '''
                    w : 1
                    wmin : 1
                    wmax : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.01
                    w =  clip(w+apost, wmin, wmax)
                    ''',
                    on_post='''
                    apost += -0.025
                    w = clip(w+apre, wmin, wmax)
                    ''',
                    method='linear')

S.connect(True)
S.w = (Threshold+0.05)/(X_size/16) #as soon as it spikes, the output spikes too
S.wmax[:, 0] = 1
S.wmin[:, 0] = 0
S.wmax[:, 1] = 1
S.wmin[:, 1] = 0
S.delay[:, 0] = 1*ms
S.delay[:, 1] = 2*ms

Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

net.add(S)
net.add(Sstate)


S2 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post += w
                    ''',
                    method='linear')

S2.connect(i=[0,1], j=[1,0])
S2.w[0, 1] = -1
S2.w[1, 0] = -1
S2.delay[0, 1] = 0*ms
S2.delay[1, 0] = 0*ms
S2state  = StateMonitor(S2, ['w'], record=True)

net.add(S2)
net.add(S2state)

ts = 0
for x_flat in my_train_X_flat:

    '''
    each pixel from 0 to 255
    is encoded in 255 to 1 (0 are ignored)
    strong values comes earlier
    255 becomes 10ms
    1   becomes 40us
    '''
    n0_s    = np.where(x_flat > 0)[0]
    n0_t    = ts_time + (256 - x_flat[n0_s])*10/255

    n0_s_list.append(n0_s)
    n0_t_list.append(n0_t)

    N0.set_spikes(n0_s, n0_t*ms)

    net.run(single_example_time)

    n0_mon_list.append(N0mon)
    n1_mon_list.append(N1mon)
    n1_state_list.append(N1state)
    s1_ex_state_list.append(Sstate)

    ts_time = ts_time + space_between_samples

    if(n0_debug):
        for k in range(X_size):
            print('Input Neuron ' + str(k) + ' spiked ' + str(len(N0mon.spike_trains()[k]/ms)))
            print(N0mon.spike_trains()[k]/ms)



plt.figure()
for N0mon in n0_mon_list[0:3]:

    for k in range(X_size):
        N0mon_times_nk_plot   = N0mon.spike_trains()[k][N0mon.spike_trains()[k]/ms < 3*space_between_samples]
        N0mon_nspikes_nk_plot = np.ones(size(N0mon_times_nk_plot))*k
        plt.plot(N0mon_times_nk_plot/ms, N0mon_nspikes_nk_plot, '.k')


for N0mon in n0_mon_list[3:6]:

    for k in range(X_size):
        N0mon_times_nk_plot   = N0mon.spike_trains()[k][N0mon.spike_trains()[k]/ms < 6*space_between_samples]
        N0mon_times_nk_plot   = N0mon_times_nk_plot[N0mon_times_nk_plot/ms >= 3*space_between_samples]
        N0mon_nspikes_nk_plot = np.ones(size(N0mon_times_nk_plot))*k
        plt.plot(N0mon_times_nk_plot/ms, N0mon_nspikes_nk_plot, '.b')

plt.ylim((0,X_size))
plt.xlim((0, 6*space_between_samples))
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.figure()
ax1= plt.subplot(411)
for k in range(len(n1_state_list)):
    plt.plot(n1_state_list[k].t/ms, n1_state_list[k].v[0], label='N1,0')
    plt.plot(n1_state_list[k].t/ms, n1_state_list[k].dynThreshold[0], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('V')

plt.subplot(412, sharex = ax1)
for k in range(len(n1_state_list)):
    plt.plot(n1_state_list[k].t/ms, n1_state_list[k].v[1], label='N1,1')
    plt.plot(n1_state_list[k].t/ms, n1_state_list[k].dynThreshold[1], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('V')


plt.subplot(413, sharex = ax1)
for k in range(len(n1_state_list)):
    for w in range(10):
        plt.plot(s1_ex_state_list[k].t/ms, s1_ex_state_list[k].w[392+10+w])
plt.legend();

plt.subplot(414, sharex = ax1)

for k in range(len(n1_state_list)):
    for w in range(10):
        plt.plot(s1_ex_state_list[k].t/ms, s1_ex_state_list[k].w[X_size+392+10+w])
plt.legend();


plt.show()

exit()

print('Output Neuron 0 spiked ' + str(len(N1mon.spike_trains()[0]/ms)))
print('Output Neuron 1 spiked ' + str(len(N1mon.spike_trains()[1]/ms)))

print('Output spikes Neuron 0')
print(N1mon.spike_trains()[0]/ms)
print('Output spikes Neuron 1')
print(N1mon.spike_trains()[1]/ms)

N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*2
N1mon_times_n1_plot   = N1mon.spike_trains()[1][N1mon.spike_trains()[1]/ms < plot_stop_ms]
N1mon_nspikes_n1_plot = np.ones(size(N1mon_times_n1_plot))*3

plt.figure()

ax1= plt.subplot(511)
plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.r')
plt.plot(N1mon_times_n1_plot/ms, N1mon_nspikes_n1_plot, '.b')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.subplot(512, sharex = ax1)
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.dynThreshold[0][plot_start_index:plot_stop_index], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(513, sharex = ax1)
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[1][plot_start_index:plot_stop_index], label='N1,1')
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.dynThreshold[1][plot_start_index:plot_stop_index], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(514, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='0-0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='1-0')
plt.legend();

plt.subplot(515, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[2][plot_start_index:plot_stop_index], label='0-1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[3][plot_start_index:plot_stop_index], label='1-1')
plt.legend();

stop()

#indices = np.append(indices, N1mon_nspikes_n0_plot)
#times   = np.append(times, N1mon_times_n0_plot/ms)
#
#N0    = SpikeGeneratorGroup(3, indices, times*ms)
#N0mon = SpikeMonitor(N0)
#
#N1      = NeuronGroup(1, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=5*ms, method='exact')
#N1mon   = SpikeMonitor(N1)
#N1state = StateMonitor(N1, ['v'], record=True)
#N1.tau  = [2]*ms
#N1.taus = [3]*ms
#N1.v    = [0]
#N1.a    = [0]
#N1.dynThreshold = [Threshold]
#
#S = Synapses(N0, N1,
#                    '''
#                    w : 1
#                    wmin : 1
#                    wmax : 1
#                    dapre/dt = -apre/taupre : 1 (clock-driven)
#                    dapost/dt = -apost/taupost : 1 (clock-driven)
#                    ''',
#                    on_pre='''
#                    v_post += w
#                    apre += 0.1
#                    w =  clip(w+apost, wmin, wmax)
#                    ''',
#                    on_post='''
#                    apost += -0.25
#                    w = clip(w+apre, wmin, wmax)
#                    ''',
#                    method='linear')
#
#S.connect(i=[0,1,2], j=[0,0,0])
#S.w[0, 0] = 0.8
#S.w[1, 0] = 0.8
#S.w[2, 0] = -0.6
#S.delay[0, 0] = 5*ms
#S.delay[1, 0] = 5*ms
#S.delay[2, 0] = 0*ms
##the output of the N2 (which is the one that learned x1>x2) should be faster to inhibit the membrane
#S.wmax[:, 0] = 1
#S.wmin[:, 0] = -1
#
#
#
#Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
#visualise_connectivity(S)
#
#run(400*ms)
#
#plot_start_ms = 0
#plot_stop_ms  = plot_start_ms + 200
#
#plot_start_index = plot_start_ms*10
#plot_stop_index  = plot_stop_ms*10
#
#N0mon_times_n0_plot   = N0mon.spike_trains()[0][N0mon.spike_trains()[0]/ms < plot_stop_ms]
#N0mon_times_n1_plot   = N0mon.spike_trains()[1][N0mon.spike_trains()[1]/ms < plot_stop_ms]
#N0mon_times_n2_plot   = N0mon.spike_trains()[2][N0mon.spike_trains()[2]/ms < plot_stop_ms]
#N0mon_nspikes_n0_plot = np.ones(size(N0mon_times_n0_plot))*0
#N0mon_nspikes_n1_plot = np.ones(size(N0mon_times_n1_plot))*1
#N0mon_nspikes_n2_plot = np.ones(size(N0mon_times_n2_plot))*2
#
#N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
#N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*3
#
#plt.figure()
#
#ax1= plt.subplot(411)
#plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
#plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
#plt.plot(N0mon_times_n2_plot/ms, N0mon_nspikes_n2_plot, '.r')
#plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.b')
#plt.xlabel('Time (ms)')
#plt.ylabel('Neuron index');
#
#plt.subplot(412, sharex = ax1)
#plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
#plt.xlabel('Time (ms)')
#plt.ylabel('v')
#
#plt.subplot(413, sharex = ax1)
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='N1 W 0')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='N1 W 1')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[2][plot_start_index:plot_stop_index], label='N1 W 2')
#plt.legend();
#
#
#plt.subplot(414, sharex = ax1)
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[0][plot_start_index:plot_stop_index], label='N1 W 0')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[1][plot_start_index:plot_stop_index], label='N1 W 1')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[2][plot_start_index:plot_stop_index], label='N1 W 2')
#plt.legend();


plt.show();
