from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import sys

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





if(len(sys.argv)<2):
    argv_num_training_example = 15
else:
    argv_num_training_example = int(sys.argv[1])

if(len(sys.argv)<3):
    argv_step_mon = 0.1
else:
    argv_step_mon = float(sys.argv[2])

if(len(sys.argv)<4):
    argv_start_plot = 0
else:
    argv_start_plot = int(sys.argv[3])

if(len(sys.argv)<5):
    argv_duration_plot = argv_num_training_example
else:
    argv_duration_plot = int(sys.argv[4])

if(len(sys.argv)<6):
    argv_step_plot = 1
else:
    argv_step_plot = int(sys.argv[5])

print('Start simulation with : ')
print("argv_num_training_example= " + str(argv_num_training_example))
print("argv_step_mon= " + str(argv_step_mon))
print("argv_start_plot= " + str(argv_start_plot))
print("argv_duration_plot= " + str(argv_duration_plot))
print("argv_step_plot= " + str(argv_step_plot))

print_input_stream = False
print_output_membrana = True
print_traces = False

start_scope()

np.random.seed(2021)

use_only_1_and_2 = True

(train_X, train_y), (test_X, test_y) = mnist.load_data()

classes       = np.arange(0,10)
classes_color = []

if(use_only_1_and_2):
    train_X = train_X[(train_y==1) | (train_y==2)]
    train_y = train_y[(train_y==1) | (train_y==2)]
    test_X  = test_X[(test_y==1) | (test_y==2)]
    test_y  = test_y[(test_y==1) | (test_y==2)]
    classes = np.arange(1,3)

train_X_flat = np.reshape(train_X,(train_X.shape[0], 28*28))
test_X_flat = np.reshape(test_X,(test_X.shape[0], 28*28))

plt_example_mnist = False

X_size          = 28*28
X_Train_Samples = train_X.shape[0]

print("train_X contans " + str(X_Train_Samples))

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
    for k in classes:
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

single_example_time   = 25

taupre = 2 * ms
taupost = 5 * ms


eqs = '''
dv/dt = -v/tau + a/taus: 1
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

n0_debug = False


net     = Network()

N0_Neurons = X_size; #28x28
N0         = SpikeGeneratorGroup(N0_Neurons, [0], [0]*ms)

N1_Neurons = 2;

N1      = NeuronGroup(N1_Neurons, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=10*ms, method='exact')

N1.tau  = [10, 10]*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                     #you can also increase the spacex0x1 and keep tau to 10ms for example

N1.taus = [30, 30]*ms
N1.taut = [150, 150]*ms
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

#all N0 neurons connected to N1::0
i0 = np.arange(0,N0_Neurons)
j0 = np.ones(size(i0))*0

i1 = np.arange(0,N0_Neurons)
j1 = np.ones(size(i1))*1

i = np.concatenate((i0, i1))
j = np.concatenate((j0, j1))
j = j.astype(int)

S.connect(i=i, j=j)

for n0 in range(N0_Neurons):
    for n1 in range(N1_Neurons):
        S.w[n0,n1] = (Threshold+0.05)/(X_size/16)*(1.0 + np.random.random_sample()/4) #as soon as it spikes, the output spikes too

S.wmax[:, 0] = 1
S.wmin[:, 0] = 0
S.wmax[:, 1] = 1
S.wmin[:, 1] = 0
S.delay[:, 0] = 1*ms
S.delay[:, 1] = 1*ms


S2 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post += w
                    ''',
                    method='linear')

S2.connect(i=[0,1], j=[1,0])
S2.w[0, 1] = -3
S2.w[1, 0] = -3
S2.delay[0, 1] = 0*ms
S2.delay[1, 0] = 0*ms

net.add(N0)
net.add(N1)
net.add(S)
net.add(S2)

start_mon = argv_start_plot*single_example_time



#simulate only first 6 samples
lim_num_samples = argv_num_training_example
my_train_X_flat  = train_X_flat[0:lim_num_samples];

print("Network created.... Start training it with " + str(lim_num_samples) + " samples")

ts_time = 0
n0_s_list = []
n0_t_list = []
i_count = 0

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

    if(i_count % 100 == 0):
        print("Trained " + str(i_count) + " samples")

    i_count = i_count + 1

    if(print_input_stream):
        n0_s_list.append(n0_s)
        n0_t_list.append(n0_t)

    N0.set_spikes(n0_s, n0_t*ms)

    if(ts_time == start_mon):
        print("Add monitors at time " + str(ts_time))
        N0mon    = SpikeMonitor(N0)
        N1mon    = SpikeMonitor(N1)
        N1state  = StateMonitor(N1, ['v', 'dynThreshold'], record=True, dt=argv_step_mon*ms)
        Sstate   = StateMonitor(S, ['w', 'apre', 'apost'], record=True, dt=argv_step_mon*ms)
        S2state  = StateMonitor(S2, ['w'], record=True, dt=argv_step_mon*ms)

        if(print_input_stream):
            net.add(N0mon)

        net.add(N1mon)
        if(print_output_membrana):
            net.add(N1state)

        net.add(Sstate)

    net.run(single_example_time*ms)

    ts_time = ts_time + single_example_time

    if(n0_debug):
        for k in range(X_size):
            print('Input Neuron ' + str(k) + ' spiked ' + str(len(N0mon.spike_trains()[k]/ms)))
            print(N0mon.spike_trains()[k]/ms)

print("Network trained....")

plt.figure()
plt.title("Input Neuron Stream")

end_plot = argv_start_plot+argv_duration_plot

for i_count in np.arange(argv_start_plot,end_plot):

    if(print_input_stream):
        color = '.k' if train_y[i_count] == 1 else '.b'
        for k in range(N0_Neurons):
            sample_time_condition = (N0mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N0mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
            N0mon_times_nk_plot   = N0mon.spike_trains()[k][sample_time_condition]
            N0mon_nspikes_nk_plot = np.ones(size(N0mon_times_nk_plot))*k
            plt.plot(N0mon_times_nk_plot/ms, N0mon_nspikes_nk_plot, color)
    else:
            color = 'k-' if train_y[i_count] == 1 else 'b-'
            times = np.arange((i_count)*single_example_time,(i_count+1)*single_example_time,0.1)
            plt.plot(times, np.ones(size(times))*(train_y[i_count]), color)

if(print_input_stream):
    plt.ylim((-0.5,N0_Neurons))
else:
    plt.ylim((-0.5,size(classes)+1))

plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');


step=argv_step_plot;



if(print_output_membrana):
    plt.figure()

    plt.title("Output Neuron Membrana")


    for i_count in np.arange(argv_start_plot,end_plot):

        color = 'k' if train_y[i_count] == 1 else 'b'
        sample_time_condition = (N1state.t/ms < (i_count+1)*single_example_time) & (N1state.t/ms >= (i_count)*single_example_time)
        sample_time_index     = np.where(sample_time_condition)[0]
        N1state_times_no_plot = N1state.t[sample_time_condition]

        sample_time_index     = sample_time_index[0:-1:step];
        N1state_times_no_plot = N1state_times_no_plot[0:-1:step];

        for n1 in range(N1_Neurons):

            if(n1==0):
                ax1 = plt.subplot(211+n1)
            else:
                plt.subplot(211+n1, sharex = ax1)

            plt.plot(N1state_times_no_plot/ms, N1state.v[n1][sample_time_index], color=color, label='N1::'+str(n1))
            plt.plot(N1state_times_no_plot/ms, N1state.dynThreshold[n1][sample_time_index], color=color, label='Threshold')

            plt.xlabel('Time (ms)')
            plt.ylabel('V')
            plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))



plt.figure()

plt.title("N0-N1 Weights and Traces")


weights_to_plot = np.arange(392+10,392+10+10)
for i_count in np.arange(argv_start_plot,end_plot):

    color = 'k' if train_y[i_count] == 1 else 'b'
    sample_time_condition = (Sstate.t/ms < (i_count+1)*single_example_time) & (Sstate.t/ms >= (i_count)*single_example_time)
    sample_time_index     = np.where(sample_time_condition)[0]
    Sstate_times_no_plot  = Sstate.t[sample_time_condition]

    sample_time_index     = sample_time_index[0:-1:step];
    Sstate_times_no_plot  = Sstate_times_no_plot[0:-1:step];

    num_suplots = 411 if print_traces else 211;

    for n1 in range(N1_Neurons):

        if(n1==0):
            ax1 = plt.subplot(num_suplots)
        else:
            plt.subplot(num_suplots, sharex = ax1)

        for weights in weights_to_plot:
            plt.plot(Sstate_times_no_plot/ms, Sstate.w[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))

        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((-0.2,+0.2))
        plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))

        num_suplots = num_suplots + 1

        if(print_traces):
            plt.subplot(num_suplots, sharex = ax1)

            for weights in weights_to_plot:
                plt.plot(Sstate_times_no_plot/ms, Sstate.apre[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))
                plt.plot(Sstate_times_no_plot/ms, Sstate.apost[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))

            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.05,0.05))
            plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))

            num_suplots = num_suplots + 1



plt.figure()

plt.title("Output Neuron Stream")

print('Output Neuron 0 spiked ' + str(len(N1mon.spike_trains()[0]/ms)))
print('Output Neuron 1 spiked ' + str(len(N1mon.spike_trains()[1]/ms)))

for i_count in np.arange(argv_start_plot,end_plot):

    color = '*k' if train_y[i_count] == 1 else '*r'
    for k in range(N1_Neurons):
        sample_time_condition = (N1mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N1mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
        N1mon_times_nk_plot   = N1mon.spike_trains()[k][sample_time_condition]
        N1mon_nspikes_nk_plot = np.ones(size(N1mon_times_nk_plot))*k
        plt.plot(N1mon_times_nk_plot/ms, N1mon_nspikes_nk_plot, color)


plt.ylim((0,N1_Neurons))
plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))

plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.show()

exit();

