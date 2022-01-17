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

end_plot = argv_start_plot+argv_duration_plot


print_input_stream = False
print_output_membrana = True
print_traces = True
print_stats = False

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


###shuffle

#index_c1 = 0
#index_c2 = 1
#train_Xs = train_X
#train_ys = train_y
#
#for index in np.arange(0,X_Train_Samples):
#
#    if train_y[index] == 1:
#        train_Xs[index_c1] = train_X[index]
#        train_ys[index_c1] = train_y[index]
#        index_c1 = index_c1 + 2
#
#    elif train_y[index] == 2:
#        train_Xs[index_c2] = train_X[index]
#        train_ys[index_c2] = train_y[index]
#        index_c2 = index_c2 + 2


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

taupre  = 5 * ms
taupost = 5 * ms

Threshold_cost = 0.06
V_reset = -0.1
A_reset = +0.1


eqs = '''
dv/dt = -v/tau + a/taus: 1 (unless refractory)
da/dt = -a/taus: 1
tau : second
taus : second
taut : second
'''
eqs_reset = '''
                v = V_reset
                a = A_reset
            '''



n0_debug = False


net     = Network()

N0_Neurons = X_size; #28x28
N0         = SpikeGeneratorGroup(N0_Neurons, [0], [0]*ms)

N1_Neurons = 2;

N1      = NeuronGroup(N1_Neurons, eqs, threshold='v>Threshold_cost', reset=eqs_reset, refractory=10*ms, method='exact')

N1.tau  = 10*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                     #you can also increase the spacex0x1 and keep tau to 10ms for example

N1.taus = 30*ms
N1.taut = 150*ms
N1.v    = 0
N1.a    = 0

S = Synapses(N0, N1,
                    '''
                    w : 1
                    wmin : 1
                    wmax : 1
                    reward : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post = v_post+w
                    apre += reward*(0.0005)
                    w =  clip(w+apost,wmin,wmax)
                    ''',
                    on_post='''
                    apost += reward*(0.0005)
                    w =  clip(w+apre,wmin,wmax)
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


weight_matrix = np.ones((N0_Neurons,N1_Neurons))*np.random.random_sample((N0_Neurons,N1_Neurons))
row_sums = weight_matrix.sum(axis=0)
weight_matrix = weight_matrix / row_sums

for n0 in range(N0_Neurons):
    for n1 in range(N1_Neurons):
        S.w[n0,n1] = weight_matrix[n0,n1] #as soon as it spikes, the output spikes too

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
S2.w[0, 1] = -Threshold_cost*3
S2.w[1, 0] = -Threshold_cost*3
S2.delay[0, 1] = 0*ms
S2.delay[1, 0] = 0*ms

net.add(N0)
net.add(N1)
net.add(S)
#net.add(S2)

start_mon = argv_start_plot*single_example_time



#simulate only first 6 samples
lim_num_samples = argv_num_training_example
my_train_X_flat  = train_X_flat[0:lim_num_samples];

print("Network created.... Start training it with " + str(lim_num_samples) + " samples")

ts_time = 0
n0_s_list = []
n0_t_list = []
i_count = 0

max_w = -1
min_w = 1

stat_freq  = np.zeros((10, my_train_X_flat.shape[1]))
stat_power = np.zeros((10, my_train_X_flat.shape[1]))

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

    stat_freq[train_y[i_count]][n0_s]  = stat_freq[train_y[i_count]][n0_s] + 1
    stat_power[train_y[i_count]]       = stat_power[train_y[i_count]] + x_flat

    if(i_count % 100 == 0):
        print("Trained " + str(i_count) + " samples")



    if(print_input_stream):
        n0_s_list.append(n0_s)
        n0_t_list.append(n0_t)

    N0.set_spikes(n0_s, n0_t*ms)


    for n1 in range(N1_Neurons):
        if (n1 == 0):
            reward_s = +1 if train_y[i_count] == 1 else -1.3
        elif (n1 == 1):
            reward_s = +1 if train_y[i_count] == 2 else -1.3

        for n0 in range(N0_Neurons):
            S.reward[n0,n1] = reward_s

    if(ts_time == start_mon):
        print("Add monitors at time " + str(ts_time))
        N0mon    = SpikeMonitor(N0)
        N1mon    = SpikeMonitor(N1)
        N1state  = StateMonitor(N1, ['v'], record=True, dt=argv_step_mon*ms)
        Sstate   = StateMonitor(S, ['w','apre', 'apost'], record=True, dt=argv_step_mon*ms)

        if(print_input_stream):
            net.add(N0mon)

        net.add(N1mon)
        if(print_output_membrana):
            net.add(N1state)
        net.add(Sstate)



    net.run(single_example_time*ms)

    for n0 in range(N0_Neurons):
        for n1 in range(N1_Neurons):
            weight_matrix[n0,n1] = S.w[n0,n1]

    if(weight_matrix.max() > max_w):
        max_w = weight_matrix.max()

    if(weight_matrix.min() < min_w):
        min_w = weight_matrix.min()


    row_sums = weight_matrix.sum(axis=0)
    weight_matrix = weight_matrix / row_sums

    for n0 in range(N0_Neurons):
        for n1 in range(N1_Neurons):
            S.w[n0,n1] = weight_matrix[n0,n1]




    ts_time = ts_time + single_example_time
    i_count = i_count + 1
    if(n0_debug):
        for k in range(X_size):
            print('Input Neuron ' + str(k) + ' spiked ' + str(len(N0mon.spike_trains()[k]/ms)))
            print(N0mon.spike_trains()[k]/ms)

print("Network trained....")

if(print_stats):
    plt.figure()
    plt.title("Pixel Frequency Class")
    for c in classes:
        color = 'k' if c == 1 else 'b'
        plt.stem(stat_freq[c],linefmt=color, markerfmt=color+'o', label='class ' + str(c))
    plt.legend()

    plt.figure()
    plt.title("Pixel Power Class")
    for c in classes:
        color = 'k' if c == 1 else 'b'
        plt.stem(stat_power[c],linefmt=color, markerfmt=color+'o', label='class ' + str(c))
    plt.legend()


for c in classes:
    color = 'k' if c == 1 else 'b'
    print("pixel max frequency class " + str(c))
    print("stat_freq[class][" + str(stat_freq[c].argmax()) + "] = " + str(stat_freq[c].max()))
    print("power is " + str(stat_power[c][stat_freq[c].argmax()]))



if(print_input_stream):
    plt.figure()
    plt.title("Input Neuron Stream")

    for i_count in np.arange(argv_start_plot,end_plot):

        if(print_input_stream):
            color = '.k' if train_y[i_count] == 1 else '.b'
            for k in range(N0_Neurons):
                sample_time_condition = (N0mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N0mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
                N0mon_times_nk_plot   = N0mon.spike_trains()[k][sample_time_condition]
                N0mon_nspikes_nk_plot = np.ones(size(N0mon_times_nk_plot))*k
                plt.plot(N0mon_times_nk_plot/ms, N0mon_nspikes_nk_plot, color)
        #else:
        #        color = 'k-' if train_y[i_count] == 1 else 'b-'
        #        times = np.arange((i_count)*single_example_time,(i_count+1)*single_example_time,0.1)
        #        plt.plot(times, np.ones(size(times))*(train_y[i_count]), color)

    plt.ylim((-0.5,N0_Neurons))
    plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');


step=argv_step_plot;



if(print_output_membrana):
    plt.figure()

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
                ax1 = plt.subplot(211+n1, sharex = ax1)

            ax1.set_title("Output Neuron Membrana neuron " + str(n1))

            plt.plot(N1state_times_no_plot/ms, N1state.v[n1][sample_time_index], color=color, label='N1::'+str(n1))

            plt.xlabel('Time (ms)')
            plt.ylabel('V')
            plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))


#MOST TWO FREQUENTS
stat_freq2 = stat_freq.copy()

stat_freq2[1][stat_freq[1].argmax()] = 0
stat_freq2[2][stat_freq[2].argmax()] = 0

w1 = [ stat_freq[1].argmax(),
        stat_freq2[1].argmax(),
     ]

w2 = [ stat_freq[2].argmax(),
         stat_freq2[2].argmax(),
     ]

weights_to_plot = [ w1,
                    w2
                  ]

print(weights_to_plot[0])
print(weights_to_plot[1])

for n1 in range(N1_Neurons):

    plt.figure()

    for i_count in np.arange(argv_start_plot,end_plot):

        color = 'k' if train_y[i_count] == 1 else 'b'
        sample_time_condition = (Sstate.t/ms < (i_count+1)*single_example_time) & (Sstate.t/ms >= (i_count)*single_example_time)
        sample_time_index     = np.where(sample_time_condition)[0]
        Sstate_times_no_plot  = Sstate.t[sample_time_condition]

        sample_time_index     = sample_time_index[0:-1:step];
        Sstate_times_no_plot  = Sstate_times_no_plot[0:-1:step];

        num_suplots = 211 if print_traces else 111;

        ax1 = plt.subplot(num_suplots)

        ax1.set_title("N" + str(n1) + " Weights")
        for weights in weights_to_plot[n1]:
            plt.plot(Sstate_times_no_plot/ms, Sstate.w[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))

        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((min_w*1.1,max_w*1.1))
        plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))

        if(print_traces):
            ax1=plt.subplot(num_suplots+1, sharex = ax1)
            ax1.set_title("N" + str(n1) + " Traces")

            for weights in weights_to_plot[n1]:
                plt.plot(Sstate_times_no_plot/ms, Sstate.apre[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))
                plt.plot(Sstate_times_no_plot/ms, Sstate.apost[weights+(n1*N0_Neurons)][sample_time_index], color=color, label='N1::'+str(n1))

            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.0012,+0.0008))
            plt.xlim((argv_start_plot*single_example_time, end_plot*single_example_time))




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

