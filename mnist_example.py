from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import sys
import json
from decimal import Decimal

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

f_param = open('parameter.json')
 
# returns JSON object as
# a dictionary
parameter = json.load(f_param)

mydevice = parameter['device'];

if mydevice == "cpp":
    import brian2cuda
    set_device('cpp_standalone', build_on_run=False)
    print("Compiling for CPP")
elif mydevice == "cuda":
    import brian2cuda
    set_device("cuda_standalone", build_on_run=False)
    print("Compiling for CUDA")
else:
    print("Using Python")

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


def printmatrix(mymatrix, myfile):
    s = [[str(e) for e in row] for row in mymatrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table), file = myfile)

def scanmatrix(mymatrix, myfile):
    row = 0
    for line in myfile:
        col = 0
        for num in line.split():
            mymatrix[row,col] = Decimal(num)
            col = col + 1
        row = row + 1

figpath = './figures/'

print_input_stream = False
print_output_membrana = True
print_weights = True
print_traces = True
print_reward_stream = True
plt_example_mnist = False
use_only_1_and_2 = False
use_only_1_and_2_and_3 = False

start_scope()

np.random.seed(2021)


(train_X, train_y), (test_X, test_y) = mnist.load_data()

classes       = np.arange(0,10)
classes_color = ['y', 'k', 'b', 'r', 'g', 'c', 'm','k', 'b', 'r' ]
classes_sizes = np.arange(0,10)

for i in classes:
    classes_sizes[i] = len(train_y[train_y==i])

if(use_only_1_and_2):
    train_X = train_X[(train_y==1) | (train_y==2)]
    train_y = train_y[(train_y==1) | (train_y==2)]
    test_X  = test_X[(test_y==1)   | (test_y==2) ]
    test_y  = test_y[(test_y==1)   | (test_y==2) ]
    classes = np.arange(1,3)
elif(use_only_1_and_2_and_3):
    train_X = train_X[(train_y==1) | (train_y==2) | (train_y==3)]
    train_y = train_y[(train_y==1) | (train_y==2) | (train_y==3)]
    test_X  = test_X[(test_y==1)   | (test_y==2)  | (test_y==3) ]
    test_y  = test_y[(test_y==1)   | (test_y==2)  | (test_y==3) ]
    classes = np.arange(1,4)

train_X_flat = np.reshape(train_X,(train_X.shape[0], 28*28))
test_X_flat  = np.reshape(test_X,(test_X.shape[0], 28*28))

for i in classes:
    print('Class ' + str(i) + ' has number of samples ' + str( classes_sizes[i] ))

X_size          = 28*28
X_Train_Samples = train_X.shape[0]
X_Test_Samples  = test_X.shape[0]


N1_Neurons         = parameter['N1_Neurons']
N2_Neurons         = parameter['N2_Neurons']

training_example   = X_Train_Samples if parameter['training_example']==-1 else parameter['training_example'];
monitor_step       = parameter['monitor_step']
plot_step_time     = parameter['plot_step_time']
testing_example    = X_Test_Samples if parameter['testing_example']==-1 else parameter['testing_example'];
testing_phase      = parameter['testing_phase']

learning_1_phase   = parameter['learning_1_phase']
learning_2_phase   = parameter['learning_2_phase']

if not testing_phase:
    set_size       = training_example
else:
    set_size       = testing_example

plot_start_time    = parameter['plot_start_time'] if parameter['plot_start_time']>=0 else (set_size+parameter['plot_start_time'])
plot_duration_time = parameter['plot_duration_time'] if parameter['plot_duration_time'] <= set_size else set_size
end_plot           = plot_start_time+plot_duration_time


print('Start simulation with : ')
print("training_example= " + str(training_example))
print("monitor_step (ms)= " + str(monitor_step))
print("plot_start_time(ms)= " + str(plot_start_time))
print("plot_duration_time(ms)= " + str(plot_duration_time))
print("plot_step_time(ms)= " + str(plot_step_time))
print("end_plot(ms)= " + str(end_plot))
print("testing_phase= " + str(testing_phase))
print("learning_1_phase= " + str(learning_1_phase))
print("learning_2_phase= " + str(learning_2_phase))

if not testing_phase:
    print("train_X contans " + str(X_Train_Samples))

if(plt_example_mnist):
    plt.figure(1)
    for i in range(9):
        plt.subplot(330 + 1 + i)
        if not testing_phase:
            plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
        else:
            plt.imshow(test_X[i], cmap=plt.get_cmap('gray'))
    if not testing_phase:
        plt.savefig(figpath + '1_first_nine_train.png')
    else:
        plt.savefig(figpath + '1_first_nine_test.png')
    plt.close(1)


print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

if(plt_example_mnist):
    #print the first 9 values
    for k in classes:
        plt.figure(1)
        for i in range(9):
            plt.subplot(330 + 1 + i)
            if not testing_phase:
                plt.imshow( (train_X[train_y==k])[i], cmap=plt.get_cmap('gray'))
            else:
                plt.imshow( (test_X[test_y==k])[i], cmap=plt.get_cmap('gray'))
        if not testing_phase:
            plt.savefig(figpath + '2_first_nine_per_class_' + str(k) + '_train.png')
        else:
            plt.savefig(figpath + '2_first_nine_per_class_' + str(k) + '_test.png')
        plt.close(1)

'''
we want to have 40FPS
thus, here we use 25ms
10ms for the frame
and 10ms for the resting
'''

single_example_time   = 25


net     = Network()

N0_Neurons = X_size; #28x28
N0         = SpikeGeneratorGroup(N0_Neurons, [0], [0]*ms)

eqs = '''
dv/dt = -v/taul1 + a/tausl1: 1 (unless refractory)
da/dt = -a/tausl1: 1
taul1 : second
tausl1 : second
sumwl1 : 1
'''
eqs_reset = '''
                v = V_resetl1
                a = A_resetl1
            '''

tauprel1  = 7 * ms
taupostl1 = 7 * ms

Thresholdl1 = 0.052
V_resetl1   = -0.1
A_resetl1   = +0.1

N1         = NeuronGroup(N1_Neurons, eqs, threshold='v>Thresholdl1', reset=eqs_reset, refractory=10*ms, method='exact')

N1.taul1  = 9*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                 #you can also increase the spacex0x1 and keep tau to 10ms for example
N1.tausl1 = 20*ms
N1.v      = 0
N1.a      = 0

if learning_1_phase:
    S010 = Synapses(N0, N1,
                        '''
                        wmin : 1
                        wmax : 1
                        w    : 1
                        dapre/dt = -apre/tauprel1 : 1 (clock-driven)
                        dapost/dt = -apost/taupostl1 : 1 (clock-driven)
                        sumwl1_post = w   : 1   (summed)
                        ''',
                        on_pre='''
                        v_post += w
                        apre += (0.0003)
                        w = w/sumwl1_post
                        w = clip(w+apost,wmin,wmax)
                        ''',
                        on_post='''
                        apost += (-0.0005)
                        w = w/sumwl1_post
                        w = clip(w+apre,wmin,wmax)
                        ''',
                        method='linear')
else: ##keep the weights constant during learning of second layer or testing
    S010 = Synapses(N0, N1,
                        '''
                        w : 1
                        ''',
                        on_pre='''
                        v_post += w
                        ''',
                        method='linear')


i_syn = []
j_syn = []

for n1 in range(N1_Neurons):
    #all N0 neurons connected to n1
    i_n = np.arange(0,N0_Neurons)
    j_n = np.ones(size(i_n))*n1
    i_syn = np.concatenate((i_syn, i_n))
    j_syn = np.concatenate((j_syn, j_n))

i_syn = i_syn.astype(int)
j_syn = j_syn.astype(int)

S010.connect(i=i_syn, j=j_syn)

if learning_1_phase:
    S010.wmax     = Thresholdl1*0.5
    S010.wmin     = 0

minDelay   = 0*ms
maxDelay   = 3*ms
deltaDelay = maxDelay - minDelay
S010.delay = 'minDelay + rand() * deltaDelay'

if learning_1_phase:
    weight_matrix = np.zeros((N0_Neurons,N1_Neurons))
    for n1 in np.arange(N1_Neurons):
        weight_matrix[:,n1] = np.random.normal(loc=1/N0_Neurons, scale=1/N0_Neurons*0.1, size=(N0_Neurons))
else:
    weight_matrix = np.zeros((N1_Neurons,28,28))
    for n1 in np.arange(N1_Neurons):
        sourceFile = open('./Weights/weights_'+str(n1)+'.txt', 'r')
        scanmatrix(weight_matrix[n1],sourceFile)
        sourceFile.close()

if learning_2_phase:
    weight_matrix2 = np.zeros((N1_Neurons,N2_Neurons))
    for n2 in np.arange(N2_Neurons):
        weight_matrix2[:,n2] = np.random.normal(loc=1/N1_Neurons, scale=1/N1_Neurons*0.1, size=(N1_Neurons))
else:
    print("NOT IMPLEMENTED YET")


S110 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post = clip(v_post+w,V_resetl1,1)
                    ''',
                    method='linear')

S110.connect('i != j')
S110.w = -0.15
S110.delay = 0*ms

net.add(N0)
net.add(N1)
net.add(S010)
net.add(S110)

if learning_2_phase:

    Thresholdl2 = Thresholdl1*0.5

    N2        = NeuronGroup(N2_Neurons, eqs, threshold='v>Thresholdl2', reset=eqs_reset, refractory=10*ms, method='exact')

    N2.taul1  = 9*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                     #you can also increase the spacex0x1 and keep tau to 10ms for example
    N2.tausl1 = 20*ms
    N2.v      = 0
    N2.a      = 0
    tau_reward = 25*ms

    S120 = Synapses(N1, N2,
                        '''
                        wmin : 1
                        wmax : 1
                        w    : 1
                        reward : 1
                        punish : 1
                        dapre/dt  = -apre/tauprel1 : 1 (clock-driven)
                        dapost/dt = -apost/taupostl1 : 1 (clock-driven)
                        ''',
                        on_pre='''
                        v_post += w
                        apre += reward*(0.0003) - punish*(0.0003)
                        w = clip(w+apost,wmin,wmax)
                        ''',
                        on_post='''
                        apost += reward*(0.0003) - punish*(0.0003)
                        w = clip(w+apre,wmin,wmax)
                        ''',
                        method='linear')

    i_syn = []
    j_syn = []

    for n2 in range(N2_Neurons):
        #all N1 neurons connected to n2
        i_n = np.arange(0,N1_Neurons)
        j_n = np.ones(size(i_n))*n2
        i_syn = np.concatenate((i_syn, i_n))
        j_syn = np.concatenate((j_syn, j_n))

    i_syn = i_syn.astype(int)
    j_syn = j_syn.astype(int)

    S120.connect(i=i_syn, j=j_syn)

    S120.wmax     = Thresholdl1*0.5
    S120.wmin     = 0

    Reward_Neurons = 2*N2_Neurons;

    NReward = SpikeGeneratorGroup(Reward_Neurons, [0], [0]*ms)

    ##Reward has 10 Neurons, S120 has N1_Neurons*N2_Neurons connections
    SRS120 = Synapses(NReward, S120,
                        '''
                         ''',
                        on_pre='''
                        reward_post = int(i<N2_Neurons)
                        punish_post = int(i>=N2_Neurons)
                        ''',
                        method='linear')


    i_syn = []
    j_syn = []

    for nr in range(Reward_Neurons):
        #nr 0 connected to synapsys 0...N1_Neurons-1 which are the ones connected N1 to neuron 0 of N2
        #nr 1 connected to synapsys N1...2*N1_Neurons-1 which are the ones connected N1 to neuron 1 of N2
        # like a flat matrix, etc
        offset = nr % N2_Neurons
        j_n = np.arange(0,N1_Neurons)+ (offset*N1_Neurons)
        i_n = np.ones(size(j_n))*nr
        i_syn = np.concatenate((i_syn, i_n))
        j_syn = np.concatenate((j_syn, j_n))

    i_syn = i_syn.astype(int)
    j_syn = j_syn.astype(int)

    SRS120.connect(i=i_syn, j=j_syn)

    net.add(N2)
    net.add(NReward)
    net.add(S120)
    net.add(SRS120)


start_mon = plot_start_time*single_example_time

if not testing_phase:
    my_set_X_flat  = train_X_flat[0:set_size];
    print("Network created.... Start training it with " + str(set_size) + " samples")
else:
    my_set_X_flat  = test_X_flat[0:set_size];
    print("Network created.... Start testing it with " + str(set_size) + " samples")


net.run(0*ms)

#for n0 in range(N0_Neurons):
#    for n1 in range(N1_Neurons):
#        S010.w['i==n0 and j==n1'] = weight_matrix[n0,n1] #as soon as it spikes, the output spikes too
#

weight_matrix_flat  = np.reshape(weight_matrix,(N0_Neurons*N1_Neurons))
S010.w              = weight_matrix_flat
weight_matrix2_flat = np.reshape(weight_matrix2,(N1_Neurons*N2_Neurons))
S120.w              = weight_matrix2_flat/10/3*2


n0_s_list = []
n0_t_list = []
nr_s_list = []
nr_t_list = []

ts_time = 0
i_count = 0

stat_freq  = np.zeros((10, my_set_X_flat.shape[1]))
stat_power = np.zeros((10, my_set_X_flat.shape[1]))

for x_flat in my_set_X_flat:

    '''
    each pixel from 0 to 255
    is encoded in 255 to 1 (0 are ignored)
    strong values comes earlier
    255 becomes 10ms
    1   becomes 40us
    '''

    n0_s    = np.where(x_flat > 0)[0]
    n0_t    = ts_time + (256 - x_flat[n0_s])*10/255

    #reward
    nr_s    = []
    nr_s.append(train_y[i_count] if train_y[i_count] <= 1 else 1)

    nr_t    = []
    nr_t.append(ts_time) ###reward

    for nr in range(int(Reward_Neurons/2)):
        if nr != nr_s[0]:
            nr_s.append(nr + int(Reward_Neurons/2))
            nr_t.append(ts_time) ###punishment

    n0_s_list = np.concatenate((n0_s_list, n0_s))
    n0_t_list = np.concatenate((n0_t_list, n0_t))

    nr_s_list = np.concatenate((nr_s_list, nr_s))
    nr_t_list = np.concatenate((nr_t_list, nr_t))

    if(i_count % 1000 == 0):
        print("Trained " + str(i_count) + " samples")

    ts_time = ts_time + single_example_time
    i_count = i_count + 1


N0.set_spikes(n0_s_list, n0_t_list*ms)

if(print_input_stream):
    N0mon    = SpikeMonitor(N0)
    net.add(N0mon)

if learning_2_phase:
    NReward.set_spikes(nr_s_list, nr_t_list*ms)
    if(print_reward_stream):
        NRewardmon    = SpikeMonitor(NReward)
        net.add(NRewardmon)

N1mon    = SpikeMonitor(N1)
net.add(N1mon)

N2mon    = SpikeMonitor(N2)
net.add(N2mon)


if(print_output_membrana):
    N1state  = StateMonitor(N1, ['v'], record=np.arange(N1_Neurons), dt=monitor_step*ms)
    N2state  = StateMonitor(N2, ['v'], record=np.arange(N2_Neurons), dt=monitor_step*ms)
    net.add(N1state)
    net.add(N2state)

if(print_traces or print_weights):
    if learning_1_phase:
        S010state   = StateMonitor(S010, ['w','apre', 'apost'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    else:
        S010state   = StateMonitor(S010, ['w'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    if learning_2_phase:
        S120state   = StateMonitor(S120, ['w','apre', 'apost', 'reward'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    else:
        S120state   = StateMonitor(S120, ['w'], record=np.arange(N1_Neurons*N2_Neurons), dt=monitor_step*ms)

    net.add(S010state)
    net.add(S120state)

net.run(set_size*single_example_time*ms)

if not testing_phase:
    print("Network trained....")
else:
    print("Network tested....")

if mydevice == "cpp":
    device.build( directory='outputcpp', compile = True, run = True, debug=False, clean = True)
elif mydevice == "cuda":
    device.build( directory='output', compile = True, run = True, debug=False, clean = True)


if(print_input_stream):
    plt.figure(1)
    plt.title("Input Neuron Stream")
    color = '.k'
    for k in range(N0_Neurons):
        sample_time_condition = (N0mon.spike_trains()[k]/ms < end_plot*single_example_time) & (N0mon.spike_trains()[k]/ms >= plot_start_time*single_example_time)
        N0mon_times_nk_plot   = N0mon.spike_trains()[k][sample_time_condition]
        N0mon_nspikes_nk_plot = np.ones(size(N0mon_times_nk_plot))*k
        plt.plot(N0mon_times_nk_plot/ms, N0mon_nspikes_nk_plot, color)
    plt.ylim((-0.5,N0_Neurons))
    plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    if not testing_phase:
        plt.savefig(figpath + '8_input_neurons.png')
    else:
        plt.savefig(figpath + '8_test_input_neurons.png')

    plt.close(1)

if(print_reward_stream):
    plt.figure(1)
    plt.title("Reward Neuron Stream")
    color = '*r'
    for k in range(Reward_Neurons):
        sample_time_condition = (NRewardmon.spike_trains()[k]/ms < end_plot*single_example_time) & (NRewardmon.spike_trains()[k]/ms >= plot_start_time*single_example_time)
        NRewardmon_times_nk_plot   = NRewardmon.spike_trains()[k][sample_time_condition]
        NRewardmon_nspikes_nk_plot = np.ones(size(NRewardmon_times_nk_plot))*k
        plt.plot(NRewardmon_times_nk_plot/ms, NRewardmon_nspikes_nk_plot, color)
    plt.ylim((-0.5,Reward_Neurons))
    plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    if not testing_phase:
        plt.savefig(figpath + '8_reward_neurons.png')

    plt.close(1)


step=int(plot_step_time*10);

if(print_output_membrana):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("Layer 1 Neuron Membrana neuron " + str(n1))

        time_plot = []
        state_plot = []

        sample_time_condition = (N1state.t/ms < end_plot*single_example_time) & (N1state.t/ms >= plot_start_time*single_example_time)
        sample_time_index     = np.where(sample_time_condition)[0]
        N1state_times_no_plot = N1state.t[sample_time_condition]

        sample_time_index     = sample_time_index[0:-1:step];
        N1state_times_no_plot = N1state_times_no_plot[0:-1:step];

        time_plot  = N1state_times_no_plot/ms
        state_plot = N1state.v[n1][sample_time_index]

        plt.plot(time_plot, state_plot, label='N1::'+str(n1))

        plt.xlabel('Time (ms)')
        plt.ylabel('V')
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)
    if not testing_phase:
        plt.savefig(figpath + '9_output_membrana.png')
    else:
        plt.savefig(figpath + '9_test_output_membrana.png')
    plt.close(1)

if(print_output_membrana):
    plt.figure(1)

    for n2 in range(N2_Neurons):

        ax1 = plt.subplot2grid((N2_Neurons,1), (n2,0))
        ax1.set_title("Output Neuron Membrana neuron " + str(n2))

        time_plot = []
        state_plot = []

        sample_time_condition = (N2state.t/ms < end_plot*single_example_time) & (N2state.t/ms >= plot_start_time*single_example_time)
        sample_time_index     = np.where(sample_time_condition)[0]
        N2state_times_no_plot = N2state.t[sample_time_condition]

        sample_time_index     = sample_time_index[0:-1:step];
        N2state_times_no_plot = N2state_times_no_plot[0:-1:step];

        time_plot  = N2state_times_no_plot/ms
        state_plot = N2state.v[n2][sample_time_index]

        plt.plot(time_plot, state_plot, label='N2::'+str(n2))

        plt.xlabel('Time (ms)')
        plt.ylabel('V')
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)
    if not testing_phase:
        plt.savefig(figpath + '9_output_membrana_reward.png')
    else:
        plt.savefig(figpath + '9_test_output_membrana_reward.png')
    plt.close(1)



weights_to_plot = np.arange(675,700+1)


if(print_weights):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("N" + str(n1) + " Weights")

        min_w = +1
        max_w = -1

        for weights in weights_to_plot:
            time_plot = []
            state_plot = []

            sample_time_condition = (S010state.t/ms < end_plot*single_example_time) & (S010state.t/ms >= plot_start_time*single_example_time)
            sample_time_index     = np.where(sample_time_condition)[0]
            S010state_times_no_plot  = S010state.t[sample_time_condition]

            sample_time_index     = sample_time_index[0:-1:step];
            S010state_times_no_plot  = S010state_times_no_plot[0:-1:step];

            if S010state.w[weights+(n1*N0_Neurons)][sample_time_index].min() < min_w:
                min_w = S010state.w[weights+(n1*N0_Neurons)][sample_time_index].min();

            if S010state.w[weights+(n1*N0_Neurons)][sample_time_index].max() > max_w:
                max_w = S010state.w[weights+(n1*N0_Neurons)][sample_time_index].max();

            time_plot  = S010state_times_no_plot/ms
            state_plot = S010state.w[weights+(n1*N0_Neurons)][sample_time_index]

            plt.plot(time_plot, state_plot,label='N1::'+str(n1))
            plt.grid(True)

        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((min_w*0.9,max_w*1.1))
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))

    if learning_1_phase:
        plt.savefig(figpath + '10_weights_stdp.png')
    else:
        plt.savefig(figpath + '10_test_weights.png')

    plt.close(1)

if learning_1_phase:
    weight_matrix = S010.w.get_item(item=np.arange(N0_Neurons*N1_Neurons))
    max_w = weight_matrix.max()
    for n1 in range(N1_Neurons):
        plt.figure(1)
        weight_img = np.reshape(S010.w[:,n1], (28,28));
        print("N" + str(n1) + " max: " + str(weight_img.max()) + " at index " + str(weight_img.argmax()))
        weight_img = weight_img/max_w*255
        weight_img = weight_img.astype(int)
        plt.imshow(weight_img, cmap=plt.get_cmap('gray'))
        plt.savefig(figpath + '10_weights_img_class_' + str(n1) + '.png')
        plt.close(1)


    weight_matrix = np.reshape(weight_matrix,(N1_Neurons,28,28))
    for n1 in np.arange(N1_Neurons):
        sourceFile = open('./Weights/weights_'+str(n1)+'.txt', 'w')
        #printmatrix(np.around(weight_matrix[n1], decimals=4),sourceFile)
        printmatrix(weight_matrix[n1],sourceFile)
        sourceFile.close()

weights2_to_plot = np.arange(0,N1_Neurons)

##np.set_printoptions(threshold=sys.maxsize)

if(print_weights):
    plt.figure(1)

    for n2 in range(N2_Neurons):

        ax1 = plt.subplot2grid((N2_Neurons,1), (n2,0))
        ax1.set_title("N" + str(n2) + " Weights - layer 2")

        min_w = +1
        max_w = -1

        for weights in weights2_to_plot:
            time_plot = []
            state_plot = []

            sample_time_condition = (S120state.t/ms < end_plot*single_example_time) & (S120state.t/ms >= plot_start_time*single_example_time)
            sample_time_index     = np.where(sample_time_condition)[0]
            S120state_times_no_plot  = S120state.t[sample_time_condition]

            sample_time_index        = sample_time_index[0:-1:step];
            S120state_times_no_plot  = S120state_times_no_plot[0:-1:step];

            if S120state.w[weights+(n2*N1_Neurons)][sample_time_index].min() < min_w:
                min_w = S120state.w[weights+(n2*N1_Neurons)][sample_time_index].min();

            if S120state.w[weights+(n2*N1_Neurons)][sample_time_index].max() > max_w:
                max_w = S120state.w[weights+(n2*N1_Neurons)][sample_time_index].max();

            time_plot  = S120state_times_no_plot/ms
            state_plot = S120state.w[weights+(n2*N1_Neurons)][sample_time_index]

            plt.plot(time_plot, state_plot,label='N2::'+str(n2))
            plt.grid(True)

        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((min_w*0.9,max_w*1.1))
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))

    if learning_2_phase:
        plt.savefig(figpath + '10_weights_stdp_reward.png')
    else:
        plt.savefig(figpath + '10_test_weights_reward.png')

    plt.close(1)

if(print_traces and learning_1_phase):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("N" + str(n1) + " Traces")


        for weights in weights_to_plot:
            time_plot = []
            state_plot = []
            state2_plot = []

            sample_time_condition = (S010state.t/ms < end_plot*single_example_time) & (S010state.t/ms >= plot_start_time*single_example_time)
            sample_time_index     = np.where(sample_time_condition)[0]
            S010state_times_no_plot  = S010state.t[sample_time_condition]

            sample_time_index     = sample_time_index[0:-1:step];
            S010state_times_no_plot  = S010state_times_no_plot[0:-1:step];

            time_plot = np.concatenate((time_plot, S010state_times_no_plot/ms))
            state_plot = np.concatenate((state_plot, S010state.apre[weights+(n1*N0_Neurons)][sample_time_index]))
            state2_plot = np.concatenate((state2_plot, S010state.apost[weights+(n1*N0_Neurons)][sample_time_index]))


            plt.plot(time_plot, state_plot, label='N1::'+str(n1))
            plt.plot(time_plot, state2_plot, label='N1::'+str(n1))
            plt.grid(True)
            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.0012,+0.0008))
            plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.savefig(figpath + '11_traces_stdp.png')
    plt.close(1)

if(print_traces and learning_2_phase):
    plt.figure(1)

    for n2 in range(N2_Neurons):

        ax1 = plt.subplot2grid((N2_Neurons,1), (n2,0))
        ax1.set_title("N" + str(n2) + " Traces")

        for weights in weights2_to_plot:
            time_plot = []
            state_plot = []
            state2_plot = []
            state3_plot = []

            sample_time_condition = (S120state.t/ms < end_plot*single_example_time) & (S120state.t/ms >= plot_start_time*single_example_time)
            sample_time_index     = np.where(sample_time_condition)[0]
            S120state_times_no_plot  = S120state.t[sample_time_condition]

            sample_time_index     = sample_time_index[0:-1:step];
            S120state_times_no_plot  = S120state_times_no_plot[0:-1:step];

            time_plot = np.concatenate((time_plot, S120state_times_no_plot/ms))
            state_plot = np.concatenate((state_plot, S120state.apre[weights+(n2*N1_Neurons)][sample_time_index]))
            state2_plot = np.concatenate((state2_plot, S120state.apost[weights+(n2*N1_Neurons)][sample_time_index]))
            state3_plot = np.concatenate((state3_plot, S120state.reward[weights+(n2*N1_Neurons)][sample_time_index]))


            plt.plot(time_plot, state_plot, label='N2::'+str(n2))
            plt.plot(time_plot, state2_plot, label='N2::'+str(n2))
            plt.plot(time_plot, state3_plot*0.0004, label='N2::'+str(n2), color='r')
            plt.grid(True)
            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.0012,+0.0008))
            plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.savefig(figpath + '11_traces_stdp_reward.png')
    plt.close(1)


plt.figure(1)

plt.title("Output Neuron Stream")

for i_count in np.arange(plot_start_time,end_plot):

    color = classes_color[train_y[i_count]]
    for k in range(N1_Neurons):
        sample_time_condition = (N1mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N1mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
        N1mon_times_nk_plot   = N1mon.spike_trains()[k][sample_time_condition]
        N1mon_nspikes_nk_plot = np.ones(size(N1mon_times_nk_plot))*k
        plt.plot(N1mon_times_nk_plot/ms, N1mon_nspikes_nk_plot, '*'+color)


plt.ylim((0,N1_Neurons))
plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))

plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');
plt.grid(True)
if not testing_phase:
    plt.savefig(figpath + '12_output_neurons.png')
else:
    plt.savefig(figpath + '12_test_output_neurons.png')

plt.close(1)

exit();

