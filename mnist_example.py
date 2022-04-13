from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from decimal import Decimal
from datasetsequence import DataSetSequence
import os

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

testing_phase = sys.argv[1] == "testing";

if not os.path.exists('./figures'):
    os.makedirs('./figures')

if not os.path.exists('./Weights'):
    os.makedirs('./Weights')

if not testing_phase:
    f_param = open('parameter.json')
else:
    f_param = open('parameter_test.json')

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

plt_example_mnist = False
use_only_0_and_1 = False
use_only_0_and_1_and_2 = False

start_scope()

np.random.seed(2021)

batch_number       = int(sys.argv[2])
previous_state     = sys.argv[3] == "true"
print("Using previous state: " + str(previous_state))
dss_train = DataSetSequence(None, None, 256,'Batches/train', False)
dss_test  = DataSetSequence(None, None, 256,'Batches/test', False)



if not testing_phase:
    (train_X, train_y) = dss_train.getItem(batch_number)
    (test_X, test_y)   = dss_test.getItem(0)
else:
    (test_X, test_y)   = dss_test.getItem(batch_number)
    (train_X, train_y) = dss_train.getItem(0)

classes       = np.arange(0,10)
classes_sizes = np.arange(0,10)

for i in classes:
    classes_sizes[i] = len(train_y[train_y==i])

if(use_only_0_and_1):
    train_X = train_X[(train_y==0) | (train_y==1)]
    train_y = train_y[(train_y==0) | (train_y==1)]
    test_X  = test_X[(test_y==0)   | (test_y==1) ]
    test_y  = test_y[(test_y==0)   | (test_y==1) ]
    classes = np.arange(0,2)
elif(use_only_0_and_1_and_2):
    train_X = train_X[(train_y==0) | (train_y==1) | (train_y==2)]
    train_y = train_y[(train_y==0) | (train_y==1) | (train_y==2)]
    test_X  = test_X[(test_y==0)   | (test_y==1)  | (test_y==2) ]
    test_y  = test_y[(test_y==0)   | (test_y==1)  | (test_y==2) ]
    classes = np.arange(0,3)

train_X_flat = np.reshape(train_X,(train_X.shape[0], 28*28))
test_X_flat  = np.reshape(test_X,(test_X.shape[0], 28*28))

for i in classes:
    print('Class ' + str(i) + ' has number of samples ' + str( classes_sizes[i] ))

X_size          = 28*28
X_Train_Samples = train_X.shape[0]
X_Test_Samples  = test_X.shape[0]

N1_Neurons         = parameter['N1_Neurons']
N2_Neurons         = parameter['N2_Neurons']
Reward_Neurons     = int(2*N2_Neurons);

first_example      = parameter['first_example']

monitor_step       = parameter['monitor_step']
plot_step_time     = parameter['plot_step_time']

if not testing_phase:
    number_examples    = X_Train_Samples if parameter['number_example']==-1 else parameter['number_example'];
else:
    number_examples    = X_Test_Samples if parameter['number_example']==-1 else parameter['number_example'];

learning_1_phase   = parameter['learning_1_phase']
learning_2_phase   = parameter['learning_2_phase']

print_neuron_l0     = parameter['print_neuron_l0'];

print_l1_membrana   = parameter['print_l1_membrana'];
print_l1_weights    = parameter['print_l1_weights'];
print_l1_traces     = parameter['print_l1_traces'];
print_l1_state      = parameter['print_l1_state'];
print_neuron_l1     = parameter['print_neuron_l1'];

print_l2_membrana   = parameter['print_l2_membrana'];
print_l2_weights    = parameter['print_l2_weights'];
print_l2_traces     = parameter['print_l2_traces'];
print_l2_state      = parameter['print_l2_state'];
print_statistics    = parameter['print_statistics'];

print_neuron_l2     = parameter['print_neuron_l2'];

print_neuron_reward = parameter['print_neuron_reward'];

use_l2              = parameter['use_l2']

plot_start_time    = parameter['plot_start_time'] if parameter['plot_start_time']>=0 else (number_examples+parameter['plot_start_time'])

if parameter['plot_duration_time'] < 0:
    plot_duration_time = number_examples
elif parameter['plot_duration_time'] > number_examples:
    plot_duration_time = number_examples
else:
    plot_duration_time = parameter['plot_duration_time']

end_plot           = plot_start_time+plot_duration_time

'''
we want to have 40FPS
thus, here we use 25ms
10ms for the frame
and 10ms for the resting
'''

single_example_time   = 25


print('Start simulation with : ')
print("number_examples = " + str(number_examples ))
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
    #print the first 3 values
    for k in classes:
        plt.figure(1)
        for i in range(3):
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

filename = './previous_state.npy'
if testing_phase:
    filename = './previous_state_test.npy'

if previous_state:
    with open(filename, 'rb') as f:
        N1v = np.load(f)
        N1a = np.load(f)
        if use_l2:
            N2v = np.load(f)
            N2a = np.load(f)
        if learning_1_phase:
            S010w     = np.load(f)
            S010apre  = np.load(f)
            S010apost = np.load(f)
        if learning_2_phase and use_l2:
            S120w     = np.load(f)
            S120apre  = np.load(f)
            S120apost = np.load(f)


net        = Network()

N0_Neurons = X_size; #28x28
N0         = SpikeGeneratorGroup(N0_Neurons, [0], [0]*ms)

eqs = '''
dv/dt = -v/taul1 + a/tausl1: 1 (unless refractory)
da/dt = -a/tausl1: 1
ddynThreshold/dt = (Thresholdl1-dynThreshold)/tautl1 : 1
taul1 : second
tausl1 : second
tautl1 : second
sumwl1 : 1
'''
eqs_reset = '''
                v = V_resetl1
                a = A_resetl1
                dynThreshold = dynThreshold+Thresholdl1*0.5
            '''

tauprel1  = 7 * ms
taupostl1 = 7 * ms

Thresholdl1 = 0.052
V_resetl1   = -0.1
A_resetl1   = +0.1

N1         = NeuronGroup(N1_Neurons, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=10*ms, method='exact')

N1.taul1  = 9*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                 #you can also increase the spacex0x1 and keep tau to 10ms for example
N1.tausl1 = 20*ms
N1.tautl1 = 1000*ms
N1.dynThreshold = Thresholdl1

if previous_state:
    N1.v = N1v
    N1.a = N1a
else:
    N1.v = 0
    N1.a = 0

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


i_s010_syn = []
j_s010_syn = []

for n1 in range(N1_Neurons):
    #all N0 neurons connected to n1
    i_n = np.arange(0,N0_Neurons)
    j_n = np.ones(size(i_n))*n1
    i_s010_syn = np.concatenate((i_s010_syn, i_n))
    j_s010_syn = np.concatenate((j_s010_syn, j_n))

i_s010_syn = i_s010_syn.astype(int)
j_s010_syn = j_s010_syn.astype(int)

S010.connect(i=i_s010_syn, j=j_s010_syn)

if learning_1_phase:
    S010.wmax     = Thresholdl1*0.5
    S010.wmin     = 0

minDelay   = 0*ms
maxDelay   = 4*ms
deltaDelay = maxDelay - minDelay
S010.delay = 'minDelay + rand() * deltaDelay'


if learning_1_phase and previous_state == False:
    weight_matrix = np.zeros((N1_Neurons,N0_Neurons))
    for n1 in np.arange(N1_Neurons):
        weight_matrix[n1,:] = np.random.normal(loc=1/N0_Neurons, scale=1/N0_Neurons*0.1, size=(N0_Neurons))
elif learning_1_phase and previous_state:
        weight_matrix = S010w
else:
    weight_matrix = np.zeros((N1_Neurons,N0_Neurons))
    with open('./Weights/l1_weights_trained.npy', 'rb') as f:
        weight_matrix = np.load(f)

if learning_2_phase and previous_state == False and use_l2:
    weight_matrix2 = np.zeros((N2_Neurons,N1_Neurons))
    for n2 in np.arange(N2_Neurons):
        weight_matrix2[n2,:] = np.random.normal(loc=1/N1_Neurons, scale=1/N1_Neurons*0.1, size=(N1_Neurons))
elif learning_2_phase and previous_state and use_l2:
        weight_matrix2 = S120w
elif use_l2:
    weight_matrix2 = np.zeros((N2_Neurons,N1_Neurons))
    with open('./Weights/l2_weights_trained.npy', 'rb') as f:
        weight_matrix2 = np.load(f)



S110 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post = clip(v_post+w,V_resetl1,1)
                    ''',
                    method='linear')

S110.connect('i != j')
S110.w = -0.025
S110.delay = 0*ms

net.add(N0)
net.add(N1)
net.add(S010)
net.add(S110)

Thresholdl2 = 0.015

N2        = NeuronGroup(N2_Neurons, eqs, threshold='v>Thresholdl2', reset=eqs_reset, refractory=10*ms, method='exact')


N2.taul1  = 9*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                 #you can also increase the spacex0x1 and keep tau to 10ms for example
N2.tausl1 = 20*ms

tau_reward = single_example_time*ms

if previous_state and use_l2:
    N2.v = N2v
    N2.a = N2a
else:
    N1.v = 0
    N1.a = 0

if use_l2:
    if learning_2_phase:


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

        i_s120_syn = []
        j_s120_syn = []

        for n2 in range(N2_Neurons):
            #all N1 neurons connected to n2
            i_n = np.arange(0,N1_Neurons)
            j_n = np.ones(size(i_n))*n2
            i_s120_syn = np.concatenate((i_s120_syn, i_n))
            j_s120_syn = np.concatenate((j_s120_syn, j_n))

        i_s120_syn = i_s120_syn.astype(int)
        j_s120_syn = j_s120_syn.astype(int)

        S120.connect(i=i_s120_syn, j=j_s120_syn)

        S120.wmax     = 0.02
        S120.wmin     = 0

        NReward = SpikeGeneratorGroup(Reward_Neurons, [0], [0]*ms)

        ##Reward has 2*N2_Neurons, S120 has N1_Neurons*N2_Neurons connections
        SRS120 = Synapses(NReward, S120,
                            '''
                             ''',
                            on_pre='''
                            reward_post = int(i<N2_Neurons)*1.0
                            punish_post = int(i>=N2_Neurons)*1.0
                            ''',
                            method='linear')


        i_srs120_syn = []
        j_srs120_syn = []

        for nr in range(N2_Neurons):
            #nr 0  connected to synapsys 0...N1_Neurons-1 which are the ones connecting N1 to neuron 0 of N2 (reward)
            #nr 10 connected to synapsys 0...N1_Neurons-1 which are the ones connecting N1 to neuron 10 of N2 (punish)
            #nr 1 connected to synapsys N1...2*N1_Neurons-1 which are the ones connecting N1 to neuron 1 of N2
            # like a flat matrix, etc
            j_n = np.arange(0,N1_Neurons) + (nr*N1_Neurons)
            i_n = np.ones(size(j_n))*nr
            i_srs120_syn = np.concatenate((i_srs120_syn, i_n))
            j_srs120_syn = np.concatenate((j_srs120_syn, j_n))
            i_n = np.ones(size(j_n))*(nr+N2_Neurons)
            i_srs120_syn = np.concatenate((i_srs120_syn, i_n))
            j_srs120_syn = np.concatenate((j_srs120_syn, j_n))


        i_srs120_syn = i_srs120_syn.astype(int)
        j_srs120_syn = j_srs120_syn.astype(int)


        SRS120.connect(i=i_srs120_syn, j=j_srs120_syn)

        net.add(N2)
        net.add(NReward)
        net.add(S120)
        net.add(SRS120)
    else:

        S120 = Synapses(N1, N2,
                            '''
                            w : 1
                            ''',
                            on_pre='''
                            v_post += w
                            ''',
                            method='linear')

        i_s120_syn = []
        j_s120_syn = []

        for n2 in range(N2_Neurons):
            #all N1 neurons connected to n2
            i_n = np.arange(0,N1_Neurons)
            j_n = np.ones(size(i_n))*n2
            i_s120_syn = np.concatenate((i_s120_syn, i_n))
            j_s120_syn = np.concatenate((j_s120_syn, j_n))

        i_s120_syn = i_s120_syn.astype(int)
        j_s120_syn = j_s120_syn.astype(int)

        S120.connect(i=i_s120_syn, j=j_s120_syn)

        net.add(N2)
        net.add(S120)

    #S120.delay = 'minDelay + rand() * deltaDelay'
    S220 = Synapses(N2, N2,
                        '''
                        w : 1
                        ''',
                        on_pre='''
                        v_post = clip(v_post+w,V_resetl1,1)
                        ''',
                        method='linear')

    S220.connect('i != j')
    S220.w = -0.15
    S220.delay = 0*ms

    net.add(S220)

start_mon = plot_start_time*single_example_time

if not testing_phase:
    my_set_X_flat  = train_X_flat[first_example:number_examples];
    print("Network created.... Start training it with " + str(number_examples) + " samples")
else:
    my_set_X_flat  = test_X_flat[first_example:number_examples];
    print("Network created.... Start testing it with " + str(number_examples) + " samples")

sim_dict = {
            'plot_start' : plot_start_time*single_example_time,
            'plot_end' : end_plot*single_example_time,
            'batch_number' : batch_number
        }

parameters = {
            'single_example_time' : single_example_time,
            'N0_Neurons' : N0_Neurons,
            'N1_Neurons' : N1_Neurons,
            'N2_Neurons' : N2_Neurons,
            'Reward_Neurons' : Reward_Neurons,
            'print_neuron_l0' : print_neuron_l0,
            'print_l1_membrana' : print_l1_membrana,
            'print_l1_weights' : print_l1_weights,
            'print_l1_traces' : print_l1_traces,
            'print_l1_state' : print_l1_state,
            'print_l2_membrana' : print_l2_membrana,
            'print_l2_weights' : print_l2_weights,
            'print_l2_traces' : print_l2_traces,
            'print_l2_state' : print_l2_state,
            'print_statistics' : print_statistics,
            'print_neuron_reward' : print_neuron_reward,
            'print_neuron_l1' : print_neuron_l1,
            'print_neuron_l2' : print_neuron_l2,
            'use_l2' : use_l2,
            'learning_1_phase' : learning_1_phase,
            'learning_2_phase' : learning_2_phase,
            }

filename = 'sim_values.json'
if testing_phase:
    filename = 'sim_values_test.json'

if previous_state:
    old_sim_json_file = open(filename)
    sim_json          = json.load(old_sim_json_file)
    sim_json['simulations'].append(sim_dict)
else:
    sim_json = {
        'parameters' : parameters,
        'simulations': [
            sim_dict
        ]
    }

with open(filename, 'w') as f:
    json_string = json.dumps(sim_json, indent=2)
    f.write(json_string)


net.run(0*ms)

#for n0 in range(N0_Neurons):
#    for n1 in range(N1_Neurons):
#        S010.w['i==n0 and j==n1'] = weight_matrix[n0,n1] #as soon as it spikes, the output spikes too
#

weight_matrix_flat  = np.reshape(weight_matrix,(N0_Neurons*N1_Neurons))
S010.w              = weight_matrix_flat
if use_l2:
    weight_matrix2_flat = np.reshape(weight_matrix2,(N1_Neurons*N2_Neurons))
    S120.w              = weight_matrix2_flat

if previous_state:
    if learning_1_phase:
        S010.apre  = S010apre
        S010.apost = S010apost
    if learning_2_phase and use_l2:
        S120.apre  = S120apre
        S120.apost = S120apost


n0_s_list = []
n0_t_list = []
nr_s_list = []
nr_t_list = []

ts_time = 0
i_count = 0


for x_flat in my_set_X_flat:

    '''
    each pixel from 0 to 255
    is encoded in 255 to 1 (0 are ignored)
    strong values comes earlier
    255 becomes 10ms
    1   becomes 40us
    '''

    n0_s    = np.where(x_flat > 190)[0]
    n0_t    = ts_time + (256 - x_flat[n0_s])*10/255

    n0_s_list = np.concatenate((n0_s_list, n0_s))
    n0_t_list = np.concatenate((n0_t_list, n0_t))

    if use_l2 and learning_2_phase:
        #reward
        nr_s    = []
        nr_s.append(train_y[i_count])

        nr_t    = []
        nr_t.append(ts_time) ###reward

        for nr in range(int(Reward_Neurons/2)):
            if nr != nr_s[0]:
                nr_s.append(nr + int(Reward_Neurons/2))
                nr_t.append(ts_time) ###punishment


        nr_s_list = np.concatenate((nr_s_list, nr_s))
        nr_t_list = np.concatenate((nr_t_list, nr_t))

    if(i_count % 1000 == 0):
        print("Trained " + str(i_count) + " samples")

    ts_time = ts_time + single_example_time
    i_count = i_count + 1


N0.set_spikes(n0_s_list, n0_t_list*ms)

if(print_neuron_l0):
    N0mon    = SpikeMonitor(N0)
    net.add(N0mon)

if learning_2_phase and use_l2:
    NReward.set_spikes(nr_s_list, nr_t_list*ms)
    if(print_neuron_reward):
        NRewardmon    = SpikeMonitor(NReward)
        net.add(NRewardmon)

if print_neuron_l1:
    N1mon    = SpikeMonitor(N1)
    net.add(N1mon)

if use_l2 and print_neuron_l2:
    N2mon    = SpikeMonitor(N2)
    net.add(N2mon)


if(print_l1_membrana and print_l1_state):
    N1state  = StateMonitor(N1, ['v'], record=np.arange(N1_Neurons), dt=monitor_step*ms)
    net.add(N1state)

if(print_l2_membrana and print_l2_state and use_l2):
    N2state  = StateMonitor(N2, ['v'], record=np.arange(N2_Neurons), dt=monitor_step*ms)
    net.add(N2state)

if(print_l1_traces or print_l1_weights):
    if learning_1_phase and print_l1_state:
        S010state   = StateMonitor(S010, ['w','apre', 'apost'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    else:
        S010state   = StateMonitor(S010, ['w'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    net.add(S010state)

if(print_l2_traces or print_l2_weights and use_l2):
    if learning_2_phase and print_l2_state:
        S120state   = StateMonitor(S120, ['w','apre', 'apost', 'reward', 'punish'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
    else:
        S120state   = StateMonitor(S120, ['w'], record=np.arange(N1_Neurons*N2_Neurons), dt=monitor_step*ms)
    net.add(S120state)

net.run((number_examples*single_example_time + 1)*ms)

if not testing_phase:
    print("Network trained....")
else:
    print("Network tested....")

if mydevice == "cpp":
    device.build( directory='outputcpp', compile = True, run = True, debug=False, clean = True)
elif mydevice == "cuda":
    device.build( directory='output', compile = True, run = True, debug=False, clean = True)

file_mode = 'ab' if previous_state else 'wb'

if(print_neuron_l0):
    n0_indices, n0_times = N0mon.it
    filename = './Weights/l0_stream.npy'
    if testing_phase:
        filename = './Weights/l0_stream_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, n0_times)
        np.save(f, n0_indices)

if(print_neuron_reward and use_l2):
    nreward_indices, nreward_times = NRewardmon.it
    filename = './Weights/lreward_stream.npy'
    if testing_phase:
        filename = './Weights/lreward_stream_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, nreward_times)
        np.save(f, nreward_indices)

if(print_neuron_l1):
    n1_indices, n1_times = N1mon.it
    filename = './Weights/l1_stream.npy'
    if testing_phase:
        filename = './Weights/l1_stream_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, n1_times)
        np.save(f, n1_indices)

if(print_neuron_l2 and use_l2):
    n2_indices, n2_times = N2mon.it
    filename = './Weights/l2_stream.npy'
    if testing_phase:
        filename = './Weights/l2_stream_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, n2_times)
        np.save(f, n2_indices)

if(print_statistics):
    filename = './y_values.npy'
    if testing_phase:
        filename = './y_values_test.npy'
    with open(filename, file_mode) as f:
        if not testing_phase:
            my_set_y_flat  = train_y[first_example:number_examples];
        else:
            my_set_y_flat  = test_y[first_example:number_examples];
            print(test_y[first_example:number_examples])
        np.save(f, my_set_y_flat)


step=int(plot_step_time*10);

if(print_l1_membrana and print_l1_state):
    sample_time_condition = (N1state.t/ms < end_plot*single_example_time) & (N1state.t/ms >= plot_start_time*single_example_time)
    sample_time_index     = np.where(sample_time_condition)[0]
    N1state_times_no_plot = N1state.t[sample_time_condition]
    sample_time_index     = sample_time_index[0:-1:step];
    N1state_times_no_plot = N1state_times_no_plot[0:-1:step];
    time_plot             = N1state_times_no_plot/ms
    filename = './Weights/l1_membrana_time.npy'
    if testing_phase:
        filename = './Weights/l1_membrana_time_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, time_plot)
    filename = './Weights/l1_membrana_value.npy'
    if testing_phase:
        filename = './Weights/l1_membrana_value_test.npy'
    with open(filename, file_mode) as f:
        for n1 in range(N1_Neurons):
            state_plot = N1state.v[n1][sample_time_index]
            np.save(f, state_plot)

if(print_l2_membrana and print_l2_state and use_l2):
    sample_time_condition = (N2state.t/ms < end_plot*single_example_time) & (N2state.t/ms >= plot_start_time*single_example_time)
    sample_time_index     = np.where(sample_time_condition)[0]
    N2state_times_no_plot = N2state.t[sample_time_condition]
    sample_time_index     = sample_time_index[0:-1:step];
    N2state_times_no_plot = N2state_times_no_plot[0:-1:step];
    time_plot             = N2state_times_no_plot/ms
    filename = './Weights/l2_membrana_time.npy'
    if testing_phase:
        filename = './Weights/l2_membrana_time_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, time_plot)
    filename = './Weights/l2_membrana_value.npy'
    if testing_phase:
        filename = './Weights/l2_membrana_value_test.npy'
    with open(filename, file_mode) as f:
        for n2 in range(N2_Neurons):
            state_plot = N2state.v[n2][sample_time_index]
            np.save(f, state_plot)

if(print_l1_weights):
    if learning_1_phase:
        filename = './Weights/l1_weights.npy'
        weight_matrix = S010.w.get_item(item=np.arange(N0_Neurons*N1_Neurons))
        weight_matrix = np.reshape(weight_matrix,(N1_Neurons,N0_Neurons))
        #here we append the weights for debug
        with open(filename, file_mode) as f:
            np.save(f, weight_matrix)
        #here we store weights so that we have them ready
        with open('./Weights/l1_weights_trained.npy', 'wb') as f:
            np.save(f, weight_matrix)
        if(print_l1_state):
            sample_time_condition    = (S010state.t/ms < end_plot*single_example_time) & (S010state.t/ms >= plot_start_time*single_example_time)
            sample_time_index        = np.where(sample_time_condition)[0]
            S010state_times_no_plot  = S010state.t[sample_time_condition]
            sample_time_index        = sample_time_index[0:-1:step];
            S010state_times_no_plot  = S010state_times_no_plot[0:-1:step];
            time_plot                = S010state_times_no_plot/ms
            filename = './Weights/l1_weights_time.npy'
            with open(filename, file_mode) as f:
                np.save(f, time_plot)
            filename = './Weights/l1_weights_value.npy'
            with open(filename, file_mode) as f:
                for n1 in range(N1_Neurons):
                    for weights in range(N0_Neurons):
                        state_plot = S010state.w[weights+(n1*N0_Neurons)][sample_time_index];
                        np.save(f, state_plot)

if(print_l2_weights and use_l2):
    if learning_2_phase:
        filename = './Weights/l2_weights.npy'
        with open(filename, file_mode) as f:
            weight_matrix = S120.w.get_item(item=np.arange(N1_Neurons*N2_Neurons))
            weight_matrix = np.reshape(weight_matrix,(N2_Neurons,N1_Neurons))
            np.save(f, weight_matrix)
        if(print_l2_state):
            sample_time_condition    = (S120state.t/ms < end_plot*single_example_time) & (S120state.t/ms >= plot_start_time*single_example_time)
            sample_time_index        = np.where(sample_time_condition)[0]
            S120state_times_no_plot  = S120state.t[sample_time_condition]
            sample_time_index        = sample_time_index[0:-1:step];
            S120state_times_no_plot  = S120state_times_no_plot[0:-1:step];
            time_plot                = S120state_times_no_plot/ms
            filename = './Weights/l2_weights_time.npy'
            with open(filename, file_mode) as f:
                np.save(f, time_plot)
            filename = './Weights/l2_weights_value.npy'
            with open(filename, file_mode) as f:
                for n2 in range(N2_Neurons):
                    for weights in range(N1_Neurons):
                        state_plot = S120state.w[weights+(n2*N1_Neurons)][sample_time_index];
                        np.save(f, state_plot)

if(print_l1_traces and learning_1_phase and print_l1_state):
    sample_time_condition   = (S010state.t/ms < end_plot*single_example_time) & (S010state.t/ms >= plot_start_time*single_example_time)
    sample_time_index       = np.where(sample_time_condition)[0]
    S010state_times_no_plot = S010state.t[sample_time_condition]
    sample_time_index       = sample_time_index[0:-1:step];
    S010state_times_no_plot = S010state_times_no_plot[0:-1:step];
    time_plot               = S010state_times_no_plot/ms
    filename = './Weights/l1_trace_time.npy'
    if testing_phase:
        filename = './Weights/l1_trace_time_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, time_plot)
    filename = './Weights/l1_trace_value.npy'
    if testing_phase:
        filename = './Weights/l1_trace_value_test.npy'
    with open(filename, file_mode) as f:
        for n1 in range(N1_Neurons):
            for weights in range(N0_Neurons):
                state_plot  = S010state.apre[weights+(n1*N0_Neurons)][sample_time_index];
                state2_plot = S010state.apost[weights+(n1*N0_Neurons)][sample_time_index];
                np.save(f, state_plot)
                np.save(f, state2_plot)

if(print_l2_traces and learning_2_phase and print_l2_state and use_l2):
    sample_time_condition   = (S120state.t/ms < end_plot*single_example_time) & (S120state.t/ms >= plot_start_time*single_example_time)
    sample_time_index       = np.where(sample_time_condition)[0]
    S120state_times_no_plot = S120state.t[sample_time_condition]
    sample_time_index       = sample_time_index[0:-1:step];
    S120state_times_no_plot = S120state_times_no_plot[0:-1:step];
    time_plot               = S120state_times_no_plot/ms
    filename = './Weights/l2_trace_time.npy'
    if testing_phase:
        filename = './Weights/l2_trace_time_test.npy'
    with open(filename, file_mode) as f:
        np.save(f, time_plot)
    filename = './Weights/l2_trace_value.npy'
    if testing_phase:
        filename = './Weights/l2_trace_value_test.npy'
    with open(filename, file_mode) as f:
        for n2 in range(N2_Neurons):
            for weights in range(N1_Neurons):
                state_plot  = S120state.apre[weights+(n2*N1_Neurons)][sample_time_index];
                state2_plot = S120state.apost[weights+(n2*N1_Neurons)][sample_time_index];
                state3_plot = S120state.reward[weights+(n2*N1_Neurons)][sample_time_index];
                state4_plot = S120state.punish[weights+(n2*N1_Neurons)][sample_time_index];
                np.save(f, state_plot)
                np.save(f, state2_plot)
                np.save(f, state3_plot)
                np.save(f, state4_plot)

filename = './previous_state.npy'
if testing_phase:
    filename = './previous_state_test.npy'

with open(filename, 'wb') as f:
    np.save(f, np.array(N1.v))
    np.save(f, np.array(N1.a))
    if use_l2:
        np.save(f, np.array(N2.v))
        np.save(f, np.array(N2.a))
    if learning_1_phase:
        np.save(f, np.array(S010.w))
        np.save(f, np.array(S010.apre))
        np.save(f, np.array(S010.apost))
    if learning_2_phase and use_l2:
        np.save(f, np.array(S120.w))
        np.save(f, np.array(S120.apre))
        np.save(f, np.array(S120.apost))



exit();

