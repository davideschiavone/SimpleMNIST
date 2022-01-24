from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import sys
import json

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

figpath = './figures/'

print_input_stream = True
print_output_membrana = True
print_weights = True
print_traces = True
print_stats = False
read_weight = False
store_weight = False
plt_example_mnist = False
use_only_1_and_2 = True

start_scope()

np.random.seed(2021)


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



X_size          = 28*28
X_Train_Samples = train_X.shape[0]


training_example   = X_Train_Samples if parameter['training_example']==-1 else parameter['training_example'];
monitor_step       = parameter['monitor_step']
plot_start_time    = parameter['plot_start_time']
plot_duration_time = parameter['plot_duration_time'] if parameter['plot_duration_time'] <= training_example else training_example
plot_step_time     = parameter['plot_step_time']

print('Start simulation with : ')
print("training_example= " + str(training_example))
print("monitor_step (ms)= " + str(monitor_step))
print("plot_start_time(ms)= " + str(plot_start_time))
print("plot_duration_time(ms)= " + str(plot_duration_time))
print("plot_step_time(ms)= " + str(plot_step_time))

end_plot = plot_start_time+plot_duration_time

print("train_X contans " + str(X_Train_Samples))

if(plt_example_mnist):
    plt.figure(1)
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.savefig(figpath + '1_first_nine.png')
    plt.close(1)


print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(10):
    print('Class ' + str(i) + ' has number of samples ' + str(len(train_y[train_y==i])))

if(plt_example_mnist):
    #print the first 9 values
    for k in classes:
        plt.figure(1)
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow( (train_X[train_y==k])[i], cmap=plt.get_cmap('gray'))
        plt.savefig(figpath + '2_first_nine_per_class_' + + str(k) + '.png')
        plt.close(1)

'''
we want to have 40FPS
thus, here we use 25ms
10ms for the frame
and 10ms for the resting
'''

single_example_time   = 25

taupre  = 7 * ms
taupost = 7 * ms

Threshold = 0.06
V_reset   = -0.1
A_reset   = +0.1


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

N1_Neurons = 4;

N1      = NeuronGroup(N1_Neurons, eqs, threshold='v>Threshold', reset=eqs_reset, refractory=10*ms, method='exact')

N1.tau  = 9*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
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
                    apre += reward*(0.0003)
                    w =  clip(w+apost,wmin,wmax)
                    ''',
                    on_post='''
                    apost += reward*(-0.0005)
                    w =  clip(w+apre,wmin,wmax)
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

S.connect(i=i_syn, j=j_syn)



if (read_weight):
    data = ""
    with open('weights.txt','r') as f:
        data = np.loadtxt(f)
        data = np.reshape(data,(N0_Neurons,N1_Neurons))
        weight_matrix = data
else:
    weight_matrix = np.random.normal(loc=1/N0_Neurons, scale=1/N0_Neurons*0.1, size=(N0_Neurons,N1_Neurons))

#row_sums = weight_matrix.sum(axis=0)
#weight_matrix = weight_matrix / row_sums


#plt.figure(1)
#count, bins, ignored = plt.hist(np.reshape(weight_matrix, (N0_Neurons*N1_Neurons)), N0_Neurons*N1_Neurons, density=True)
#plt.xlim(min(bins)*1.1, max(bins)*1.1)
#plt.ylim(0, max(count)*1.1)
#plt.grid(True)
#plt.title("Creation time after Norm")
#plt.savefig(figpath+'3_weight_creation.png')
#plt.close(1)


S.wmax     = 1
S.wmin     = 0
#S.delay   = 1*ms
minDelay   = 0*ms
maxDelay   = 3*ms
deltaDelay = maxDelay - minDelay
S.delay    = 'minDelay + rand() * deltaDelay'

S2 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post = clip(v_post+w,V_reset,1)
                    ''',
                    method='linear')

S2.connect('i != j')
S2.w = -0.15
S2.delay = 0*ms

net.add(N0)
net.add(N1)
net.add(S)
net.add(S2)

start_mon = plot_start_time*single_example_time


my_train_X_flat  = train_X_flat[0:training_example];

print("Network created.... Start training it with " + str(training_example) + " samples")

ts_time = 0
n0_s_list = []
n0_t_list = []
i_count = 0

max_w = -1
min_w = 1

stat_freq  = np.zeros((10, my_train_X_flat.shape[1]))
stat_power = np.zeros((10, my_train_X_flat.shape[1]))


net.run(0*ms)

#for n0 in range(N0_Neurons):
#    for n1 in range(N1_Neurons):
#        S.w['i==n0 and j==n1'] = weight_matrix[n0,n1] #as soon as it spikes, the output spikes too
#
weight_matrix_flat =  np.reshape(weight_matrix,(N0_Neurons*N1_Neurons))
S.w =weight_matrix_flat
S.reward = 1
avg_img = np.zeros(my_train_X_flat.shape[1])

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
    avg_img                            = avg_img + x_flat

    if(i_count % 100 == 0):
        print("Trained " + str(i_count) + " samples")



    if(print_input_stream):
        n0_s_list.append(n0_s)
        n0_t_list.append(n0_t)

    N0.set_spikes(n0_s, n0_t*ms)

    reward_s = +1
    #for n1 in range(N1_Neurons):
    #    if (n1 == 0):
    #        reward_s = +1 if train_y[i_count] == 1 else -1.3
    #    elif (n1 == 1):
    #        reward_s = +1 if train_y[i_count] == 2 else -1.3

    #    for n0 in range(N0_Neurons):
    #        S.reward['i==n0 and j==n1'] = reward_s
    

    if(ts_time == start_mon):
        print("Add monitors at time " + str(ts_time))
        if(print_input_stream):
            N0mon    = SpikeMonitor(N0)
            net.add(N0mon)

        N1mon    = SpikeMonitor(N1)
        net.add(N1mon)

        if(print_output_membrana):
            N1state  = StateMonitor(N1, ['v'], record=np.arange(N1_Neurons), dt=monitor_step*ms)
            net.add(N1state)

        if(print_traces or print_weights):
            Sstate   = StateMonitor(S, ['w','apre', 'apost'], record=np.arange(N0_Neurons*N1_Neurons), dt=monitor_step*ms)
            net.add(Sstate)

    net.run(single_example_time*ms)

    #for n0 in range(N0_Neurons):
    #    for n1 in range(N1_Neurons):
    #        weight_matrix[n0,n1] = S.w['i==n0 and j==n1']
    #weight_matrix.flatten = S.w
    ##ERROR IN CPP AS YOU CANNOT READ
    #weight_matrix_flat = S.w

    if(i_count % 20 == 0 and store_weight):
        plt.figure(1)
        count, bins, ignored = plt.hist(weight_matrix_flat, N0_Neurons*N1_Neurons, density=True)
        plt.xlim(min(bins)*1.1, max(bins)*1.1)
        plt.ylim(0, max(count)*1.1)
        plt.grid(True)
        plt.title("After Run "+str(i_count))
        plt.savefig(figpath+'4_weight_after_run'+str(i_count)+'.png')
        plt.close(1)


    #if(weight_matrix.max() > max_w):
    #    max_w = weight_matrix.max()

    #if(weight_matrix.min() < min_w):
    #    min_w = weight_matrix.min()


    #row_sums = weight_matrix.sum(axis=0)

    #for n1 in range(N1_Neurons):
    #    if row_sums[n1] != 0:
    #        weight_matrix[:,n1] = weight_matrix[:,n1] / row_sums[n1]

    ##for n0 in range(N0_Neurons):
    ##    for n1 in range(N1_Neurons):
    ##        S.w['i==n0 and j==n1'] = weight_matrix[n0,n1]
    #S.w = weight_matrix_flat

    ts_time = ts_time + single_example_time
    i_count = i_count + 1
    if(n0_debug):
        for k in range(X_size):
            print('Input Neuron ' + str(k) + ' spiked ' + str(len(N0mon.spike_trains()[k]/ms)))
            print(N0mon.spike_trains()[k]/ms)


print("Network trained....")

if mydevice == "cpp":
    device.build( directory='outputcpp', compile = True, run = True, debug=False, clean = True)
elif mydevice == "cuda":
    device.build( directory='output', compile = True, run = True, debug=False, clean = True)


avg_img     = avg_img/training_example
avg_img     = np.reshape(avg_img,(28, 28))
avg_img     = avg_img.astype(int)

sourceFile = open('avg_img.txt', 'w')
printmatrix(avg_img,sourceFile)
sourceFile.close()

plt.figure(1)
plt.imshow(avg_img, cmap=plt.get_cmap('gray'))
plt.savefig(figpath + '5_avg_training.png')
plt.close(1)


if(print_stats):
    plt.figure(1)
    plt.title("Pixel Frequency Class")
    for c in classes:
        color = 'k' if c == 1 else 'b'
        plt.stem(stat_freq[c],linefmt=color, markerfmt=color+'o', label='class ' + str(c))
    plt.legend()
    plt.savefig(figpath + '6_pixel_freq.png')
    plt.close(1)

    plt.figure(1)
    plt.title("Pixel Power Class")
    for c in classes:
        color = 'k' if c == 1 else 'b'
        plt.stem(stat_power[c],linefmt=color, markerfmt=color+'o', label='class ' + str(c))
    plt.legend()
    plt.savefig(figpath + '7_pixel_power.png')
    plt.close(1)



for c in classes:
    color = 'k' if c == 1 else 'b'
    print("pixel max frequency class " + str(c))
    print("stat_freq[class][" + str(stat_freq[c].argmax()) + "] = " + str(stat_freq[c].max()))
    print("power is " + str(stat_power[c][stat_freq[c].argmax()]))


if(print_input_stream):
    plt.figure(1)
    plt.title("Input Neuron Stream")

    for i_count in np.arange(plot_start_time,end_plot):

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
    plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    plt.savefig(figpath + '8_input_neurons.png')
    plt.close(1)




step=int(plot_step_time*10);



if(print_output_membrana):
    plt.figure(1)


    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("Output Neuron Membrana neuron " + str(n1))

        time_plot = []
        state_plot = []

        for i_count in np.arange(plot_start_time,end_plot):

            sample_time_condition = (N1state.t/ms < (i_count+1)*single_example_time) & (N1state.t/ms >= (i_count)*single_example_time)
            sample_time_index     = np.where(sample_time_condition)[0]
            N1state_times_no_plot = N1state.t[sample_time_condition]

            sample_time_index     = sample_time_index[0:-1:step];
            N1state_times_no_plot = N1state_times_no_plot[0:-1:step];

            time_plot = np.concatenate((time_plot, N1state_times_no_plot/ms))
            state_plot = np.concatenate((state_plot, N1state.v[n1][sample_time_index]))


        plt.plot(time_plot, state_plot, label='N1::'+str(n1))

        plt.xlabel('Time (ms)')
        plt.ylabel('V')
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)
    plt.savefig(figpath + '9_output_membrana.png')
    plt.close(1)


#MOST TWO FREQUENTS
stat_freq2 = stat_freq.copy()

stat_freq2[1][stat_freq[1].argmax()] = 0
stat_freq2[2][stat_freq[2].argmax()] = 0

weights_to_plot = [  stat_freq[1].argmax(),
        stat_freq2[1].argmax(),
        stat_freq[2].argmax(),
        stat_freq2[2].argmax(),
     ]
#[158, 159, 155, 156]
weights_to_plot = []

for row in np.arange(23,24):
    for col in np.arange(6,27):
        weights_to_plot.append(row*28+col)

weights_to_plot = np.arange(650,700+1)

print(weights_to_plot)

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

            for i_count in np.arange(plot_start_time,end_plot):

                sample_time_condition = (Sstate.t/ms < (i_count+1)*single_example_time) & (Sstate.t/ms >= (i_count)*single_example_time)
                sample_time_index     = np.where(sample_time_condition)[0]
                Sstate_times_no_plot  = Sstate.t[sample_time_condition]

                sample_time_index     = sample_time_index[0:-1:step];
                Sstate_times_no_plot  = Sstate_times_no_plot[0:-1:step];

                if Sstate.w[weights+(n1*N0_Neurons)][sample_time_index].min() < min_w:
                    min_w = Sstate.w[weights+(n1*N0_Neurons)][sample_time_index].min();

                if Sstate.w[weights+(n1*N0_Neurons)][sample_time_index].max() > max_w:
                    max_w = Sstate.w[weights+(n1*N0_Neurons)][sample_time_index].max();

                time_plot = np.concatenate((time_plot, Sstate_times_no_plot/ms))
                state_plot = np.concatenate((state_plot, Sstate.w[weights+(n1*N0_Neurons)][sample_time_index]))


            plt.plot(time_plot, state_plot,label='N1::'+str(n1))
            plt.grid(True)

        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((min_w*1.1,max_w*1.1))
        plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))

    plt.savefig(figpath + '10_weights_stdp.png')
    plt.close(1)



    for n1 in range(N1_Neurons):
        plt.figure(1)
        weight_img = np.reshape(S.w[:,n1], (28,28));
        max_w      = weight_img.max()
        weight_img = weight_img/max_w*255
        weight_img = weight_img.astype(int)
        plt.imshow(avg_img, cmap=plt.get_cmap('gray'))
        plt.savefig(figpath + '10_weights_img_class_' + str(n1) + '.png')
        plt.close(1)



with open('outfile.txt','w') as f:
    for line in weight_matrix:
        np.savetxt(f, line)


if(print_traces):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("N" + str(n1) + " Traces")


        for weights in weights_to_plot:
            time_plot = []
            state_plot = []
            state2_plot = []

            for i_count in np.arange(plot_start_time,end_plot):

                color = 'k' if train_y[i_count] == 1 else 'b'
                sample_time_condition = (Sstate.t/ms < (i_count+1)*single_example_time) & (Sstate.t/ms >= (i_count)*single_example_time)
                sample_time_index     = np.where(sample_time_condition)[0]
                Sstate_times_no_plot  = Sstate.t[sample_time_condition]

                sample_time_index     = sample_time_index[0:-1:step];
                Sstate_times_no_plot  = Sstate_times_no_plot[0:-1:step];

                time_plot = np.concatenate((time_plot, Sstate_times_no_plot/ms))
                state_plot = np.concatenate((state_plot, Sstate.apre[weights+(n1*N0_Neurons)][sample_time_index]))
                state2_plot = np.concatenate((state2_plot, Sstate.apost[weights+(n1*N0_Neurons)][sample_time_index]))


            plt.plot(time_plot, state_plot, label='N1::'+str(n1))
            plt.plot(time_plot, state2_plot, label='N1::'+str(n1))
            plt.grid(True)
            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.0012,+0.0008))
            plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))
    plt.savefig(figpath + '11_traces_stdp.png')
    plt.close(1)



plt.figure(1)

plt.title("Output Neuron Stream")

print('Output Neuron 0 spiked ' + str(len(N1mon.spike_trains()[0]/ms)))
print('Output Neuron 1 spiked ' + str(len(N1mon.spike_trains()[1]/ms)))

for i_count in np.arange(plot_start_time,end_plot):

    color = '*k' if train_y[i_count] == 1 else '*r'
    for k in range(N1_Neurons):
        sample_time_condition = (N1mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N1mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
        N1mon_times_nk_plot   = N1mon.spike_trains()[k][sample_time_condition]
        N1mon_nspikes_nk_plot = np.ones(size(N1mon_times_nk_plot))*k
        plt.plot(N1mon_times_nk_plot/ms, N1mon_nspikes_nk_plot, color)


plt.ylim((0,N1_Neurons))
plt.xlim((plot_start_time*single_example_time, end_plot*single_example_time))

plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');
plt.grid(True)
plt.savefig(figpath + '12_output_neurons.png')
plt.close(1)

exit();

