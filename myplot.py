import matplotlib.pyplot as plt
import numpy as np
import json
figpath = './figures/'

json_file = open('sim_values.json')
sim_json  = json.load(json_file)

xlim_start     = sim_json['simulations'][0]['plot_start']
total_duration = 0
plot_start_lst = [ sim_json['simulations'][0]['plot_start'] ]

for sim in sim_json['simulations']:
    total_duration+= sim['plot_end'] - sim['plot_start']
    plot_start_lst.append(total_duration)

print(plot_start_lst)
xlim_end   = xlim_start + total_duration

N0_Neurons          = sim_json['parameters']['N0_Neurons']
N1_Neurons          = sim_json['parameters']['N1_Neurons']
N2_Neurons          = sim_json['parameters']['N2_Neurons']
Reward_Neurons      = sim_json['parameters']['Reward_Neurons']
testing_phase       = sim_json['parameters']['testing_phase']
print_neuron_l0     = sim_json['parameters']['print_neuron_l0']
print_l1_membrana   = sim_json['parameters']['print_l1_membrana']
print_l1_weights    = sim_json['parameters']['print_l1_weights']
print_l1_traces     = sim_json['parameters']['print_l1_traces']
print_l2_state      = sim_json['parameters']['print_l1_state']
print_l2_membrana   = sim_json['parameters']['print_l2_membrana']
print_l2_weights    = sim_json['parameters']['print_l2_weights']
print_l2_traces     = sim_json['parameters']['print_l2_traces']
print_l2_state      = sim_json['parameters']['print_l2_state']
print_neuron_reward = sim_json['parameters']['print_neuron_reward']
print_neuron_l1     = sim_json['parameters']['print_neuron_l1']
print_neuron_l2     = sim_json['parameters']['print_neuron_l2']
learning_1_phase    = sim_json['parameters']['learning_1_phase']
learning_2_phase    = sim_json['parameters']['learning_2_phase']

if(print_neuron_l0):
    plt.figure(1)
    plt.title("Input Neuron Stream")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    i_count = 0
    with open('./Weights/l0_stream.npy', 'rb') as f:
        for sim in sim_json['simulations']:
            n0_times   = np.load(f)
            n0_indices = np.load(f)
            n0_times   = n0_times*1000 + plot_start_lst[i_count]
            plt.plot(n0_times, n0_indices, '.k')
            i_count+=1
    plt.xlim((xlim_start, xlim_end))
    plt.ylim((-0.5,N0_Neurons))
    if not testing_phase:
        plt.savefig(figpath + 'l0_stream.png')
    else:
        plt.savefig(figpath + 'l0_stream_test.png')
    plt.close(1)

if(print_neuron_reward):
    plt.figure(1)
    plt.title("Reward Neuron Stream")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    i_count = 0
    with open('./Weights/lreward_stream.npy', 'rb') as f:
        for sim in sim_json['simulations']:
            nr_times   = np.load(f)
            nr_indices = np.load(f)
            nr_times   = nr_times*1000 + plot_start_lst[i_count]
            plt.plot(nr_times, nr_indices, '*r')
            i_count+=1

    plt.xlim((xlim_start, xlim_end))
    plt.ylim((-0.5,Reward_Neurons))
    if not testing_phase:
        plt.savefig(figpath + 'lreward_stream.png')
    plt.close(1)

if(print_neuron_l1):
    plt.figure(1)
    plt.title("L1 Neuron Stream")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    i_count = 0
    with open('./Weights/l1_stream.npy', 'rb') as f:
        for sim in sim_json['simulations']:
            n1_times   = np.load(f)
            n1_indices = np.load(f)
            n1_times   = n1_times*1000 + plot_start_lst[i_count]
            plt.plot(n1_times, n1_indices, '.k')
            i_count+=1

    plt.ylim((-0.5,N1_Neurons))
    plt.xlim((xlim_start, xlim_end))
    if not testing_phase:
        plt.savefig(figpath + 'l1_stream.png')
    else:
        plt.savefig(figpath + 'l1_stream_test.png')
    plt.close(1)


if(print_neuron_l2):
    plt.figure(1)
    plt.title("L2 Neuron Stream")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    i_count = 0
    with open('./Weights/l2_stream.npy', 'rb') as f:
        for sim in sim_json['simulations']:
            n2_times   = np.load(f)
            n2_indices = np.load(f)
            n2_times   = n2_times*1000 + plot_start_lst[i_count]
            plt.plot(n2_times, n2_indices, '.k')
            i_count+=1
    plt.ylim((-0.5,N2_Neurons))
    plt.xlim((xlim_start, xlim_end))
    if not testing_phase:
        plt.savefig(figpath + 'l2_stream.png')
    else:
        plt.savefig(figpath + 'l2_stream_test.png')
    plt.close(1)

if(print_l1_membrana and print_l1_state):
    analog_plots = {
                'times': [],
                'neurons': {}
                }
    with open('./Weights/l1_membrana_time.npy', 'rb') as f:
        i_count = 0
        time_plot_lst = []
        for sim in sim_json['simulations']:
            time_plot = np.load(f)
            analog_plots['times'].append(time_plot + plot_start_lst[i_count])
            i_count+=1

    with open('./Weights/l1_membrana_value.npy', 'rb') as f:
        neurons_plots = {str(n1):[] for n1 in range(N1_Neurons)}
        for sim in sim_json['simulations']:
            for n1 in range(N1_Neurons):
                state_plot = np.load(f)
                neurons_plots[str(n1)].append(state_plot)
        analog_plots['neurons'] = neurons_plots

    fig_counter = 1
    stop_counter = 0
    for n1 in range(N1_Neurons):
        if (n1 % 4 == 0):
            plt.figure(1)
            stop_counter = 1
            fig_counter = fig_counter + 1

        ax1 = plt.subplot2grid((4,1), (n1 % 4,0))
        ax1.set_title("neuron " + str(n1))
        plt.xlim((xlim_start, xlim_end))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)

        i_count = 0
        for time_plot in analog_plots['times']:
            state_plot = analog_plots['neurons'][str(n1)][i_count]
            plt.plot(time_plot, state_plot)
            i_count+=1

        if (n1 % 4 == 3):
            stop_counter = 0
            if not testing_phase:
                plt.savefig(figpath + 'l1_membrana_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + 'l1_membrana_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath + 'l1_membrana_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + 'l1_membrana_value_test' + str(fig_counter-1) + '.png')
        plt.close(1)

if(print_l2_membrana and print_l2_state):
    analog_plots = {
                'times': [],
                'neurons': {}
                }
    with open('./Weights/l2_membrana_time.npy', 'rb') as f:
        i_count = 0
        time_plot_lst = []
        for sim in sim_json['simulations']:
            time_plot = np.load(f)
            analog_plots['times'].append(time_plot + plot_start_lst[i_count])
            i_count+=1

    with open('./Weights/l2_membrana_value.npy', 'rb') as f:
        neurons_plots = {str(n2):[] for n2 in range(N2_Neurons)}
        for sim in sim_json['simulations']:
            for n2 in range(N2_Neurons):
                state_plot = np.load(f)
                neurons_plots[str(n2)].append(state_plot)
        analog_plots['neurons'] = neurons_plots

    fig_counter = 1
    stop_counter = 0
    for n2 in range(N2_Neurons):
        if (n2 % 4 == 0):
            plt.figure(1)
            stop_counter = 1
            fig_counter = fig_counter + 1

        ax1 = plt.subplot2grid((4,1), (n2 % 4,0))
        ax1.set_title("neuron " + str(n2))
        plt.xlim((xlim_start, xlim_end))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)

        i_count = 0
        for time_plot in analog_plots['times']:
            state_plot = analog_plots['neurons'][str(n2)][i_count]
            plt.plot(time_plot, state_plot)
            i_count+=1

        if (n2 % 4 == 3):
            stop_counter = 0
            if not testing_phase:
                plt.savefig(figpath + 'l2_membrana_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + 'l2_membrana_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath + 'l2_membrana_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + 'l2_membrana_value_test' + str(fig_counter-1) + '.png')
        plt.close(1)

if(print_l2_weights and print_l2_state):
    analog_plots = {
                'times': [],
                'weights': {}
                }
    with open('./Weights/l2_weights_time.npy', 'rb') as f:
        i_count = 0
        time_plot_lst = []
        for sim in sim_json['simulations']:
            time_plot = np.load(f)
            analog_plots['times'].append(time_plot + plot_start_lst[i_count])
            i_count+=1

    with open('./Weights/l2_weights_value.npy', 'rb') as f:
        weights_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        for sim in sim_json['simulations']:
            for n2 in range(N2_Neurons):
                for n1 in range(N1_Neurons):
                    state_plot = np.load(f)
                    weights_plots[str(n2)+":"+str(n1)].append(state_plot)

        analog_plots['weights'] = weights_plots

    fig_counter = 1
    stop_counter = 0

    for n2 in range(N2_Neurons):
        for n1 in range(N1_Neurons):

            if (n1 % 8 == 0):
                min_w = +10
                max_w = -10
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((8,1), (n1 % 8,0))
            ax1.set_title(str(n1)+"::"+str(n2))

            i_count = 0
            for time_plot in analog_plots['times']:
                state_plot = analog_plots['weights'][str(n2)+":"+str(n1)][i_count]
                plt.plot(time_plot, state_plot)
                i_count+=1

            if(state_plot.min() < min_w):
                min_w = state_plot.min()
            if(state_plot.max() > max_w):
                max_w = state_plot.max()

            plt.xlim((xlim_start, xlim_end))
            plt.ylim((min_w*0.9, max_w*1.1))
            plt.grid(True)

            if (n1 % 8 == 7):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath + 'l2_weight_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath + 'l2_weight_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath + 'l2_weight_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + 'l2_weight_value_test' + str(fig_counter-1) + '.png')
        plt.close(1)

if(print_l1_weights and print_l1_state):
    analog_plots = {
                'times': [],
                'weights': {}
                }
    with open('./Weights/l1_weights_time.npy', 'rb') as f:
        i_count = 0
        time_plot_lst = []
        for sim in sim_json['simulations']:
            time_plot = np.load(f)
            analog_plots['times'].append(time_plot + plot_start_lst[i_count])
            i_count+=1

    with open('./Weights/l1_weights_value.npy', 'rb') as f:
        weights_plots = {str(n1)+":"+str(n0):[] for n1 in range(N1_Neurons) for n0 in range(N0_Neurons)}
        for sim in sim_json['simulations']:
            for n1 in range(N1_Neurons):
                for n0 in range(N0_Neurons):
                    state_plot = np.load(f)
                    weights_plots[str(n1)+":"+str(n0)].append(state_plot)

        analog_plots['weights'] = weights_plots

    fig_counter = 1
    stop_counter = 0

    for n1 in range(N1_Neurons):
        for n0 in range(N0_Neurons):

            if (n0 % 8 == 0):
                min_w = +10
                max_w = -10
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((8,1), (n0 % 8,0))
            ax1.set_title(str(n0)+"::"+str(n1))

            i_count = 0
            for time_plot in analog_plots['times']:
                state_plot = analog_plots['weights'][str(n1)+":"+str(n0)][i_count]
                plt.plot(time_plot, state_plot)
                i_count+=1

            if(state_plot.min() < min_w):
                min_w = state_plot.min()
            if(state_plot.max() > max_w):
                max_w = state_plot.max()

            plt.xlim((xlim_start, xlim_end))
            plt.ylim((min_w*0.9, max_w*1.1))
            plt.grid(True)

            if (n0 % 8 == 7):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath + 'l1_weight_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath + 'l1_weight_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath + 'l1_weight_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + 'l1_weight_value_test' + str(fig_counter-1) + '.png')
        plt.close(1)

if(print_l2_traces and learning_2_phase and print_l2_state):
    analog_plots = {
                'times': [],
                'apre': {},
                'apost': {},
                'reward': {},
                'punish': {}
                }
    with open('./Weights/l2_trace_time.npy', 'rb') as f:
        i_count = 0
        time_plot_lst = []
        for sim in sim_json['simulations']:
            time_plot = np.load(f)
            analog_plots['times'].append(time_plot + plot_start_lst[i_count])
            i_count+=1

    with open('./Weights/l2_trace_value.npy', 'rb') as f:
        apre_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        apost_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        reward_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        punish_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        for sim in sim_json['simulations']:
            for n2 in range(N2_Neurons):
                for n1 in range(N1_Neurons):
                    state_plot = np.load(f)
                    apre_plots[str(n2)+":"+str(n1)].append(state_plot)
                    state_plot = np.load(f)
                    apost_plots[str(n2)+":"+str(n1)].append(state_plot)
                    state_plot = np.load(f)
                    reward_plots[str(n2)+":"+str(n1)].append(state_plot)
                    state_plot = np.load(f)
                    punish_plots[str(n2)+":"+str(n1)].append(state_plot)

        analog_plots['apre'] = apre_plots
        analog_plots['apost'] = apost_plots
        analog_plots['reward'] = reward_plots
        analog_plots['punish'] = punish_plots


    fig_counter = 1
    stop_counter = 0

    for n2 in range(N2_Neurons):
        for n1 in range(N1_Neurons):

            if (n1 % 8 == 0):
                min_w = +10
                max_w = -10
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((8,1), (n1 % 8,0))
            ax1.set_title(str(n1)+"::"+str(n2))

            i_count = 0
            for time_plot in analog_plots['times']:
                state_plot = analog_plots['apre'][str(n2)+":"+str(n1)][i_count]
                plt.plot(time_plot, state_plot,'r')
                state_plot = analog_plots['apost'][str(n2)+":"+str(n1)][i_count]
                plt.plot(time_plot, state_plot,'b')
                i_count+=1

            plt.xlim((xlim_start, xlim_end))
            plt.ylim((-0.0004, +0.0004))
            plt.grid(True)

            if (n1 % 8 == 7):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath + 'l2_trace_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath + 'l2_trace_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath + 'l2_trace_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + 'l2_trace_value_test' + str(fig_counter-1) + '.png')
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

if learning_2_phase:
    with open('./Weights/l2_weights.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            weight_matrix = np.load(f)
            max_w = weight_matrix.max()
            for n2 in range(N2_Neurons):
                plt.figure(1)
                weight_img = np.reshape(weight_matrix[n2,:], (10,10));
                weight_img = weight_img/max_w*255
                plt.imshow(weight_img, cmap=plt.get_cmap('gray'))
                plt.savefig(figpath + 'l2_weights_img_' + str(i_count) + '_' + str(n2) + '.png')
                plt.close(1)
            i_count+=1

