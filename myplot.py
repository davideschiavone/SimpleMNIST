import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import os

figpath = './figures/'

json_file = open('sim_values.json')
sim_json  = json.load(json_file)

argc = len(sys.argv)

if argc > 1:
    xlim_start_idx = int(sys.argv[1])
else:
    xlim_start_idx = 0

xlim_start = sim_json['simulations'][xlim_start_idx]['plot_start']

total_duration = 0
plot_start_lst = [ sim_json['simulations'][xlim_start_idx]['plot_start'] ]

if argc > 2:
    xlim_end_idx = int(sys.argv[2])
else:
    xlim_end_idx = 0


i_count = 0
for sim in sim_json['simulations']:
    total_duration+= sim['plot_end'] - sim['plot_start']
    plot_start_lst.append(total_duration)
    if argc > 2:
        if i_count >= xlim_end_idx:
            break
    i_count+=1

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
print_l1_state      = sim_json['parameters']['print_l1_state']
print_l2_membrana   = sim_json['parameters']['print_l2_membrana']
print_l2_weights    = sim_json['parameters']['print_l2_weights']
print_l2_traces     = sim_json['parameters']['print_l2_traces']
print_l2_state      = sim_json['parameters']['print_l2_state']
print_statistics    = sim_json['parameters']['print_statistics']
print_neuron_reward = sim_json['parameters']['print_neuron_reward']
print_neuron_l1     = sim_json['parameters']['print_neuron_l1']
print_neuron_l2     = sim_json['parameters']['print_neuron_l2']
learning_1_phase    = sim_json['parameters']['learning_1_phase']
learning_2_phase    = sim_json['parameters']['learning_2_phase']
print_l1_weights_charts = False

if(print_neuron_l0):
    plt.figure(1)
    plt.title("Input Neuron Stream")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    with open('./Weights/l0_stream.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            n0_times   = np.load(f)
            n0_indices = np.load(f)
            n0_times   = n0_times*1000 + plot_start_lst[i_count]
            plt.plot(n0_times, n0_indices, '.k')
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
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
    with open('./Weights/lreward_stream.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            nr_times   = np.load(f)
            nr_indices = np.load(f)
            nr_times   = nr_times*1000 + plot_start_lst[i_count]
            plt.plot(nr_times, nr_indices, '*r')
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
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

    if(print_statistics):
        n1_times_list = []
        n1_indices_list = []

    with open('./Weights/l1_stream.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            n1_times   = np.load(f)
            n1_indices = np.load(f)
            n1_times   = n1_times*1000 + plot_start_lst[i_count]
            if(print_statistics):
                n1_times_list.append(n1_times)
                n1_indices_list.append(n1_indices)
            plt.plot(n1_times, n1_indices, '.k')
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
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
    if(print_statistics):
        n2_times_list = []
        n2_indices_list = []
    with open('./Weights/l2_stream.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            n2_times   = np.load(f)
            n2_indices = np.load(f)
            n2_times   = n2_times*1000 + plot_start_lst[i_count]
            if(print_statistics):
                n2_times_list.append(n2_times)
                n2_indices_list.append(n2_indices)
            plt.plot(n2_times, n2_indices, '.k')
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1
    plt.ylim((-0.5,N2_Neurons))
    plt.xlim((xlim_start, xlim_end))
    if not testing_phase:
        plt.savefig(figpath + 'l2_stream.png')
    else:
        plt.savefig(figpath + 'l2_stream_test.png')
    plt.close(1)


if(print_statistics):
    plt.figure(1)
    plt.title("Classes")
    plt.xlabel('Time (ms)')
    plt.ylabel('Class')
    with open('./y_values.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            y_values = np.load(f)
            times    = np.arange(0, len(y_values)*25, 25) + plot_start_lst[i_count]
            i_count  = i_count + 1
            plt.plot(times, y_values, '.k')
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    plt.ylim((-0.5, 10))
    plt.xlim((xlim_start, xlim_end))
    if not testing_phase:
        plt.savefig(figpath + 'classes.png')
    else:
        plt.savefig(figpath + 'classes_test.png')
    plt.close(1)

if(print_neuron_l1 and print_statistics):
    y_values_list = []
    with open('./y_values.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            y_values_list.append(np.load(f))
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    i_count = 0

    class_stats = np.zeros(10);

    stats_dict = {
                'per_class_firing_neurons' : np.zeros((10,N1_Neurons)),
                'per_class_firing_neurons_norm' : np.zeros((10,N1_Neurons)),
                'per_class_max_firing_neurons' : np.zeros(10),
                'per_class_min_firing_neurons' : 9999999999*np.ones(10),
                'per_class_samples' : np.zeros(10)
            }

    with open('./l1_firing.txt', 'w') as f:

        sample_proc = 0
        sim_step = 0
        for y_values in y_values_list:
            num_samples = len(y_values)
            for samples in range(num_samples):
                class_sample   = y_values[samples]
                n1t            = n1_times_list[i_count]
                time_condition = (n1t < sim_step + 25*(samples+1)) & (n1t > sim_step + 25*samples)
                n1i            = n1_indices_list[i_count]
                index          = np.where(time_condition)[0]
                n1t_cond       = n1t[index]
                n1i_cond       = n1i[index]
                stats_dict['per_class_samples'][class_sample] += 1
                stats_dict['per_class_firing_neurons'][class_sample,n1i_cond]+=np.ones(n1i_cond.shape)
                stats_dict['per_class_max_firing_neurons'][class_sample] = stats_dict['per_class_max_firing_neurons'][class_sample] if stats_dict['per_class_max_firing_neurons'][class_sample] > len(n1i_cond) else len(n1i_cond)
                stats_dict['per_class_min_firing_neurons'][class_sample] = stats_dict['per_class_min_firing_neurons'][class_sample] if stats_dict['per_class_min_firing_neurons'][class_sample] < len(n1i_cond) else len(n1i_cond)

                sample_proc += 1

            sim_step = sim_step + 25*num_samples
            i_count = i_count+1

        for y in range(10):
            if stats_dict['per_class_samples'][y] != 0:
                stats_dict['per_class_firing_neurons_norm'][y] = stats_dict['per_class_firing_neurons'][y]/stats_dict['per_class_samples'][y]

            print("stats_dict['per_class_firing_neurons'][" + str(y) + "] is: ", file=f)
            np.savetxt(f, stats_dict['per_class_firing_neurons'][y].astype(int), fmt='%i', newline=",")

            print("\nstats_dict['per_class_firing_neurons_norm'][" + str(y) + "] is: ", file=f)
            np.savetxt(f, stats_dict['per_class_firing_neurons_norm'][y], fmt='%.4f', newline=",")

            print("\nMax Firing: " + str(stats_dict['per_class_max_firing_neurons']),file=f)
            print("Min Firing: " + str(stats_dict['per_class_min_firing_neurons']),file=f)
            print("Total Samples: " + str( stats_dict['per_class_samples'][y] ),file=f)

        print("\nGeneral view\n", file=f)
        print("Max of Max Firing: " + str(np.max(stats_dict['per_class_max_firing_neurons'])) ,file=f)
        print("Max of Max Firing Class: " + str(np.argmax(stats_dict['per_class_max_firing_neurons'])) ,file=f)
        print("Min of Min Firing: " + str(np.min(stats_dict['per_class_min_firing_neurons'])) ,file=f)
        print("Min of Min Firing Class: " + str(np.argmin(stats_dict['per_class_min_firing_neurons'])) ,file=f)

    plt.figure(1)
    plt.title("L1 Statistics")
    plt.xlabel('N1 Neurons')
    plt.ylabel('Counts')
    plt.grid(True)
    for y in range(10):
        ax1 = plt.subplot2grid((10,1), (y,0))
        ax1.set_title(str(y))
        plt.plot(range(N1_Neurons),stats_dict['per_class_firing_neurons_norm'][y])
    plt.ylim((0,0.4))
    plt.xlim((0, N1_Neurons))

    if not testing_phase:
        plt.savefig(figpath + 'l1_statistics.png')
    else:
        plt.savefig(figpath + 'l1_statistics_test.png')
    plt.close(1)

if(print_neuron_l2 and print_statistics):
    stat_matrix = np.zeros((10, N2_Neurons));
    y_values_list = []
    with open('./y_values.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            y_values_list.append(np.load(f))
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    i_count = 0
    for y_values in y_values_list:
        num_samples = len(y_values)
        for samples in range(num_samples):
            n2t            = n2_times_list[i_count]
            time_condition = (n2t < 25*(samples+1)) & (n2t > 25*samples)
            n2i            = n2_indices_list[i_count]
            index          = np.where(time_condition)[0]
            n2t_cond       = n2t[index]
            n2i_cond       = n2i[index]
            stat_matrix[y_values[samples], n2i_cond]+=np.ones(n2i_cond.shape)

        i_count = i_count+1
        for y in range(10):
            sumrow = stat_matrix[y].sum(axis=0)
            if sumrow != 0:
                stat_matrix[y] /= sumrow

    plt.figure(1)
    plt.title("L2 Statistics")
    plt.xlabel('N2 Neurons')
    plt.ylabel('Counts')
    plt.grid(True)
    for y in range(10):
        ax1 = plt.subplot2grid((10,1), (y,0))
        ax1.set_title(str(y))
        plt.plot(range(N2_Neurons),stat_matrix[y])
    plt.ylim((0,0.4))
    plt.xlim((0, N2_Neurons))

    if not testing_phase:
        plt.savefig(figpath + 'l2_statistics.png')
    else:
        plt.savefig(figpath + 'l2_statistics_test.png')
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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    with open('./Weights/l1_membrana_value.npy', 'rb') as f:
        neurons_plots = {str(n1):[] for n1 in range(N1_Neurons)}
        i_count = 0
        for sim in sim_json['simulations']:
            for n1 in range(N1_Neurons):
                state_plot = np.load(f)
                neurons_plots[str(n1)].append(state_plot)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    with open('./Weights/l2_membrana_value.npy', 'rb') as f:
        neurons_plots = {str(n2):[] for n2 in range(N2_Neurons)}
        i_count = 0
        for sim in sim_json['simulations']:
            for n2 in range(N2_Neurons):
                state_plot = np.load(f)
                neurons_plots[str(n2)].append(state_plot)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1
        analog_plots['neurons'] = neurons_plots

    fig_counter = 1
    stop_counter = 0
    for n2 in range(N2_Neurons):

        figpath_n2 = figpath + 'membrana_l2'
        if not os.path.exists(figpath_n2):
            os.makedirs(figpath_n2)

        if (n2 % 5 == 0):
            plt.figure(1)
            stop_counter = 1
            fig_counter = fig_counter + 1

        ax1 = plt.subplot2grid((5,1), (n2 % 5,0))
        ax1.set_title("neuron " + str(n2))
        plt.xlim((xlim_start, xlim_end))
        plt.ylim((-0.22, 0.12))
        plt.grid(True)

        i_count = 0
        for time_plot in analog_plots['times']:
            state_plot = analog_plots['neurons'][str(n2)][i_count]
            plt.plot(time_plot, state_plot)
            i_count+=1

        if (n2 % 5 == 4):
            stop_counter = 0
            if not testing_phase:
                plt.savefig(figpath_n2 + '/l2_membrana_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath_n2 + '/l2_membrana_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath_n2 + '/l2_membrana_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath_n2 + '/l2_membrana_value_test' + str(fig_counter-1) + '.png')
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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    with open('./Weights/l2_weights_value.npy', 'rb') as f:
        weights_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        i_count = 0
        for sim in sim_json['simulations']:
            for n2 in range(N2_Neurons):
                for n1 in range(N1_Neurons):
                    state_plot = np.load(f)
                    weights_plots[str(n2)+":"+str(n1)].append(state_plot)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

        analog_plots['weights'] = weights_plots

    fig_counter = 1
    stop_counter = 0

    for n2 in range(N2_Neurons):
        figpath_n2 = figpath + 'weights_l2/n' + str(n2)
        if not os.path.exists(figpath_n2):
            os.makedirs(figpath_n2)

        for n1 in range(N1_Neurons):

            if (n1 % 10 == 0):
                min_w = +10
                max_w = -10
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((10,1), (n1 % 10,0))
            ax1.set_title(str(n1)+"::"+str(n2))

            i_count = 0
            for time_plot in analog_plots['times']:
                state_plot = analog_plots['weights'][str(n2)+":"+str(n1)][i_count]
                if(state_plot.min() < min_w):
                    min_w = state_plot.min()
                if(state_plot.max() > max_w):
                    max_w = state_plot.max()
                plt.plot(time_plot, state_plot)
                i_count+=1

            plt.xlim((xlim_start, xlim_end))
            plt.ylim((min_w*0.9, max_w*1.1))
            plt.grid(True)

            if (n1 % 10 == 9):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath_n2 + '/l2_weight_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath_n2 + '/l2_weight_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

        if (stop_counter):
            stop_counter = 0
            if not testing_phase:
                plt.savefig(figpath_n2 + '/l2_weight_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath_n2 + '/l2_weight_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath_n2 + '/l2_weight_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath_n2 + '/l2_weight_value_test' + str(fig_counter-1) + '.png')
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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    with open('./Weights/l1_weights_value.npy', 'rb') as f:
        weights_plots = {str(n1)+":"+str(n0):[] for n1 in range(N1_Neurons) for n0 in range(N0_Neurons)}
        i_count = 0
        for sim in sim_json['simulations']:
            for n1 in range(N1_Neurons):
                for n0 in range(N0_Neurons):
                    state_plot = np.load(f)
                    weights_plots[str(n1)+":"+str(n0)].append(state_plot)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

        analog_plots['weights'] = weights_plots


    if(print_l1_weights_charts):
        fig_counter = 1
        stop_counter = 0

        for n1 in range(N1_Neurons):
            figpath_n1 = figpath + 'weights_l1/n' + str(n1)
            if not os.path.exists(figpath_n1):
                os.makedirs(figpath_n1)
            print("priting l1 weights of neuron " + str(n1))
            for n0 in range(N0_Neurons):

                if (n0 % 14 == 0):
                    min_w = +10
                    max_w = -10
                    plt.figure(1)
                    stop_counter = 1
                    fig_counter = fig_counter + 1

                ax1 = plt.subplot2grid((14,1), (n0 % 14,0))
                ax1.set_title(str(n0)+"::"+str(n1))

                i_count = 0
                for time_plot in analog_plots['times']:
                    state_plot = analog_plots['weights'][str(n1)+":"+str(n0)][i_count]
                    if(state_plot.min() < min_w):
                        min_w = state_plot.min()
                    if(state_plot.max() > max_w):
                        max_w = state_plot.max()
                    plt.plot(time_plot, state_plot, '*r')

                    i_count+=1

                plt.xlim((xlim_start, xlim_end))
                plt.ylim((min_w*0.9, max_w*1.1))
                plt.grid(True)

                if (n0 % 14 == 13):
                    stop_counter = 0
                    if not testing_phase:
                        plt.savefig(figpath_n1 + '/l1_weight_value' + str(fig_counter-1) + '.png')
                    else:
                        plt.savefig(figpath_n1 + '/l1_weight_value_test' + str(fig_counter-1) + '.png')
                    plt.close(1)

            if (stop_counter):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath_n1 + '/l1_weight_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath_n1 + '/l1_weight_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

        if (stop_counter):
            if not testing_phase:
                plt.savefig(figpath_n1 + '/l1_weight_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath_n1 + '/l1_weight_value_test' + str(fig_counter-1) + '.png')
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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

    with open('./Weights/l2_trace_value.npy', 'rb') as f:
        apre_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        apost_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        reward_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        punish_plots = {str(n2)+":"+str(n1):[] for n2 in range(N2_Neurons) for n1 in range(N1_Neurons)}
        i_count = 0
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
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

        analog_plots['apre'] = apre_plots
        analog_plots['apost'] = apost_plots
        analog_plots['reward'] = reward_plots
        analog_plots['punish'] = punish_plots


    fig_counter = 1
    stop_counter = 0

    for n2 in range(N2_Neurons):

        figpath_n2 = figpath + 'traces_l2/n' + str(n2)
        if not os.path.exists(figpath_n2):
            os.makedirs(figpath_n2)

        for n1 in range(N1_Neurons):

            if (n1 % 10 == 0):
                min_w = +10
                max_w = -10
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((10,1), (n1 % 10,0))
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

            if (n1 % 10 == 9):
                stop_counter = 0
                if not testing_phase:
                    plt.savefig(figpath_n2 + '/l2_trace_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath_n2 + '/l2_trace_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

    if (stop_counter):
        if not testing_phase:
            plt.savefig(figpath_n2 + '/l2_trace_value' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath_n2 + '/l2_trace_value_test' + str(fig_counter-1) + '.png')
        plt.close(1)

if learning_1_phase:
    with open('./Weights/l1_weights.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            weight_matrix = np.load(f)
            max_w = weight_matrix.max()
            for n1 in range(N1_Neurons):
                figpath_n1 = figpath + 'weights_l1/n' + str(n1)
                if not os.path.exists(figpath_n1):
                    os.makedirs(figpath_n1)
                plt.figure(1)
                weight_img = np.reshape(weight_matrix[n1,:], (28,28));
                weight_img = weight_img/max_w*255
                plt.imshow(weight_img, cmap=plt.get_cmap('gray'))
                plt.savefig(figpath_n1 + '/l1_weights_img_' + str(i_count) + '_' + str(n1) + '.png')
                plt.close(1)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

if learning_2_phase:
    with open('./Weights/l2_weights.npy', 'rb') as f:
        i_count = 0
        for sim in sim_json['simulations']:
            weight_matrix = np.load(f)
            max_w = weight_matrix.max()
            for n2 in range(N2_Neurons):
                figpath_n2 = figpath + 'weights_l2/n' + str(n2)
                if not os.path.exists(figpath_n2):
                    os.makedirs(figpath_n2)
                plt.figure(1)
                weight_img = np.reshape(weight_matrix[n2,:], (10,10));
                weight_img = weight_img/max_w*255
                plt.imshow(weight_img, cmap=plt.get_cmap('gray'))
                plt.savefig(figpath_n2 + '/l2_weights_img_' + str(i_count) + '_' + str(n2) + '.png')
                plt.close(1)
            if argc > 2:
                if i_count >= xlim_end_idx:
                    break
            i_count+=1

