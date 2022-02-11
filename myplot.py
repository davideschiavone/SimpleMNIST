import matplotlib.pyplot as plt
import numpy as np

figpath = './figures/'

with open('sim_values.npy', 'rb') as f:
        plot_start = np.load(f)
        plot_end   = np.load(f)
        N0_Neurons = np.load(f)
        N1_Neurons = np.load(f)
        N2_Neurons = np.load(f)
        Reward_Neurons = np.load(f)
        testing_phase = np.load(f)
        print_neuron_l0     = np.load(f)
        print_l1_membrana   = np.load(f)
        print_l1_weights    = np.load(f)
        print_l1_traces     = np.load(f)
        print_l2_membrana   = np.load(f)
        print_l2_weights    = np.load(f)
        print_l2_traces     = np.load(f)
        print_neuron_reward = np.load(f)
        print_neuron_l1     = np.load(f)
        print_neuron_l2     = np.load(f)


if(print_neuron_l0):
    with open('./Weights/l0_stream.npy', 'rb') as f:
        n0_times   = np.load(f)
        n0_indices = np.load(f)
        n0_times   = n0_times*1000
        plt.figure(1)
        plt.title("Input Neuron Stream")
        plt.plot(n0_times, n0_indices, '.k')
        plt.ylim((-0.5,N0_Neurons))
        plt.xlim((plot_start, plot_end))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index');
        if not testing_phase:
            plt.savefig(figpath + 'l0_stream.png')
        else:
            plt.savefig(figpath + 'l0_stream_test.png')
        plt.close(1)

if(print_neuron_reward):
    with open('./Weights/lreward_stream.npy', 'rb') as f:
        nr_times   = np.load(f)
        nr_indices = np.load(f)
        nr_times   = nr_times*1000
        plt.figure(1)
        plt.title("Reward Neuron Stream")
        plt.plot(nr_times, nr_indices, '*r')
        plt.ylim((-0.5,Reward_Neurons))
        plt.xlim((plot_start, plot_end))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index');
        if not testing_phase:
            plt.savefig(figpath + 'lreward_stream.png')
    plt.close(1)

if(print_neuron_l1):
    with open('./Weights/l1_stream.npy', 'rb') as f:
        n1_times   = np.load(f)
        n1_indices = np.load(f)
        n1_times   = n1_times*1000
        plt.figure(1)
        plt.title("L1 Neuron Stream")
        plt.plot(n1_times, n1_indices, '.k')
        plt.ylim((-0.5,N1_Neurons))
        plt.xlim((plot_start, plot_end))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index');
        if not testing_phase:
            plt.savefig(figpath + 'l1_stream.png')
        else:
            plt.savefig(figpath + 'l1_stream_test.png')
        plt.close(1)

if(print_neuron_l2):
    with open('./Weights/l2_stream.npy', 'rb') as f:
        n2_times   = np.load(f)
        n2_indices = np.load(f)
        n2_times   = n2_times*1000
        plt.figure(1)
        plt.title("L2 Neuron Stream")
        plt.plot(n2_times, n2_indices, '.k')
        plt.ylim((-0.5,N2_Neurons))
        plt.xlim((plot_start, plot_end))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index');
        if not testing_phase:
            plt.savefig(figpath + 'l2_stream.png')
        else:
            plt.savefig(figpath + 'l2_stream_test.png')
        plt.close(1)

if(print_l1_membrana):
    with open('./Weights/l1_membrana_time.npy', 'rb') as f:
        time_plot = np.load(f)
    with open('./Weights/l1_membrana_value.npy', 'rb') as f:

        fig_counter = 1
        for n1 in range(N1_Neurons):
            state_plot = np.load(f)

            if (n1 % 4 == 0):
                plt.figure(1)
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((4,1), (n1 % 4,0))
            ax1.set_title("L1 Neuron Membrana neuron " + str(n1))
            plt.plot(time_plot, state_plot)

            if (n1 % 4 == 3):

                plt.xlabel('Time (ms)')
                plt.ylabel('V')
                plt.xlim((plot_start, plot_end))
                plt.ylim((-0.22, 0.12))
                plt.grid(True)
                if not testing_phase:
                    plt.savefig(figpath + 'l1_membrana_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath + 'l1_membrana_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

        if ( (n1-1) % 4 < 3):
            plt.xlabel('Time (ms)')
            plt.ylabel('V')
            plt.xlim((plot_start, plot_end))
            plt.ylim((-0.22, 0.12))
            plt.grid(True)
            if not testing_phase:
                plt.savefig(figpath + 'l1_membrana_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + 'l1_membrana_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

if(print_l2_membrana):
    with open('./Weights/l2_membrana_time.npy', 'rb') as f:
        time_plot = np.load(f)
    with open('./Weights/l2_membrana_value.npy', 'rb') as f:
        fig_counter = 1
        stop_counter = 0

        for n2 in range(N2_Neurons):
            state_plot = np.load(f)

            if (n2 % 4 == 0):
                plt.figure(1)
                stop_counter = 1
                fig_counter = fig_counter + 1

            ax1 = plt.subplot2grid((4,1), (n2 % 4,0))
            ax1.set_title("neuron " + str(n2))
            plt.plot(time_plot, state_plot)

            if (n2 % 4 == 3):
                stop_counter = 0
                plt.xlabel('Time (ms)')
                plt.ylabel('V')
                plt.xlim((plot_start, plot_end))
                plt.ylim((-0.22, 0.12))
                plt.grid(True)
                if not testing_phase:
                    plt.savefig(figpath + 'l2_membrana_value' + str(fig_counter-1) + '.png')
                else:
                    plt.savefig(figpath + 'l2_membrana_value_test' + str(fig_counter-1) + '.png')
                plt.close(1)

        if (stop_counter):
            plt.xlabel('Time (ms)')
            plt.ylabel('V')
            plt.xlim((plot_start, plot_end))
            plt.ylim((-0.22, 0.12))
            plt.grid(True)
            if not testing_phase:
                plt.savefig(figpath + 'l2_membrana_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + 'l2_membrana_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)


if(print_l2_weights):
    with open('./Weights/l2_weights_time.npy', 'rb') as f:
        time_plot = np.load(f)
    with open('./Weights/l2_weights_value.npy', 'rb') as f:
        fig_counter = 1
        stop_counter = 0
        for n2 in range(N2_Neurons):
            for weights in range(N1_Neurons):

                state_plot = np.load(f)

                if (weights % 8 == 0):
                    plt.figure(1)
                    stop_counter = 1
                    fig_counter = fig_counter + 1

                ax1 = plt.subplot2grid((8,1), (weights % 8,0))
                ax1.set_title(str(weights)+"::"+str(n2))
                plt.plot(time_plot, state_plot)

                if (weights % 8 == 7):
                    stop_counter = 0
                    plt.xlim((plot_start, plot_end))
                    plt.ylim((-0.22, 0.12))
                    plt.grid(True)
                    if not testing_phase:
                        plt.savefig(figpath + 'l2_weight_value' + str(fig_counter-1) + '.png')
                    else:
                        plt.savefig(figpath + 'l2_weight_value_test' + str(fig_counter-1) + '.png')
                    plt.close(1)

        if (stop_counter):
            plt.xlabel('Time (ms)')
            plt.ylabel('V')
            plt.xlim((plot_start, plot_end))
            plt.ylim((-0.22, 0.12))
            plt.grid(True)
            if not testing_phase:
                plt.savefig(figpath + 'l2_weight_value' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + 'l2_weight_value_test' + str(fig_counter-1) + '.png')
            plt.close(1)

exit()


if(print_l1_weights):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("N" + str(n1) + " Weights")

        min_w = +1
        max_w = -1

        for weights in weights_to_plot:
            time_plot = []
            state_plot = []

            sample_time_condition = (S010state.t/ms < plot_end) & (S010state.t/ms >= plot_start)
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
        plt.xlim((plot_start, plot_end))

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

weights2_to_plot = np.arange(0,N1_Neurons, step=20)

np.set_printoptions(threshold=sys.maxsize)
print(weights2_to_plot)

if(print_weights):
    plt.figure(1)

    fig_counter = 1
    for n2 in range(N2_Neurons):

        if (n2 % 4 == 0):
            plt.figure(1)
            fig_counter = fig_counter + 1

        ax1 = plt.subplot2grid((4,1), (n2 % 4,0))
        ax1.set_title("N" + str(n2) + " Weights - layer 2")

        min_w = +1
        max_w = -1

        for weights in weights2_to_plot:
            time_plot = []
            state_plot = []

            sample_time_condition = (S120state.t/ms < plot_end) & (S120state.t/ms >= plot_start)
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


        if (n2 % 4 == 3):
            plt.xlabel('Time (ms)')
            plt.ylabel('Weights')
            plt.ylim((min_w*0.9,max_w*1.1))
            plt.xlim((plot_start, plot_end))
            plt.grid(True)
            if learning_2_phase:
                plt.savefig(figpath + '10_weights_stdp_reward_' + str(fig_counter-1) + '.png')
            else:
                plt.savefig(figpath + '10_test_weights_reward_' + str(fig_counter-1) + '.png')
            plt.close(1)

    if ( (n2-1) % 4 < 3):
        plt.xlabel('Time (ms)')
        plt.ylabel('Weights')
        plt.ylim((min_w*0.9,max_w*1.1))
        plt.xlim((plot_start, plot_end))
        plt.grid(True)
        if learning_2_phase:
            plt.savefig(figpath + '10_weights_stdp_reward_' + str(fig_counter-1) + '.png')
        else:
            plt.savefig(figpath + '10_test_weights_reward_' + str(fig_counter-1) + '.png')
        plt.close(1)


if(print_l1_traces and learning_1_phase):
    plt.figure(1)

    for n1 in range(N1_Neurons):

        ax1 = plt.subplot2grid((N1_Neurons,1), (n1,0))
        ax1.set_title("N" + str(n1) + " Traces")


        for weights in weights_to_plot:
            time_plot = []
            state_plot = []
            state2_plot = []

            sample_time_condition = (S010state.t/ms < plot_end) & (S010state.t/ms >= plot_start)
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
            plt.xlim((plot_start, plot_end))
    plt.savefig(figpath + '11_traces_stdp.png')
    plt.close(1)

if(print_traces and learning_2_phase):
    plt.figure(1)

    fig_counter = 1
    for n2 in range(N2_Neurons):

        if (n2 % 4 == 0):
            plt.figure(1)
            fig_counter = fig_counter + 1

        ax1 = plt.subplot2grid((4,1), (n2 % 4,0))
        ax1.set_title("N" + str(n2) + " Traces")

        for weights in weights2_to_plot:
            time_plot = []
            state_plot = []
            state2_plot = []
            state3_plot = []
            state4_plot = []

            sample_time_condition = (S120state.t/ms < plot_end) & (S120state.t/ms >= plot_start)
            sample_time_index     = np.where(sample_time_condition)[0]
            S120state_times_no_plot  = S120state.t[sample_time_condition]

            sample_time_index     = sample_time_index[0:-1:step];
            S120state_times_no_plot  = S120state_times_no_plot[0:-1:step];

            time_plot = np.concatenate((time_plot, S120state_times_no_plot/ms))
            state_plot = np.concatenate((state_plot, S120state.apre[weights+(n2*N1_Neurons)][sample_time_index]))
            state2_plot = np.concatenate((state2_plot, S120state.apost[weights+(n2*N1_Neurons)][sample_time_index]))
            state3_plot = np.concatenate((state3_plot, S120state.reward[weights+(n2*N1_Neurons)][sample_time_index]))
            state4_plot = np.concatenate((state4_plot, S120state.punish[weights+(n2*N1_Neurons)][sample_time_index]))

            plt.plot(time_plot, state_plot, label='N2::'+str(n2))
            plt.plot(time_plot, state2_plot, label='N2::'+str(n2))
            plt.plot(time_plot, state3_plot*0.0004, label='N2::'+str(n2), color='r')
            plt.plot(time_plot, -state4_plot*0.0004, label='N2::'+str(n2), color='b')


        if (n2 % 4 == 3):
            plt.xlabel('Time (ms)')
            plt.ylabel('Traces')
            plt.ylim((-0.0012,+0.0008))
            plt.xlim((plot_start, plot_end))
            plt.grid(True)
            plt.savefig(figpath + '11_traces_stdp_reward' + str(fig_counter-1) + '.png')
            plt.close(1)

    if ( (n2-1) % 4 < 3):
        plt.xlabel('Time (ms)')
        plt.ylabel('Traces')
        plt.ylim((-0.0012,+0.0008))
        plt.xlim((plot_start, plot_end))
        plt.grid(True)
        plt.savefig(figpath + '11_traces_stdp_reward' + str(fig_counter-1) + '.png')
        plt.close(1)


if print_neuron_l1:
    plt.figure(1)
    plt.title("L1 Neuron Stream")
    for i_count in np.arange(plot_start_time,end_plot):

        color = classes_color[train_y[i_count]]
        for k in range(N1_Neurons):
            sample_time_condition = (N1mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N1mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
            N1mon_times_nk_plot   = N1mon.spike_trains()[k][sample_time_condition]
            N1mon_nspikes_nk_plot = np.ones(size(N1mon_times_nk_plot))*k
            plt.plot(N1mon_times_nk_plot/ms, N1mon_nspikes_nk_plot, '*'+color)
    plt.ylim((0,N1_Neurons))
    plt.xlim((plot_start, plot_end))
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    plt.grid(True)
    if not testing_phase:
        plt.savefig(figpath + '12_l1_neurons.png')
    else:
        plt.savefig(figpath + '12_test_l1_neurons.png')
    plt.close(1)


plt.figure(1)
plt.title("Output Neuron Stream")
for i_count in np.arange(plot_start_time,end_plot):

    color = classes_color[train_y[i_count]]
    for k in range(N2_Neurons):
        sample_time_condition = (N2mon.spike_trains()[k]/ms < (i_count+1)*single_example_time) & (N2mon.spike_trains()[k]/ms >= (i_count)*single_example_time)
        N2mon_times_nk_plot   = N2mon.spike_trains()[k][sample_time_condition]
        N2mon_nspikes_nk_plot = np.ones(size(N2mon_times_nk_plot))*k
        plt.plot(N2mon_times_nk_plot/ms, N2mon_nspikes_nk_plot, '*'+color)
plt.ylim((0,N2_Neurons))
plt.xlim((plot_start, plot_end))
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');
plt.grid(True)
if not testing_phase:
    plt.savefig(figpath + '13_output_neurons.png')
else:
    plt.savefig(figpath + '13_test_output_neurons.png')
