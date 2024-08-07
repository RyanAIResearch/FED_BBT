import pickle
import matplotlib.pyplot as plt
import numpy as np

popsize = [5, 10]
pert_rate = [0.2, 0.5, 0.8]

for pop in popsize:
    for pert in pert_rate:

        # f = open("/workspace/Black-Box-Tuning/results/agnews/schedule_sigma/prompt/fedbbt_fitness_iid0_alpha1.0_popsize5.pkl", "rb+")
        f_orig = open("/workspace/Black-Box-Tuning/results/sst2/larger_global_pop_new_sigma_pert/prompt/fedbbt_fitness_orig_iid0_alpha0.5_popsize{}_perturtb1_pertrate{}.pkl".format(pop, pert), "rb+")

        orig_dict = pickle.load(f_orig)

        f_orig.close()

        f_pert = open("/workspace/Black-Box-Tuning/results/sst2/larger_global_pop_new_sigma_pert/prompt/fedbbt_fitness_pert_iid0_alpha0.5_popsize{}_perturtb1_pertrate{}.pkl".format(pop, pert), "rb+")

        pert_dict = pickle.load(f_pert)

        f_pert.close()

        colormap = plt.cm.gist_ncar
        # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0,0.9, 10)])

        for i in range(1):
            plt.plot(np.max(orig_dict[i], axis=-1), '--')

        for i in range(1):
            plt.plot(np.min(pert_dict[i], axis=-1), '.')

        plt.xlabel("communication round")
        plt.ylabel("loss")
        plt.legend(["clean", "perturbance"])

        plt.savefig("fitness_pop{}_pert{}.png".format(pop, pert))