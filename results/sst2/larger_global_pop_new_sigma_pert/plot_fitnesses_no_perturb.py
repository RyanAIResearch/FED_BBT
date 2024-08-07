import pickle
import matplotlib.pyplot as plt
import numpy as np

popsize = [5, 10]


for pop in popsize:

    # f = open("/workspace/Black-Box-Tuning/results/agnews/schedule_sigma/prompt/fedbbt_fitness_iid0_alpha1.0_popsize5.pkl", "rb+")
    f_orig = open("/workspace/Black-Box-Tuning/results/sst2/larger_global_pop_new_sigma_pert/prompt/fedbbt_fitness_orig_iid0_alpha0.5_popsize{}_perturtb0_pertrate0.5.pkl".format(pop), "rb+")

    orig_dict = pickle.load(f_orig)

    f_orig.close()

    f_pert = open("/workspace/Black-Box-Tuning/results/sst2/larger_global_pop_new_sigma_pert/prompt/fedbbt_fitness_pert_iid0_alpha0.5_popsize{}_perturtb0_pertrate0.5.pkl".format(pop), "rb+")

    pert_dict = pickle.load(f_pert)

    f_pert.close()

    colormap = plt.cm.gist_ncar
    # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0,0.9, 10)])

    for i in range(1):
        plt.plot(orig_dict[i], '--')

    for i in range(1):
        plt.plot(pert_dict[i], '.')
    

    plt.xlabel("communication round")
    plt.ylabel("loss")
    plt.legend(["clean", "perturbance"])

    plt.savefig("fitness_pop{}_nopert.png".format(pop))