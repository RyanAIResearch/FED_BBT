import pickle
import matplotlib.pyplot as plt
import numpy as np

# f = open("/workspace/Black-Box-Tuning/results/agnews/schedule_sigma/prompt/fedbbt_fitness_iid0_alpha1.0_popsize5.pkl", "rb+")
f = open("/workspace/Black-Box-Tuning/results/yelpp/larger_global_pop_new_sigma_pert/prompt/fedbbt_fitness_orig_iid0_alpha0.5_popsize5_perturtb0_pertrate0.5.pkl", "rb+")

acc_dict = pickle.load(f)

f.close()

colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0,0.9, 10)])

for i in range(10):
    plt.plot(np.max(iid_dict[i], axis=-1), '--')

for i in range(10):
    plt.plot(np.min(iid_dict[i], axis=-1), '.')

plt.xlabel("communication round")
plt.ylabel("local loss")

plt.savefig("iid_temp.png")