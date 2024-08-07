import pickle
import numpy as np
import matplotlib.pyplot as plt

lp = 5

with open("/workspace/Black-Box-Tuning/results/agnews/prompt/fedbbt_prompt_iid0_alpha1.0_popsize{}.pkl".format(lp), 'rb') as f:
    prompt_noniid_dict = pickle.load(f)

with open("/workspace/Black-Box-Tuning/results/agnews/prompt/fedbbt_prompt_iid1_alpha0.5_popsize{}.pkl".format(lp), 'rb') as f:
    prompt_iid_dict = pickle.load(f)



d_iid = np.linalg.norm(prompt_iid_dict[-1]-prompt_iid_dict[-1], axis=-1)

for i in range(10):
    # d_iid += np.linalg.norm(prompt_iid_dict[i]-prompt_iid_dict[-1], axis=-1)/np.linalg.norm(prompt_iid_dict[-1], axis=-1)
    d_iid += np.linalg.norm(prompt_iid_dict[i]-prompt_iid_dict[-1], axis=-1)
d_iid /= 10

d_noniid = np.linalg.norm(prompt_noniid_dict[-1]-prompt_noniid_dict[-1], axis=-1)

for i in range(10):
    # d_noniid += np.linalg.norm(prompt_noniid_dict[i]-prompt_noniid_dict[-1], axis=-1)/np.linalg.norm(prompt_noniid_dict[-1], axis=-1)
    d_noniid += np.linalg.norm(prompt_noniid_dict[i]-prompt_noniid_dict[-1], axis=-1)
d_noniid /= 10

n_iid = np.linalg.norm(prompt_iid_dict[-1], axis=-1)
n_noniid = np.linalg.norm(prompt_noniid_dict[-1], axis=-1)


# plt.plot(n_iid)
plt.plot(d_iid)
# plt.plot(n_noniid)
plt.plot(d_noniid)

# plt.legend(['norm_iid', 'divergence_iid', 'norm_noniid', 'divergence_noniid'])
plt.legend(['iid', 'noniid'])
plt.xlabel("communication round")
plt.ylabel("divergence")
plt.title("average norm of local differences @ l_pop{}".format(lp))
plt.plot()
plt.savefig("divergence{}.png".format(lp))
