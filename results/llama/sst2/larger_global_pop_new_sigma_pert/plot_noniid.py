import matplotlib.pyplot as pyplot

popsize = 5
# f_name_fedprompt = '/workspace/Black-Box-Tuning/results/llama/sst2/fedprompt/fl_iid_kshot40_lr1e-2_frac1_noscore_norm'

f_name_fl = 'fl_noniid_alpha0.5_kshot40_frac1_lpop5_perturb0.8_normprompt_15_upper22'

acc_fedprompt = []
acc_fedbbt = []



with open(f_name_fl, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:15] == "Global test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_fedbbt.append(acc)


# with open(f_name_fedprompt, encoding='utf-8') as f:
#     for line in f.readlines():
#         if line[:15] == "Global test acc":
#             str_split = line.split(' ')
#             acc = float(str_split[-1])
#             acc_fedprompt.append(acc)

# print(max(acc_fedprompt))
print(max(acc_fedbbt))




# pyplot.plot(acc_fedprompt)
pyplot.plot(acc_fedbbt)

pyplot.legend(["FedBBT"])
pyplot.savefig("noniid_acc.jpg")

pyplot.figure()

norm_fedprompt = []
norm_fedbbt = []



with open(f_name_fl, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:18] == "Global prompt norm":
            str_split = line.split(' ')
            norm = float(str_split[-1])
            norm_fedbbt.append(norm)


# with open(f_name_fedprompt, encoding='utf-8') as f:
#     for line in f.readlines():
#         if line[:18] == "Global prompt norm":
#             str_split = line.split(' ')
#             norm = float(str_split[-1])
#             norm_fedprompt.append(norm)

# print(max(norm_fedprompt))
print(max(norm_fedbbt))




# pyplot.plot(norm_fedprompt)
pyplot.plot(norm_fedbbt)

pyplot.legend(["FedBBT"])
pyplot.savefig("noniid_norm.jpg")