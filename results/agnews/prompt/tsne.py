import pickle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

lp = 5

with open("/workspace/Black-Box-Tuning/results/agnews/prompt/fedbbt_prompt_iid0_alpha1.0_popsize{}.pkl".format(lp), 'rb') as f:
    prompt_noniid_dict = pickle.load(f)

with open("/workspace/Black-Box-Tuning/results/agnews/prompt/fedbbt_prompt_iid1_alpha0.5_popsize{}.pkl".format(lp), 'rb') as f:
    prompt_iid_dict = pickle.load(f)


print(np.linalg.norm(prompt_noniid_dict[-1][0]-prompt_iid_dict[-1][0]))
print(np.linalg.norm(prompt_noniid_dict[-1][-1]-prompt_iid_dict[-1][-1]))

y = []
prompt_noniid = np.concatenate([prompt_noniid_dict[-1], prompt_noniid_dict[1], prompt_noniid_dict[5], prompt_noniid_dict[7]], axis=0)
y += ['noniid_server' for i in range(prompt_noniid_dict[-1].shape[0])]
y += ['noniid_client1' for i in range(prompt_noniid_dict[1].shape[0])]
y += ['noniid_client5' for i in range(prompt_noniid_dict[5].shape[0])]
y += ['noniid_client7' for i in range(prompt_noniid_dict[7].shape[0])]
prompt_iid = np.concatenate([prompt_iid_dict[-1], prompt_iid_dict[1], prompt_iid_dict[5], prompt_iid_dict[7]], axis=0)
y += ['iid_server' for i in range(prompt_iid_dict[-1].shape[0])]
y += ['iid_client1' for i in range(prompt_iid_dict[1].shape[0])]
y += ['iid_client5' for i in range(prompt_iid_dict[5].shape[0])]
y += ['iid_client7' for i in range(prompt_iid_dict[7].shape[0])]
prompts = np.concatenate([prompt_noniid, prompt_iid], axis=0)

print(f'prompts of shape: {prompts.shape}')

p_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=50).fit_transform(prompts)





df = pd.DataFrame()
df["y"] = y
df["comp-1"] = p_embedded[:,0]
df["comp-2"] = p_embedded[:,1]

plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 8),
                data=df)
plot.set(title="prompts T-SNE projection")
fig = plot.get_figure()
fig.savefig(f'tsne_{lp}.png')