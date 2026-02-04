import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from os.path import isfile, join
import pickle
from tqdm import tqdm


with open('video_name','rb') as file:
    video_name = pickle.load(file)

path_emb = '../data/CMU-MOSEI_Qwen3-VL-4B-Instruct/hidden_states/train'


def calculate_entropy(features, num_bins=10):
    entropies_sum = 0.0
    for i in range(features.shape[2]):
        feature_values = features[:, :, i]
        counts, bin_edges = np.histogram(feature_values, bins=num_bins)
        probabilities = counts / counts.sum()
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        entropies_sum += entropy
    return entropies_sum / (1.0 * features.shape[2])


res = []
for layer_num in tqdm(range(36, 0, -1)):
    def load_hidden_states(key, _layer_num):
        hidden_states = []
        for video_file in tqdm(video_name[key]):
            with open(join(path_emb, video_file[:-4] + '.pkl'), "rb") as f:  
                hidden_states.append(pickle.load(f)[_layer_num])
        return hidden_states
    len = 0
    Neutral = load_hidden_states('Neutral', layer_num)
    len = max(len, max([x.shape[1] for x in Neutral]))
    Anger = load_hidden_states('Anger', layer_num)
    len = max(len, max([x.shape[1] for x in Anger]))
    Disgust = load_hidden_states('Disgust', layer_num)
    len = max(len, max([x.shape[1] for x in Disgust]))
    Fear = load_hidden_states('Fear', layer_num)
    len = max(len, max([x.shape[1] for x in Fear]))
    Happiness = load_hidden_states('Happiness', layer_num)
    len = max(len, max([x.shape[1] for x in Happiness]))
    Sadness = load_hidden_states('Sadness', layer_num)
    len = max(len, max([x.shape[1] for x in Sadness]))
    Surprise = load_hidden_states('Surprise', layer_num)
    len = max(len, max([x.shape[1] for x in Surprise]))

    Neutral = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Neutral])
    Anger = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Anger])
    Disgust = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Disgust])
    Fear = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Fear])
    Happiness = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Happiness])
    Sadness = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Sadness])
    Surprise = np.concat([np.pad(x.detach().float().cpu().numpy(), ((0, 0), (0, max(0, len - x.shape[1])), (0, 0)), "constant") for x in Surprise])


    sum_kl = 0.0
    sum_ldr = 0.0
    for d in tqdm(range(Neutral.shape[2])):
        sum_kl += (
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Anger[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Disgust[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Fear[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Happiness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Sadness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Neutral[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Anger[:, :, d]), dim=1), F.softmax(torch.from_numpy(Disgust[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Anger[:, :, d]), dim=1), F.softmax(torch.from_numpy(Fear[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Anger[:, :, d]), dim=1), F.softmax(torch.from_numpy(Happiness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Anger[:, :, d]), dim=1), F.softmax(torch.from_numpy(Sadness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Anger[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Disgust[:, :, d]), dim=1), F.softmax(torch.from_numpy(Fear[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Disgust[:, :, d]), dim=1), F.softmax(torch.from_numpy(Happiness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Disgust[:, :, d]), dim=1), F.softmax(torch.from_numpy(Sadness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Disgust[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Fear[:, :, d]), dim=1), F.softmax(torch.from_numpy(Happiness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Fear[:, :, d]), dim=1), F.softmax(torch.from_numpy(Sadness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Fear[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Happiness[:, :, d]), dim=1), F.softmax(torch.from_numpy(Sadness[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Happiness[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)) +
            F.kl_div(F.log_softmax(torch.from_numpy(Sadness[:, :, d]), dim=1), F.softmax(torch.from_numpy(Surprise[:, :, d]), dim=1)))
    sum_kl /= (1.0 * Neutral.shape[2])
    sum_ldr += (
        (np.mean(Neutral) - np.mean(Anger))**2 / (np.var(Neutral) + np.var(Anger) + 0.001) +
        (np.mean(Neutral) - np.mean(Disgust))**2 / (np.var(Neutral) + np.var(Disgust) + 0.001) +
        (np.mean(Neutral) - np.mean(Fear))**2 / (np.var(Neutral) + np.var(Fear) + 0.001) +
        (np.mean(Neutral) - np.mean(Happiness))**2 / (np.var(Neutral) + np.var(Happiness) + 0.001) +
        (np.mean(Neutral) - np.mean(Sadness))**2 / (np.var(Neutral) + np.var(Sadness) + 0.001) +
        (np.mean(Neutral) - np.mean(Surprise))**2 / (np.var(Neutral) + np.var(Surprise) + 0.001) +
        (np.mean(Anger) - np.mean(Disgust))**2 / (np.var(Anger) + np.var(Disgust) + 0.001) +
        (np.mean(Anger) - np.mean(Fear))**2 / (np.var(Anger) + np.var(Fear) + 0.001) +
        (np.mean(Anger) - np.mean(Happiness))**2 / (np.var(Anger) + np.var(Happiness) + 0.001) +
        (np.mean(Anger) - np.mean(Sadness))**2 / (np.var(Anger) + np.var(Sadness) + 0.001) +
        (np.mean(Anger) - np.mean(Surprise))**2 / (np.var(Anger) + np.var(Surprise) + 0.001) +
        (np.mean(Disgust) - np.mean(Fear))**2 / (np.var(Disgust) + np.var(Fear) + 0.001) +
        (np.mean(Disgust) - np.mean(Happiness))**2 / (np.var(Disgust) + np.var(Happiness) + 0.001) +
        (np.mean(Disgust) - np.mean(Sadness))**2 / (np.var(Disgust) + np.var(Sadness) + 0.001) +
        (np.mean(Disgust) - np.mean(Surprise))**2 / (np.var(Disgust) + np.var(Surprise) + 0.001) +
        (np.mean(Fear) - np.mean(Happiness))**2 / (np.var(Fear) + np.var(Happiness) + 0.001) +
        (np.mean(Fear) - np.mean(Sadness))**2 / (np.var(Fear) + np.var(Sadness) + 0.001) +
        (np.mean(Fear) - np.mean(Surprise))**2 / (np.var(Fear) + np.var(Surprise) + 0.001) +
        (np.mean(Happiness) - np.mean(Sadness))**2 / (np.var(Happiness) + np.var(Sadness) + 0.001) +
        (np.mean(Happiness) - np.mean(Surprise))**2 / (np.var(Happiness) + np.var(Surprise) + 0.001) +
        (np.mean(Sadness) - np.mean(Surprise))**2 / (np.var(Sadness) + np.var(Surprise) + 0.001))
    sum_ldr /= (1.0 * Neutral.shape[2])
    features = np.concat([Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise])
    res.append((layer_num, sum_kl, sum_ldr, float(calculate_entropy(features))))
    pd.DataFrame(res, columns=["layer_num", "KL Divergence", "Local Discriminant Ratio", "Entropy"]).to_csv("metrics_by_layers.csv")
