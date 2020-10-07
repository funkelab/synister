from synister.synister_db import SynisterDb
from synister.read_config import read_predict_config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import statistics
import os

def parse_prediction(db_credentials,
                     predict_config_path):

    predict_cfg = read_predict_config(predict_config_path)
    db = SynisterDb(db_credentials, predict_cfg["db_name_data"])

    predictions = db.get_predictions(predict_cfg["split_name"],
                                     predict_cfg["experiment"],
                                     predict_cfg["train_number"],
                                     predict_cfg["predict_number"])

    synapses = db.get_synapses()
    skeletons = db.get_skeletons()

    predicted_synapses =\
    {
            synapse_id: 
            {
                **{"prediction": prediction["prediction"]}, 
                **synapses[synapse_id], 
                **skeletons[synapses[synapse_id]["skeleton_id"]]
                }
            for synapse_id, prediction in predictions.items()
    }

    return predicted_synapses, predict_cfg

def synaptic_cross_confusion_matrix(synapses_target, synapses_source, 
                                    predict_config, normalize=False):

    synapse_types = predict_config["synapse_types"]
    cm = np.zeros([len(synapse_types)] * 2, dtype=float)

    n = 0
    for synapse_id, synapse_data in synapses_source.items():
        if synapse_data["prediction"] == "null":
            continue

        source_class = np.argmax(synapse_data["prediction"])
        target_class = np.argmax(synapses_target[synapse_id]["prediction"])

        cm[target_class, source_class] += 1
        n += 1

    if normalize:
        cm_row_sum = np.sum(cm, axis=1)
        cm = (cm.transpose()/cm_row_sum).transpose()

    return cm

def skeleton_cross_confusion_matrix(synapses_target, synapses_source, predict_config, normalize=False):
    synapse_types = predict_config["synapse_types"]
    cm = np.zeros([len(synapse_types)] * 2, dtype=float)

    skeleton_ids = set([s["skeleton_id"] for s in synapses_source.values()])
    skeleton_to_source_prediction = {skid: [] for skid in skeleton_ids}
    skeleton_to_target_prediction = {skid: [] for skid in skeleton_ids}

    for synapse_id, synapse_data in synapses_source.items():
        zeros_source = np.zeros(len(synapse_types))
        arg_max_source = np.argmax(synapse_data["prediction"])
        zeros_source[arg_max_source] = 1
        skeleton_to_source_prediction[synapse_data["skeleton_id"]].append(
                zeros_source)

        zeros_target = np.zeros(len(synapse_types))
        arg_max_target = np.argmax(synapses_target[synapse_id]["prediction"])
        zeros_target[arg_max_target] = 1
        skeleton_to_target_prediction[synapse_data["skeleton_id"]].append(
                zeros_target)

    for skeleton_id, predictions in skeleton_to_source_prediction.items():
        majority_vote_source = np.argmax(np.sum(np.array(predictions), axis=0))
        majority_vote_target = np.argmax(np.sum(np.array(skeleton_to_target_prediction[skeleton_id]), axis=0))
 
        cm[majority_vote_target, majority_vote_source] += 1

    if normalize:
        cm_row_sum = np.sum(cm, axis=1)
        cm = (cm.transpose()/cm_row_sum).transpose()

    return cm

def expected_probability_matrix(synapses, predict_config):
    synapse_types = predict_config["synapse_types"]
    
    expected_probability = {st: np.zeros(len(synapse_types)) for st in synapse_types}
    samples_per_type = {st: 0 for st in synapse_types}
    for synapse_id, synapse_data in synapses.items():
        if synapse_data["prediction"] == "null":
            continue

        nt_known = synapse_data["nt_known"]
        if len(nt_known)>1:
            raise Warning("More than one known nt")
        nt_known = nt_known[0]
        gt_class = synapse_types.index(nt_known)
        expected_probability[nt_known] += np.array(synapse_data["prediction"])
        samples_per_type[nt_known] += 1

    confusion_matrix = np.zeros([len(synapse_types)] * 2, dtype=float)
    for nt_known in expected_probability:
        expected_probability[nt_known] /= samples_per_type[nt_known]

        gt_class = synapse_types.index(nt_known)
        i = 0
        for p in expected_probability[nt_known]:
            confusion_matrix[gt_class, i] = p
            i += 1

    return confusion_matrix

def synaptic_confusion_matrix(synapses, predict_config, normalization_factor=None, normalize=False, n_min=None):
    synapse_types = predict_config["synapse_types"]
    cm = np.zeros([len(synapse_types)] * 2, dtype=float)

    if normalization_factor is None:
        normalization_factor = {synapse_id: 1. for synapse_id in synapses}

    if not n_min is None:
        synapses_per_skeleton = {}
        for synapse_id, synapse_data in synapses.items():
            skid = synapse_data["skeleton_id"]  
            if not skid in synapses_per_skeleton:
                synapses_per_skeleton[skid] = [synapse_id]
            else:
                synapses_per_skeleton[skid].append(synapse_id)


    n = 0
    for synapse_id, synapse_data in synapses.items():
        if synapse_data["prediction"] == "null":
            continue

        if not n_min is None:
            if len(synapses_per_skeleton[synapse_data["skeleton_id"]]) < n_min:
                continue

        nt_known = synapse_data["nt_known"]
        if len(nt_known)>1:
            raise Warning("More than one known nt")
        nt_known = nt_known[0]

        gt_class = synapse_types.index(nt_known)
        predicted_class = np.argmax(synapse_data["prediction"])

        cm[gt_class, predicted_class] += 1 * normalization_factor[synapse_id]
        n += 1

    if normalize:
        cm_row_sum = np.sum(cm, axis=1)
        cm = (cm.transpose()/cm_row_sum).transpose()

    return cm

def skeleton_confusion_matrix(synapses, predict_config, normalize=False, n_min=None, cutoff=None):
    synapse_types = predict_config["synapse_types"]
    cm = np.zeros([len(synapse_types)] * 2, dtype=float)
    skeleton_ids = set([s["skeleton_id"] for s in synapses.values()])
    skeleton_to_prediction = {skid: [] for skid in skeleton_ids}

    if not n_min is None:
        synapses_per_skeleton = {}
        for synapse_id, synapse_data in synapses.items():
            skid = synapse_data["skeleton_id"]  

            if not skid in synapses_per_skeleton:
                synapses_per_skeleton[skid] = [synapse_id]
            else:
                synapses_per_skeleton[skid].append(synapse_id)

    skeleton_to_gt = {}
    for synapse_id, synapse_data in synapses.items():
        if not n_min is None:
            if len(synapses_per_skeleton[synapse_data["skeleton_id"]]) < n_min:
                continue

        zeros = np.zeros(len(synapse_types))
        arg_max = np.argmax(synapse_data["prediction"])
        zeros[arg_max] = 1
        skeleton_to_prediction[synapse_data["skeleton_id"]].append(
                zeros)
        skeleton_to_gt[synapse_data["skeleton_id"]] =\
                synapse_types.index(synapse_data["nt_known"][0])

    for skeleton_id, predictions in skeleton_to_prediction.items():
        if not n_min is None:
            if len(synapses_per_skeleton[skeleton_id]) < n_min:
                continue
        majority_vote = np.argmax(np.sum(np.array(predictions), axis=0))

        if cutoff is not None:
            gt_transmitter = skeleton_to_gt[skeleton_id]
            majority_vote = np.argmax(np.sum(np.array(predictions), axis=0))
            n_majority_vote = np.sum([p[majority_vote] for p in predictions])
            n_tot = np.sum([1 for p in predictions])
            p_major = n_majority_vote/float(n_tot)
            if p_major < cutoff:
                continue

        cm[skeleton_to_gt[skeleton_id], majority_vote] += 1

    if normalize:
        cm_row_sum = np.sum(cm, axis=1)
        cm = (cm.transpose()/cm_row_sum).transpose()

    return cm

def plot_confusion_matrix(heatmap_values, predict_cfg, 
                          annotation_values=None, name="",
                          save=False,
                          xlabel="Predicted",
                          ylabel="Actual"):
    synapse_types = predict_cfg["synapse_types"]
    df_hm = pd.DataFrame(heatmap_values, index = [i for i in synapse_types],
                                         columns = [i for i in synapse_types])
    if annotation_values is not None:
        annotations = pd.DataFrame(annotation_values, index = [i for i in synapse_types],
                                                      columns = [i for i in synapse_types])
    else:
        annotations = True
 
    plt.figure(figsize = (10,10))
    ax = sn.heatmap(df_hm, annot=annotations, fmt="")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+.5, top-.5)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    train_number = predict_cfg["train_number"]
    iteration = int(predict_cfg["train_checkpoint"].split("_")[-1])
    plt.title("Confusion Matrix t{} i{} {}".format(train_number, iteration, name))
    plt.show()

    if save:
        if name:
            name = "_" + name
        plt.savefig("confusion_matrix_t{}_i{}{}".format(train_number, iteration, name))

def get_accuracy(confusion_matrix):                         
    #returns a tuple (overall accuracy, average accuracy)
    diagonal = np.diagonal(confusion_matrix)
    correct = np.sum(diagonal)
    total = np.sum(confusion_matrix)
    overall_accuracy = correct/total

    accuracies = []
    n = 0
    for synapse_type in confusion_matrix:
        row_sum = np.sum(synapse_type)
        if row_sum > 0:
            accuracies.append(synapse_type[n]/row_sum)
        n+=1

    avg_accuracy = statistics.mean(accuracies)

    return (overall_accuracy, avg_accuracy)

def plot_accuracy(db_credentials, predict_path, 
                  train_numbers, predict_numbers=None,
                  train_labels=None):

    highest_accuracies = []
    train_avg_accuracies = {}
    if predict_numbers is None:
        predict_numbers = {t: None for t in train_numbers}

    for i in range(len(train_numbers)):
        train_number = train_numbers[i]
        setups = os.listdir(predict_path)
        setups = [i for i in setups if i.startswith("setup_t{}".format(train_number))]
        predict_numbers_train = predict_numbers[train_number]

        if predict_numbers_train is None:
            predict_numbers_train = [int(i[i.rindex("p")+1:]) 
                    for i in setups if i.startswith("setup_t{}".format(train_number))]
            predict_numbers_train.sort()

        overall_accuracies = []
        avg_accuracies = []
        iterations = []
        for predict_number in predict_numbers_train:
            synapses, predict_cfg = parse_prediction(db_credentials,
                                                     predict_path +\
                        "setup_t{}_p{}/predict_config.ini".format(train_number, predict_number))
            checkpoint = predict_cfg["train_checkpoint"]
            iteration = checkpoint[checkpoint.rindex("_")+1:]

            cm = synaptic_confusion_matrix(synapses, predict_cfg)
            accuracy = find_accuracy(cm)
            overall_accuracies.append(accuracy[0])
            avg_accuracies.append(accuracy[1])
            iterations.append(int(iteration)/1000)

        highest_accuracies.append((train_number, max(overall_accuracies),max(avg_accuracies)))
        train_avg_accuracies[train_number] = avg_accuracies

        label=None
        if train_labels is not None:
            label = train_labels[train_number]
        sn.lineplot(iterations, overall_accuracies, marker='*', label=label)
 
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy [Total]")
        print("Overall: t{}".format(train_number))
        print(iterations)
        print(overall_accuracies)

    if train_labels is not None:
        plt.legend()
    plt.show()

    for train_number in train_numbers:
        avg_accuracies = train_avg_accuracies[train_number]
        label=None
        if train_labels is not None:
            label = train_labels[train_number]
        sn.lineplot(iterations, avg_accuracies, marker='*', label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy [Average]")
        print("Average: t{}".format(train_number))
        print(iterations)
        print(avg_accuracies)

    if train_labels is not None:
        plt.legend()
    plt.show()
