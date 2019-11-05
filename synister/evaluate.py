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

def confusion_matrix(synapses, predict_config, normalization_factor=None):
    synapse_types = predict_config["synapse_types"]
    confusion_matrix = np.zeros([len(synapse_types)] * 2, dtype=float)

    if normalization_factor is None:
        normalization_factor = {synapse_id: 1. for synapse_id in synapses}

    n = 0
    for synapse_id, synapse_data in synapses.items():
        if synapse_data["prediction"] == "null":
            print("skip")
            continue
        print("Insert synapse {}/{}".format(n + 1, len(synapses)))
        nt_known = synapse_data["nt_known"]
        if len(nt_known)>1:
            raise Warning("More than one known nt")
        nt_known = nt_known[0]

        gt_class = synapse_types.index(nt_known)
        predicted_class = np.argmax(synapse_data["prediction"])

        confusion_matrix[gt_class, predicted_class] += 1 * normalization_factor[synapse_id]
        n += 1

    return confusion_matrix

def skeleton_confusion_matrix(synapses, predict_config):
    synapse_types = predict_config["synapse_types"]
    confusion_matrix = np.zeros([len(synapse_types)] * 2, dtype=float)
    skeleton_ids = set([s["skeleton_id"] for s in synapses.values()])
    skeleton_to_prediction = {skid: [] for skid in skeleton_ids}
    skeleton_to_gt = {}
    for synapse_id, synapse_data in synapses.items():
        zeros = np.zeros(len(synapse_types))
        arg_max = np.argmax(synapse_data["prediction"])
        zeros[arg_max] = 1
        skeleton_to_prediction[synapse_data["skeleton_id"]].append(
                zeros)
        skeleton_to_gt[synapse_data["skeleton_id"]] =\
                synapse_types.index(synapse_data["nt_known"][0])


    for skeleton_id, predictions in skeleton_to_prediction.items():
        majority_vote = np.argmax(np.sum(np.array(predictions), axis=0))
        confusion_matrix[skeleton_to_gt[skeleton_id], majority_vote] += 1

    return confusion_matrix

def plot_confusion_matrix(cm, predict_cfg, name=""):
    synapse_types = predict_cfg["synapse_types"]
    df_cm = pd.DataFrame(cm, index = [i for i in synapse_types],
                                columns = [i for i in synapse_types])
    plt.figure(figsize = (10,10))
    ax = sn.heatmap(df_cm, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+.5, top-.5)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    train_number = predict_cfg["train_number"]
    iteration = int(predict_cfg["train_checkpoint"].split("_")[-1])
    plt.title("Confusion Matrix t{} i{} {}".format(train_number, iteration, name))
    if name:
        name = "_" + name
    plt.savefig("confusion_matrix_t{}_i{}{}".format(train_number, iteration, name))

def plot_confusion_matrix_normalized(cm, predict_cfg, name=""):
    synapse_types = predict_cfg["synapse_types"]
    cm_row_sum = np.sum(cm, axis=1)
    cm_normalized = (cm.transpose()/cm_row_sum).transpose()
    df_cm = pd.DataFrame(cm_normalized, index = [i for i in synapse_types],
                                columns = [i for i in synapse_types])
    plt.figure(figsize = (10,10))
    ax = sn.heatmap(df_cm, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+.5, top-.5)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    checkpoint = predict_cfg["train_checkpoint"]
    iteration = checkpoint[checkpoint.rindex("_")+1:]
    train_number = predict_cfg["train_number"]
    plt.title("Normalized Confusion Matrix t{} i{} {}".format(train_number, iteration, name))
    if name:
        name = "_" + name
    plt.savefig("normalized_confusion_matrix_t{}_i{}{}".format(train_number, iteration, name))
    plt.show()

def find_accuracy(confusion_matrix):                         #returns a tuple (overall accuracy, average accuracy)
    diagonal = np.diagonal(confusion_matrix)
    correct = np.sum(diagonal)
    total = np.sum(confusion_matrix)
    overall_accuracy = correct/total

    accuracies = []
    n = 0
    for synapse_type in confusion_matrix:
        row_sum = np.sum(synapse_type)
        accuracies.append(synapse_type[n]/row_sum)
        n+=1

    avg_accuracy = statistics.mean(accuracies)

    return (overall_accuracy, avg_accuracy)

def plot_accuracy(db_credentials, predict_path, train_numbers):
    highest_accuracies = []
    train_avg_accuracies = {}
    for train_number in train_numbers:
        setups = os.listdir(predict_path)
        setups = [i for i in setups if i.startswith("setup_t{}".format(train_number))]
        predict_numbers = [int(i[i.rindex("p")+1:]) for i in setups if i.startswith("setup_t{}".format(train_number))]
        predict_numbers.sort()
        overall_accuracies = []
        avg_accuracies = []
        iterations = []
        for predict_number in predict_numbers:
            synapses, predict_cfg = parse_prediction(db_credentials,
                                                     predict_path+"setup_t{}_p{}/predict_config.ini".format(train_number, predict_number))
            checkpoint = predict_cfg["train_checkpoint"]
            iteration = checkpoint[checkpoint.rindex("_")+1:]

            cm = confusion_matrix(synapses, predict_cfg)
            accuracy = find_accuracy(cm)
            overall_accuracies.append(accuracy[0])
            avg_accuracies.append(accuracy[1])
            iterations.append(int(iteration)/1000)

        highest_accuracies.append((train_number, max(overall_accuracies),max(avg_accuracies)))
        train_avg_accuracies[train_number] = avg_accuracies

        plt.scatter(iterations, overall_accuracies, label="t{} overall accuracy".format(train_number))
        plt.plot(iterations, overall_accuracies)
        plt.legend()

    f = open("t{}_highest_accuracies.txt".format(train_numbers),"w+")
    f.write("(train_number, highest overall accuracy, highest average accuracy)\n")
    for setup in highest_accuracies:
        f.write(str(setup)+"\n")
    f.close()
    plt.savefig("plot_overall_accuracies_t{}".format(train_numbers))
    plt.show()

    for train_number in train_numbers:
        avg_accuracies = train_avg_accuracies[train_number]
        plt.scatter(iterations, avg_accuracies, label="t{} average accuracy".format(train_number))
        plt.plot(iterations, avg_accuracies)
        plt.legend()
    plt.savefig("plot_average_accuracies_t{}".format(train_numbers))
    plt.show()
