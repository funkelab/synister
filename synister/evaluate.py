from synister.synister_db import SynisterDB
from synister.read_config import read_predict_config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import statistics

def parse_prediction(db_credentials,
                     predict_config_path):

    predict_cfg = read_predict_config(predict_config_path)

    db = SynisterDB(db_credentials)
    db_name = "{}_predictions".format(predict_cfg["db_name_data"])
    collection_name = "{}_{}_t{}_p{}".format(predict_cfg["split_name"],
                                             predict_cfg["experiment"],
                                             predict_cfg["train_number"],
                                             predict_cfg["predict_number"])

    synapses = {}
    predictions = db.get_collection(db_name, collection_name)

    n = 0
    for prediction in predictions:
        print("Parse prediction {}/{}".format(n+1, len(predictions)))

        synapse = db.get_synapse(predict_cfg["db_name_data"],
                                 prediction["synapse_id"])

        synapse["prediction"] = prediction["prediction"]
        assert[synapse[predict_cfg["split_name"]] == "test"]
        synapses[synapse["synapse_id"]] = synapse

        n += 1

    return synapses, predict_cfg


def confusion_matrix(synapses, predict_config):
    synapse_types = predict_config["synapse_types"]
    confusion_matrix = np.zeros([len(synapse_types)] * 2, dtype=float)

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

        confusion_matrix[gt_class, predicted_class] += 1
        n += 1

    # normalize:
    #for gt_class in range(len(synapse_types)):
    #    confusion_matrix[gt_class, :]/= np.sum(confusion_matrix[gt_class, :])

    return confusion_matrix


def plot_confusion_matrix(cm, synapse_types):
    df_cm = pd.DataFrame(cm, index = [i for i in synapse_types],
                                columns = [i for i in synapse_types])
    plt.figure(figsize = (10,10))
    ax = sn.heatmap(df_cm, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+.5, top-.5)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def accuracy(confusion_matrix):                         #returns a tuple (overall accuracy, average accuracy)
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

    avg_accuracy = statistics.mean(accuracy)

    return (overall_accuracy, avg_accuracy)


if __name__ == "__main__":
    synapses, predict_cfg = parse_prediction("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
                                "/groups/funke/home/ecksteinn/Projects/synex/synister_experiments/fafb/03_predict/setup_t2_p0/predict_config.ini")
    # synapses, predict_cfg = parse_prediction("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
    #                             "/nrs/funke/ecksteinn/micron_experiments/test_experiments/02_predict/setup_t0_p0/predict_config.ini")

    confusion_matrix = confusion_matrix(synapses, predict_cfg)
    print(confusion_matrix)
    # plot_confusion_matrix(confusion_matrix, predict_cfg["synapse_types"])


    print(accuracy(confusion_matrix))
