from synister.synister_db import SynisterDB
from synister.read_config import read_predict_config


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
    confusion_matrix = np.zeros([len(synapse_types)] * 2, dtype=int)

    n = 0
    for synapse in synapses:
        print("Insert synapse {}/{}".format(n + 1, len(synapses)))
        nt_known = synapse["nt_known"]
        if len(nt_known)>1:
            raise Warning("More than one known nt")
        nt_known = nt_known[0]

        gt_class = synapse_types.index(nt_known)
        predicted_class = np.argmax(synapse["prediction"])

        confusion_matrix[gt_class, predicted_class] += 1
        n += 1

    return confusion_matrix


if __name__ == "__main__":
    synapses = parse_prediction("/groups/funke/home/ecksteinn/Projects/synex/synister/db_credentials.ini",
                               "/groups/funke/home/ecksteinn/Projects/synex/synister_experiments/fafb/03_predict/setup_t0_p1/predict_config.ini")

    print(synapses)




    


    

    




