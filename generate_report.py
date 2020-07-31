import json
from zipfile import ZipFile
import sys
import os
import numpy as np
import pandas as pd 
from collections import Counter
from shutil import copyfile
import subprocess

def transform(data, skid, nts):
    table_data = []
    for synapse in data:
        table_row = [skid]
        table_columns = ["skid"]
        table_row.append(synapse[0]) # Connector ID
        table_columns.append("Connector")
        for pos in synapse[1]: # position
            table_row.append(pos)
        table_columns.append("z")
        table_columns.append("y")
        table_columns.append("x")

        for nt in nts:
            table_row.append(synapse[2][nt])
            table_columns.append(nt)
        table_data.append(table_row)

    df = pd.DataFrame(table_data, columns=table_columns)
    df_pred = df[[nt for nt in nts]]
    df['max_prediction'] = df_pred.max(axis=1)
    df['max_nt'] = df_pred.idxmax(axis=1)
    df_level_0 = df
    df_level_0.set_index("skid", inplace=True)

    counts = Counter(list(df["max_nt"]))
    winner = counts.most_common(1)[0]

    counts_tmp = {}
    for nt in nts:
        if nt in counts.keys():
            counts_tmp[nt] = counts[nt]
        else:
            counts_tmp[nt] = 0
    counts = counts_tmp
    counts["skid"] = skid
    
    df_level_1 = pd.DataFrame(counts, index=[0]) # use from_dict instead
    df_level_1.set_index("skid", inplace=True)

    df_level_2 = pd.DataFrame([[skid, winner[0], int(winner[1]), int(len(df_level_0.index))]],
                              columns=["skid", "majorityNT", "synapsesNT", "synapsesSkid"])
    df_level_2.set_index("skid", inplace=True)

    return df_level_0, df_level_1, df_level_2

def write(level_0, level_1, level_2, output_dir, skid):
    level_0.to_csv(output_dir + "/{}_l0.csv".format(skid), index=True)
    level_1.to_csv(output_dir + "/{}_l1.csv".format(skid), index=True)
    level_2.to_csv(output_dir + "/{}_l2.csv".format(skid), index=True)

def read(predict_json):
    data = json.load(open(predict_json, "r"))
    return data

def summarize(skid_list, directory, no_synapses):
    l0_paths = [directory + "/{}_{}".format(skid, "l0.csv") for skid in skid_list]
    l1_paths = [directory + "/{}_{}".format(skid, "l1.csv") for skid in skid_list]
    l2_paths = [directory + "/{}_{}".format(skid, "l2.csv") for skid in skid_list]

    dfs_l0 = [pd.read_csv(l0, index_col="skid") for l0 in l0_paths]
    df_l0_combined = pd.concat(dfs_l0, ignore_index=False)
    df_l0_combined.to_csv(directory + "/l0.csv", index=True)

    dfs_l1 = [pd.read_csv(l1, index_col="skid") for l1 in l1_paths]
    df_l1_combined = pd.concat(dfs_l1, ignore_index=False)
    df_l1_combined.to_csv(directory + "/l1.csv", index=True)

    dfs_l2 = [pd.read_csv(l2, index_col="skid") for l2 in l2_paths]
    df_l2_combined = pd.concat(dfs_l2, ignore_index=False)
    df_l2_combined.to_csv(directory + "/l2.csv", index=True)

    for skid in no_synapses:
        l0_append = "{}".format(skid) + ",NA" * 12 
        l1_append = "{},0,0,0,0,0,0".format(skid)
        l2_append = "{},NA,0,0".format(skid)
        with open(directory + "/l0.csv", 'a') as fd:
            fd.write(l0_append)
        with open(directory + "/l1.csv", 'a') as fd:
            fd.write(l1_append)
        with open(directory + "/l2.csv", 'a') as fd:
            fd.write(l2_append)

def process(predict_json, skid, nt_list):
    data = read(predict_json)
    output_dir = os.path.dirname(predict_json)
    l0, l1, l2 = transform(data, skid, nt_list)
    write(l0, l1, l2, output_dir, skid)

def read_neuron_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pd.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

def prepare_archive(path_to_csv):
    base_dir = os.path.abspath(os.path.dirname(path_to_csv))
    levels = [0,1,2]
    level_files = [os.path.join(base_dir, "predictions/l{}.csv".format(i)) for i in levels]
    archive_base = os.path.join(base_dir, os.path.basename(base_dir).split("_")[-1].lower())
    archive = ZipFile(archive_base + '_nt_predictions.zip', 'w')
    # Add multiple files to the zip

    for level_file, level in zip(level_files, levels):
        archive.write(level_file, "level_{}.csv".format(level))
        #print(level_file)

    #print(gen_report(path_to_csv))
    archive.write(gen_report(path_to_csv), "report.pdf")
    # close the Zip File
    archive.close()

def gen_report(path_to_csv):
    base_dir = os.path.abspath(os.path.dirname(path_to_csv))
    skids = read_neuron_csv(path_to_csv)
    levels = [0,1,2]
    level_files = [os.path.join(base_dir, "predictions/l{}.csv".format(i)) for i in levels]

    # copy l1 to report dir:
    copyfile(level_files[1], 
            "/groups/funke/home/ecksteinn/Projects/synex/synister_report/l1.csv")

    # Read in the file
    bar_template = '/groups/funke/home/ecksteinn/Projects/synex/synister_report/bar_template.tikz.tex' 
    bar_dest = '/groups/funke/home/ecksteinn/Projects/synex/synister_report/bar.tikz.tex' 
 
    with open(bar_template, 'r') as f:
      filedata = f.read()

    # Replace the target string
    filedata = filedata.replace('replace', '\\foreach \\skididx in {{1,...,{}}}{{%'.format(len(skids)))

    # Write the file out again
    with open(bar_dest, 'w') as f:
      f.write(filedata)

    build_ret = subprocess.check_call("cd /groups/funke/home/ecksteinn/Projects/synex/synister_report; pdflatex main.tex",
                                      shell = True)

    """
    build_ret = subprocess.check_call("pdflatex /groups/funke/home/ecksteinn/Projects/synex/synister_report/main.tex",
                                      shell = True)
    """
    copyfile("/groups/funke/home/ecksteinn/Projects/synex/synister_report/main.pdf", 
            base_dir + "/predictions/report.pdf")

    return base_dir + "/predictions/report.pdf" 

if __name__ == "__main__":
    skids_csv = sys.argv[1]
    #skids_csv ="/groups/funke/home/ecksteinn/Projects/synex/synister_predictions/requests/r_130620_Istvan/skids.csv" 
    base_dir = os.path.dirname(skids_csv) + "/predictions"
    model_config = json.load(open(base_dir + "/model_config.json","r"))
    nt_list = model_config["neurotransmitter_list"]

    print("Generate statistics for {}".format(skids_csv))
    skids = read_neuron_csv(skids_csv) 
    processed = []
    missing = []
    no_synapses = []
    for skid in skids:
        #try:
        predict_json = base_dir + "/skid_{}.json".format(skid)
        if os.path.exists(predict_json):
            data = read(predict_json)
            if not data:
                no_synapses.append(skid)
                continue
            process(predict_json, skid, nt_list)
            processed.append(skid)
        else:
            missing.append(skid)

    summarize(processed, base_dir, no_synapses)
    print("Processed {} skids, {} are missing".format(len(processed), len(missing)))
    print("{} had no synapses".format(no_synapses))
    print("Missing:", missing)

    print("Prepare archive...")
    prepare_archive(skids_csv)
    gen_report(skids_csv)
