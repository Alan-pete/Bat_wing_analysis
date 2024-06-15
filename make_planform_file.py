import numpy as np
import json

def make_planform(input_file, output_file, LE_data_name="LE", TE_data_name="TE"):

    # read in the json file
    json_string = open(input_file).read()
    input_dict = json.loads(json_string)

    # get points from the file
    data_coll = input_dict["datasetColl"]
    for i in range(len(data_coll)):
        if data_coll[i]["name"] == LE_data_name:
            le_index = i
        if data_coll[i]["name"] == TE_data_name:
            te_index = i

    # pull out points dictionaries
    LE_data = data_coll[le_index]["data"]
    TE_data = data_coll[te_index]["data"]
    LE = np.array([LE_data[j]["value"] for j in range(len(LE_data))])
    TE = np.array([TE_data[j]["value"] for j in range(len(TE_data))])

    # force 0 and 0.5
    LE[0, 0] = TE[0, 0] = 0.0
    LE[-1, 0] = TE[-1, 0] = 0.5

    # collate and sort z/b values from LE and TE
    zbs = np.unique(np.concatenate((LE[:, 0], TE[:, 0])))

    # determine leading and trailing edge values
    les = np.interp(zbs, LE[:, 0], LE[:, 1])
    tes = np.interp(zbs, TE[:, 0], TE[:, 1])

    # calculate chord
    cbs = les - tes

    # write to file
    with open(output_file, "w") as f:
        f.write("z/b\tc/b\n")

        # run through and write z/b and c/b to file
        for i in range(len(zbs)):
            f.write("{:<14.12f}\t{:<14.12f}\n".format(zbs[i], cbs[i]))

        f.close()

    return 0


if __name__ == "__main__":
    make_planform("bat_plotdigitizer.json", "bat_planform.txt")