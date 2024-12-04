import pickle
import os


class Testing_data(object):
    def __init__(self, LTL, workspace, task):
        self.LTL = LTL
        self.workspace = workspace
        self.task = task


def save_testing_data(testing_data, path, filename):
    filename_list = os.listdir(path)
    filename_list = str(filename_list)
    num_of_files = filename_list.count(filename)

    filename = path + filename + str(num_of_files + 1) + ".pkl"
    file = open(filename, "wb")
    pickle.dump(testing_data, file)
    file.close()
    print("save testing data " + filename + " successfully")


def load_testing_data(path, filename):
    filename = path + filename + ".pkl"
    file = open(filename, "rb")
    testing_data = pickle.load(file)
    file.close()
    print("load testing data " + filename + " successfully")
    return testing_data
