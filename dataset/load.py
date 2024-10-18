
from dataset.dataset import AnimalDataset


def load_labels(labels_file: str) -> dict[
    str, int]:  # get name of CSV file, return pairs of image name and group id (a.k.a. label)
    res = dict()

    with open(labels_file, "r") as file:
        lines = file.readlines()  # read all file-label pairs

        for raw_label in lines[1:]:
            _, group_number, image_name = raw_label.split(",")
            res[image_name[:-1]] = int(group_number)  # cut out the newline symbol and add the label to the dictionary

        return res
