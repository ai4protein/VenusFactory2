import json


def get_uid_from_rcsb_meata_data(meta_data_file):
    """
    get uniprot ids from rcsb meta data file
    """
    with open(meta_data_file, "r") as f:
        data = json.load(f)
    uniprot_id = data["data"]["entry"]["polymer_entities"]["uniprots"]["rcsb_id"]
    return uniprot_id