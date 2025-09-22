import datetime
import json


def store_dict_as_json(dictdata: dict, filename: str):
    with open(filename, "w") as json_file:
        json.dump(dictdata, json_file, indent=4)

def load_json(filename: str) -> dict:
    data = None
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def seconds_to_hours_mins(period_secs: int) -> int:
    hours = int(period_secs // 3600)  # Integer division for whole hours
    minutes = int((period_secs % 3600) // 60)  # Remainder after hours, then divide by 60 for minutes
    return hours, minutes