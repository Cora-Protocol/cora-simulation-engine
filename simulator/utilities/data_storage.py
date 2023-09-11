import csv
from pathlib import Path
from typing import List


class DataStorage:
    def __init__(self, data_folder: str = "data"):
        self.create_if_not_exists(data_folder)

    def create_if_not_exists(self, data_folder: str):
        root_path = Path(__file__).parent.parent.parent
        data_path = root_path / data_folder
        data_path.mkdir(parents=True, exist_ok=True)
        self.data_folder = data_path

    def get_file(self, file_name: str):
        file = self.data_folder / file_name
        if not file.exists():
            return None

        with open(file, "r") as f:
            return f.read()

    def save_file(self, file_name: str, content: str):
        file = self.data_folder / file_name
        with open(file, "a") as f:
            f.write(content)

    def get_csv(self, file_name: str):
        file = (self.data_folder / file_name).with_suffix(".csv")
        if not file.exists():
            return None

        with open(file, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def write_csv(self, file_name: str, content: List[dict]):
        file = (self.data_folder / file_name).with_suffix(".csv")
        with open(file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=content[0].keys())
            writer.writeheader()
            writer.writerows(content)

    def append_csv(self, file_name: str, content: List[dict]):
        file = (self.data_folder / file_name).with_suffix(".csv")
        with open(file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=content[0].keys())
            writer.writerows(content)

    def write_or_append_csv(self, file_name: str, content: List[dict]):
        file = (self.data_folder / file_name).with_suffix(".csv")
        if not file.exists():
            self.write_csv(file_name, content)
        else:
            self.append_csv(file_name, content)
