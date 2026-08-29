import json
import random
def load_weibo_data(file_paths):
    res = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = line.strip()
                if line_data:
                    obj = json.loads(line_data)
                    res.append(obj)
    return res  

def load_AMTCele_data(file_paths):
    res = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = line.strip()
                if line_data:
                    obj = json.loads(line_data)
                    res.append(obj)
    return res

class Dataset_Weibo:
    def __init__(self, file_path):
        items = load_weibo_data(file_path)
        self.data = self.load_data(items)

    def load_data(self, items):
        res = {
            "科技": [],
            "军事": [],
            "教育考试": [],
            "灾难事故": [],
            "政治": [],
            "医药健康": [],
            "财经商业": [],
            "文体娱乐": [],
            "社会生活": [],
        }

        for item in items:
            if item["category"] in res:
                res[item["category"]].append(item)

        return res

    def verify_data(self):
        for category, items in self.data.items():
            print(f"Category: {category}, Number of items: {len(items)}")

    def shuffle_data(self, categorys):
        for category in categorys:
            if category in self.data:
                random.shuffle(self.data[category])
            else:
                print(f"Category '{category}' not found in the dataset.")
    

class Dataset_AMTCele:
    def __init__(self, file_path):
        items = load_AMTCele_data(file_path)
        self.data = self.load_data(items)

    def load_data(self, items):
        res = {
            "biz": [],
            "celebrity": [],
            "edu": [],
            "entmt": [],
            "polit": [],
            "sports": [],
            "tech": [],
        }

        for item in items:
            category = item["domain"].rstrip("0123456789")
            if category in res:
                res[category].append(item)

        return res

    def verify_data(self):
        for category, items in self.data.items():
            print(f"Category: {category}, Number of items: {len(items)}")

    def shuffle_data(self, categorys):
        for category in categorys:
            if category in self.data:
                random.shuffle(self.data[category])
            else:
                print(f"Category '{category}' not found in the dataset.")


if __name__ == "__main__":
    file_path = ["./datasets/AMTCele/AMTCele.jsonl"]
    dataset = Dataset_AMTCele(file_path)
    dataset.verify_data()

