import json

data = {
    "onehot": {
        "dnnre": {
            "train": {},
            "test": {}
        },
        "alpa": {
            "train": {},
            "test": {}
        }
    },
    "woe": {
        "dnnre": {
            "train": {},
            "test": {}
        },
        "alpa": {
            "train": {},
            "test": {}
        }
    }
}

with open("eval/onehot/dnnre/test_eval.json", "r") as file:
    data["onehot"]["dnnre"]["test"] = json.load(file)
with open("eval/onehot/dnnre/train_eval.json", "r") as file:
    data["onehot"]["dnnre"]["train"] = json.load(file)
with open("eval/onehot/alpa/test_eval.json", "r") as file:
    data["onehot"]["alpa"]["test"] = json.load(file)
with open("eval/onehot/alpa/train_eval.json", "r") as file:
    data["onehot"]["alpa"]["train"] = json.load(file)
with open("eval/woe/dnnre/test_eval.json", "r") as file:
    data["woe"]["dnnre"]["test"] = json.load(file)
with open("eval/woe/dnnre/train_eval.json", "r") as file:
    data["woe"]["dnnre"]["train"] = json.load(file)
with open("eval/woe/alpa/test_eval.json", "r") as file:
    data["woe"]["alpa"]["test"] = json.load(file)
with open("eval/woe/alpa/train_eval.json", "r") as file:
    data["woe"]["alpa"]["train"] = json.load(file)

for cat in ["onehot", "woe"]:
    for algo in ["dnnre", "alpa"]:
        print(f"{algo} with {cat}: test data (training data)")
        print(
            f"\tnn accuracy {data:[cat][algo]['test']['nn_acc']:.2}"
            f" ({data[cat][algo]['train']['nn_acc']:.2})")
        print(
            f"\trule accuracy {data[cat][algo]['test']['rules_acc']:.2}"
            f" ({data[cat][algo]['train']['rules_acc']:.2})")
        print(
            f"\tfidelity {data[cat][algo]['test']['rules_fid']:.2}"
            f" ({data[cat][algo]['train']['rules_fid']:.2})")
