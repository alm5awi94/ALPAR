import json

data = {
    "dnnre": {
        "train": {},
        "test": {}
    },
    "alpa": {
        "train": {},
        "test": {}
    }
}

with open("eval/dnnre/test_eval.json", "r") as file:
    data["dnnre"]["test"] = json.load(file)
with open("eval/dnnre/train_eval.json", "r") as file:
    data["dnnre"]["train"] = json.load(file)
with open("eval/alpa/test_eval.json", "r") as file:
    data["alpa"]["test"] = json.load(file)
with open("eval/alpa/train_eval.json", "r") as file:
    data["alpa"]["train"] = json.load(file)

for algo in ["dnnre", "alpa"]:
    print(f"{algo}: test data (training data)")
    print(
        f"\tnn accuracy {data[algo]['test']['nn_acc']}"
        f" ({data[algo]['train']['nn_acc']})")
    print(
        f"\trule accuracy {data[algo]['test']['rules_acc']}"
        f" ({data[algo]['train']['rules_acc']})")
    print(
        f"\tfidelity {data[algo]['test']['rules_fid']}"
        f" ({data[algo]['train']['rules_fid']})")
