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

output = ""

for algo in ["dnnre", "alpa"]:
    output += f"{algo}: test data (training data)\n"
    output += f"\tnn accuracy {data[algo]['test']['nn_acc']:.2} " \
              f"({data[algo]['train']['nn_acc']:.2})\n"
    output += f"\trule accuracy {data[algo]['test']['rules_acc']:.2}" \
              f" ({data[algo]['train']['rules_acc']:.2})\n"
    output += f"\tfidelity {data[algo]['test']['rules_fid']:.2}" \
              f" ({data[algo]['train']['rules_fid']:.2})\n"
    output += f"\tnrules {data[algo]['test']['n_rules']}\n"

print(output)
with open("results.txt", "w+") as file:
    file.write(output)
