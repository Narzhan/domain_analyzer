import pandas

domains = {}
dataset = pandas.read_csv("test_data.csv", index_col=13)
for cat in list(dataset):
    domains[cat] = {"clean": 0, "malicious": 0}
columns = dataset.astype(bool).sum(axis=0)
for index, row in dataset.iterrows():
    if row["label"] == 1:
        label = "malicious"
    else:
        label = "clean"
    for cat in list(dataset):
        if row[cat] != 0:
            domains[cat][label] += 1
for k, v in domains.items():
    print(k)
    for x, y in v.items():
        print("     {}: {}".format(x, y))
print(dataset.loc[(dataset["full_path"] != 0) & (dataset["label"] == 1)])

