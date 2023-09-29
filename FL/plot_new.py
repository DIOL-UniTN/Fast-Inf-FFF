import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt


INPUT_SIZE = 768
OUTPUT_SIZE = 10


df = pd.DataFrame()


dirs = os.listdir()
dirs = filter(lambda x: "ff" in x and os.path.isdir(x), dirs)

for dirname in dirs:
    files = os.listdir(dirname)
    dicts = []
    for f in tqdm(files):
        with open(os.path.join(dirname, f)) as file:
            for i, line in enumerate(file):
                tmpdict = eval(line)
                tmpdict["Round"] = i
                dicts.append(tmpdict)
    tmp = pd.DataFrame.from_dict(dicts)
    tmp = tmp[tmp.Round == 299]
    tmp["Method"] = dirname
    if dirname == "ff":
        tmp["Mac"] = INPUT_SIZE * (16 * (2 ** 3)) * OUTPUT_SIZE
    else:
        l = dirname.split("_")[1:]
        d = int(l[-1][1:])
        l = int(l[0][1:])
        tmp["Mac"] = INPUT_SIZE * (d - 1) + INPUT_SIZE * l * OUTPUT_SIZE
    df = pd.concat([df, tmp], ignore_index=True)

df.columns = list(map(lambda x: x.capitalize(), df.columns))
df = df.groupby("Method").mean()
print(df)
sns.scatterplot(x="Mac", y="Accuracy", hue="Method", data=df)
plt.show()
