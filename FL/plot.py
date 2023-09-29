import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt


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
    tmp["Method"] = dirname

    df = pd.concat([df, tmp], ignore_index=True)

df.columns = list(map(lambda x: x.capitalize(), df.columns))
print(df)
sns.lineplot(x="Round", y="Accuracy", hue="Method", data=df)
plt.tight_layout()
plt.savefig("Accuracy.pdf")
plt.show()
sns.lineplot(x="Round", y="Loss", hue="Method", data=df)
plt.tight_layout()
plt.savefig("Loss.pdf")
plt.show()
sns.boxplot(x="Method", y="Accuracy", data=df[df.Round == 299])
plt.tight_layout()
plt.savefig("Boxplot.pdf")
plt.show()

