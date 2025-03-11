import pandas as pd
import torch
import numpy as np
import os

print(torch.__version__)

cd = os.path.dirname(os.path.abspath(__file__))
file_name = cd+"\\data.xlsx"
fp = os.path.join(cd, file_name)

df = pd.read_excel(fp)
tensor = torch.tensor(df.values)