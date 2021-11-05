import os
import pandas as pd
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)  # 可以看到原始表格中的空值NA被识别成了NaN
print('1.原始数据:\n', data)

# 通过位置索引iloc，我们将data分成inputs和outputs，
# 其中前者为data的前两列，而后者为data的最后一列。对于inputs中缺少的数值，用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:-1], data.iloc[:, -1]
inputs = inputs.fillna(inputs.mean())
print(inputs)
#print(outputs)