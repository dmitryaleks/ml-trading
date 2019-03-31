import pandas as pd

df = pd.DataFrame(
    [['Jan',1,2,3,4,5],
     ['Feb',11,12,13,14,15],
     ['Mar',21,22,23,24,25]],
    index = [0,1,2],
    columns = ['months', 'priceA', 'priceB', 'priceC', 'priceD', 'priceE'])

print(df.tail())

print(df.dtypes)

print(df.index)

print(df.columns)

print(df.describe())

print(df.sort_values('priceC', ascending=False))

print(df.priceA)

print(df[2:3])

# Slice first and second rows and print selected columns
print(df.loc[1:3,['priceA', 'priceC']])

# Add a sum column
df['sum'] = (df.priceA + df.priceB + df.priceC + df.priceD + df.priceE)
print(df)

# Access index and rows
for index, row in df.iterrows():
    print (index, row['sum'])

df.to_csv('data.csv')

dfA = pd.read_csv('data.csv')
print(df)
