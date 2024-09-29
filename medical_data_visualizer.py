import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = 0
imc = df['weight']/(0.01*df['height'])**2
df.loc[imc > 25, 'overweight'] = 1

# 3
df['cholesterol'] = df['cholesterol'].replace(1, 0).replace([2, 3], 1)
df['gluc'] = df['gluc'].replace(1, 0).replace([2, 3], 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['cardio'], value_vars=[
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'
    ])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', 
                      data=df_cat, kind='bar', height=5, aspect=1.2)

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) & 
                  (df['height'] <= df['height'].quantile(0.975)) & 
                  (df['weight'] >= df['weight'].quantile(0.025)) & 
                  (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax, cmap='coolwarm', 
                vmin=-1, vmax=1, cbar_kws={"shrink": .75})

    # 16
    fig.savefig('heatmap.png')
    return fig
