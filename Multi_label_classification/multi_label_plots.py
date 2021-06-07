import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("../data/category_columns_dataset.csv",
                     sep=',',
                     header=0,
                     skiprows=0)
    # df = df.sample(n=1000, replace=True, random_state=42)
    print(df.head(5))

    print(df['categories'].value_counts())

    plot_df = df['categories'].value_counts()
    print(plot_df.head(2))

    print(type(plot_df))

    list_X = plot_df.tolist()

    print(list_X)

    list_Y = ['cs.it math.it',
              'cs.lg stat.ml',
              'cs.na math.na',
              'cs.cv cs.lg',
              'cs.cv eess.iv',
              'cs.ai cs.lg stat.ml',
              'cs.sy eess.sy',
              'cs.dm math.co',
              'cs.cv cs.lg stat.ml',
              'cs.ai cs.lg',
              'cs.cl cs.lg',
              'cs.cv cs.lg eess.iv',
              'cs.ai cs.cl',
              'cs.it eess.sp math.it',
              'cs.it cs.ni math.it',
              'cs.cv cs.ro',
              'cs.sy math.oc',
              'cs.lg math.oc stat.ml',
              'cs.sd eess.as',
              'cs.ai cs.cv']

    # x = plot_df['category']
    # y = plot_df['value']

    # print(x)
    # print(y)

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.bar(list_Y, list_X)
    # plt.show()

    plt.bar(list_Y,list_X, align='center', alpha=0.5)
    # plt.xticks(X.columns, features)
    plt.xlabel('Categories')
    plt.xticks(rotation=90)
    plt.ylabel('Number of papers')
    plt.title('Dataset')
    plt.gcf().subplots_adjust(bottom=0.25)
    # plt.savefig("Figs/Built-in Feature Importance.png")
    # print(model.feature_importances_)
    # for i, v in enumerate(list_X):
    #     print(i)
    #     print(v)
    #     plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    for index, value in enumerate(list_Y):
        print(index)
        print(value)
        int = list_X[index]
        print(int)
        plt.text(value, int, str(int), rotation=90)
    plt.show()
    # plt.close()


    df_classes = df['categories'].value_counts().rename_axis('Categories').reset_index(name='Row count')
    print(df_classes)
    ax = df_classes.plot.bar(x='Categories', y='Row count', rot=0, title='Categories distribution')
    ax.figure.savefig('categories_distribution.png')
