from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


def create_wordcloud():
    df = pd.read_csv("../data/preprocessed_conc_dataset.csv",
                     # df = pd.read_csv("../data/random_undersampled_20.csv",
                     sep=',',
                     header=0,
                     skiprows=0)

    text_list = [row for row in df['concatenation']]

    # Convert list to string
    long_string = ''.join(text_list)

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800, height=400)

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Save the word cloud
    wordcloud.to_file("../output/wordcloud2.png")

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.figure(figsize=(20, 10))
    plt.show()


def plot_categories_percentage():

    df = pd.read_csv("../data/category_columns_dataset.csv",
                     # df = pd.read_csv("../data/random_undersampled_20.csv",
                     sep=',',
                     header=0,
                     skiprows=0)

    df = df.drop(labels=['categories',
                       'concatenation',
                       'title',
                       'abstract',
                       'Unnamed: 0',
                       'Unnamed: 0.1'
                       ],
                 axis=1)

    df = pd.DataFrame(df.iloc[:, :].apply(sum)).reset_index().rename(columns={0: 'count'})

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(df['index'], df['count'], color='maroon', width=0.4)
    plt.xticks(rotation='vertical')
    plt.xlabel("Categories")
    plt.ylabel("No. of Papers")
    plt.title("Nο. of papers by category")
    plt.tight_layout()
    plt.savefig("../output/categories.png")
    plt.show()


if __name__ == '__main__':
    #create_wordcloud()
    plot_categories_percentage()


