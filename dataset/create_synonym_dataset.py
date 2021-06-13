import pandas as pd
from Imbalanced.imbalance_generate_synonym_text import *
from preprocessing.create_preprocessed_csv import preprocess_generated

# Options for pandas -----
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# read pre-processed data set
df = pd.read_csv("../data/preprocessed_conc_dataset.csv",
                 sep=',',
                 header=0,
                 skiprows=0)

# drop unwanted columns
df = df.drop('abstract', 1)
df = df.drop('title', 1)

# print head
print(df.head(5))

# # separate features and target
# X = df.iloc[:, df.columns == "concatenation"].values
# y = df.iloc[:, df.columns == "categories"].values

# print categories count before oversampling
print(df['categories'].value_counts(ascending=True))

# create empty list
list = []
# create dfs based on category and append in list
df_csai_cscv_1108 = df[df['categories'] == 'cs.ai cs.cv']
list.append(df_csai_cscv_1108)
df_cssd_eessas_1245 = df[df['categories'] == 'cs.sd eess.as']
list.append(df_cssd_eessas_1245)
df_cslg_mathoc_statml_1286 = df[df['categories'] == 'cs.lg math.oc stat.ml']
list.append(df_cslg_mathoc_statml_1286)
df_cssy_mathoc_1368 = df[df['categories'] == 'cs.sy math.oc']
list.append(df_cssy_mathoc_1368)
df_cscv_csro_1470 = df[df['categories'] == 'cs.cv cs.ro']
list.append(df_cscv_csro_1470)
df_csit_csni_mathit_1542 = df[df['categories'] == 'cs.it cs.ni math.it']
list.append(df_csit_csni_mathit_1542)
df_csit_eesssp_mathit_1811 = df[df['categories'] == 'cs.it eess.sp math.it']
list.append(df_csit_eesssp_mathit_1811)
df_csai_cscl_1861 = df[df['categories'] == 'cs.ai cs.cl']
list.append(df_csai_cscl_1861)
df_cscv_cslg_eessiv_1978 = df[df['categories'] == 'cs.cv cs.lg eess.iv']
list.append(df_cscv_cslg_eessiv_1978)
df_cscl_cslg_2131 = df[df['categories'] == 'cs.cl cs.lg']
list.append(df_cscl_cslg_2131)
df_csai_cslg_2207 = df[df['categories'] == 'cs.ai cs.lg']
list.append(df_csai_cslg_2207)
df_cscv_cslg_statml_2595 = df[df['categories'] == 'cs.cv cs.lg stat.ml']
list.append(df_cscv_cslg_statml_2595)
df_csdm_mathco_2642 = df[df['categories'] == 'cs.dm math.co']
list.append(df_csdm_mathco_2642)
df_cssy_eesssy_2724 = df[df['categories'] == 'cs.sy eess.sy']
list.append(df_cssy_eesssy_2724)
df_csai_cslg_statml_2780 = df[df['categories'] == 'cs.ai cs.lg stat.ml']
list.append(df_csai_cslg_statml_2780)
df_cscv_eessiv_3072 = df[df['categories'] == 'cs.cv eess.iv']
list.append(df_cscv_eessiv_3072)
df_cscv_cslg_3715 = df[df['categories'] == 'cs.cv cs.lg']
list.append(df_cscv_cslg_3715)
df_csna_mathna_4531 = df[df['categories'] == 'cs.na math.na']
list.append(df_csna_mathna_4531)
df_cslg_statml_16230 = df[df['categories'] == 'cs.lg stat.ml']
# list.append(df_cslg_statml_16230) # do not add this to list -> no need to generate new rows
df_csit_mathit_19580 = df[df['categories'] == 'cs.it math.it']
# list.append(df_csit_mathit_19580) # do not add this to list -> no need to generate new rows

# # ...
# print(list)

# create Object from class TextRegenerator
trgnr = TextRegenerator()
# loop through list and read one dataframe at a time
for dataframe_of_list in list:
    # # generate synonyms for this dataframe category based on len(df_cslg_statml_16230) / len(dataframe_of_list)
    times_each_row_will_generate_a_synonym = round((len(df_csit_mathit_19580) / len(dataframe_of_list))/2)
    # generate synonyms for this dataframe category based on N number / len(dataframe_of_list)
    # times_each_row_will_generate_a_synonym = round(9000 / len(dataframe_of_list))
    # # generate synonyms for this dataframe category based N number
    # times_each_row_will_generate_a_synonym = 5
    # iterate through all dataframe's rows
    for row, text in dataframe_of_list.iterrows():
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(text['concatenation'])
        # get all possible synonym texts in a list
        outs = trgnr.generateStrVariations(text['concatenation'])
        # get only the top synonyms based on the number chosen earlier
        top_synonyms = outs[:times_each_row_will_generate_a_synonym]
        # loop through all synonyms and append in new row of dataframe
        for synonym_text in top_synonyms:
            synonym_text = synonym_text.replace('_', ' ')
            print("-------------------------------------------")
            print(synonym_text)
            new_row = {
                'concatenation': synonym_text,
                'categories': text['categories']
            }
            df = df.append(new_row, ignore_index=True)

    # print(round(times_each_row_will_generate_a_synonym))

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# drop Unamed col
df = df.drop('Unnamed: 0', 1)

# save new .csv with generated synonyms
df.to_csv('../data/sampling_generated_synonyms.csv')

# print categories count before oversampling
print(df['categories'].value_counts(ascending=True))

# preprocess new dataset and generate new .csv
preprocess_generated(df)
