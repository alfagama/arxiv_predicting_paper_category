import pandas as pd

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
df = df.drop('Unnamed: 0', 1)

# print head
print(df.head(5))
# print(len(df))  # 75876

# A limitation of undersampling is that examples from the majority class are deleted that may be useful, important,
# or perhaps critical to fitting a robust decision boundary. Given that examples are deleted randomly, there is no
# way to detect or preserve “good” or more information-rich examples from the majority class.

# print categories count before oversampling
print(df['categories'].value_counts(ascending=True))

# create empty dataframe
random_undersampled_df = pd.DataFrame()

# create dfs based on category and append in list
df_csai_cscv_1108 = df[df['categories'] == 'cs.ai cs.cv'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csai_cscv_1108)
df_cssd_eessas_1245 = df[df['categories'] == 'cs.sd eess.as'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cssd_eessas_1245)
df_cslg_mathoc_statml_1286 = df[df['categories'] == 'cs.lg math.oc stat.ml'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cslg_mathoc_statml_1286)
df_cssy_mathoc_1368 = df[df['categories'] == 'cs.sy math.oc'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cssy_mathoc_1368)
df_cscv_csro_1470 = df[df['categories'] == 'cs.cv cs.ro'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscv_csro_1470)
df_csit_csni_mathit_1542 = df[df['categories'] == 'cs.it cs.ni math.it'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csit_csni_mathit_1542)
df_csit_eesssp_mathit_1811 = df[df['categories'] == 'cs.it eess.sp math.it'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csit_eesssp_mathit_1811)
df_csai_cscl_1861 = df[df['categories'] == 'cs.ai cs.cl'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csai_cscl_1861)
df_cscv_cslg_eessiv_1978 = df[df['categories'] == 'cs.cv cs.lg eess.iv'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscv_cslg_eessiv_1978)
df_cscl_cslg_2131 = df[df['categories'] == 'cs.cl cs.lg'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscl_cslg_2131)
df_csai_cslg_2207 = df[df['categories'] == 'cs.ai cs.lg'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csai_cslg_2207)
df_cscv_cslg_statml_2595 = df[df['categories'] == 'cs.cv cs.lg stat.ml'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscv_cslg_statml_2595)
df_csdm_mathco_2642 = df[df['categories'] == 'cs.dm math.co'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csdm_mathco_2642)
df_cssy_eesssy_2724 = df[df['categories'] == 'cs.sy eess.sy'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cssy_eesssy_2724)
df_csai_cslg_statml_2780 = df[df['categories'] == 'cs.ai cs.lg stat.ml'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csai_cslg_statml_2780)
df_cscv_eessiv_3072 = df[df['categories'] == 'cs.cv eess.iv'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscv_eessiv_3072)
df_cscv_cslg_3715 = df[df['categories'] == 'cs.cv cs.lg'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cscv_cslg_3715)
df_csna_mathna_4531 = df[df['categories'] == 'cs.na math.na'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csna_mathna_4531)
df_cslg_statml_16230 = df[df['categories'] == 'cs.lg stat.ml'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_cslg_statml_16230)
df_csit_mathit_19580 = df[df['categories'] == 'cs.it math.it'].sample(n=1000)
random_undersampled_df = random_undersampled_df.append(df_csit_mathit_19580)

# shuffle dataset
random_undersampled_df = random_undersampled_df.sample(frac=1).reset_index(drop=True)

# save to csv
random_undersampled_df.to_csv("../data/random_undersampled.csv")
