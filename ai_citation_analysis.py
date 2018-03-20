import pandas as pd
import numpy as np

def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
		  
#read citing docs file
dfCite = pd.read_excel('my_file_of_citing_docs.xlsx')
		  		  
# fill all null values in columns
dfCite = dfCite.fillna.('nope')

#Convert References column to list
df = df.assign(References=np.core.defchararray.split(df.References.values.astype(str), ';'))

#Run explode function to get each reference on its own line
explode(df, ['References'])

#write file to csv or excel
df.to_csv('exported_dataframe.csv', encoding = 'utf-8')
#writer = pd.ExcelWriter('citing_docs.xlsx', engine = 'xlsxwriter')
#df.to_excel(writer, sheet_name='Sheet1')
#writer.save()



#---scratch code
#convert References column to list of lists
dfCite = dfCite[['References']].tolist()

#to turn the references column into its own dataframe with title column as index
dfCite2 = pd.DataFrame(dfCite.References.str.split(';').tolist(), index = dfCite.Title).stack()

#alternate, with just references column
x = df['References'].map(lambda x: pd.Series(x.split(';'))).head()
