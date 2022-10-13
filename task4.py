#!/usr/bin/env python
# coding: utf-8

# 1. Loading Data:

# In[1]:


#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('Downloads/Global Terrorism - START data/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df.head()


# 2. Familiarizing with Data:
# 

# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[8]:


# Important data for further processing

df=df[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[9]:


df.head()


# In[10]:


#checking for null value

df.isna().sum()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


#unique value in dataset

df.nunique()    


# In[15]:


#checking for outlier

col=[ "Year","Month","Day","latitude","longitude",  "Killed" ,  "Wounded"  ]
new_data=df
for i in col:
    new_data=new_data.sort_values(by=[i])
    q1=new_data[i].quantile(0.25)
    q3=new_data[i].quantile(0.75)
    iqr=q3-q1
    lwo=q1-1.5*iqr
    upo=q3+1.5*iqr
    new_data=new_data[(new_data[i]<upo) & (new_data[i]>lwo)]
    new_data=new_data.sort_index().reset_index(drop=True)
    
    
if(new_data.size<df.size):
    print("There exist outlier in dataset.")
    print("Shape of dataset before removing Outlier : ",new_data.shape)
    print("Shape of dataset after removing Outlier : ",df.shape)


# In[16]:


df['Year'].value_counts(dropna = False).sort_index()


# 3. Visualizing the data:

# In[17]:


#histogram for Visualizing

df.hist(bins = 50,figsize = (15,11));


# In[18]:


#correlation among dataset

df.corr().abs()


# In[19]:


#Correlation heatmap

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True)
plt.show()


# Terrorist Activities Each Year
# 

# In[20]:


x_year = df['Year'].unique()
y_count_years = df['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = x_year, y = y_count_years, palette = 'rocket')
plt.xticks(rotation = 50)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks Each Year')
plt.title('Attack In Years')
plt.show()


# In[21]:


plt.subplots(figsize=(20,10))
sns.countplot('Year',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=50)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# Terrorist Activities By Region In Each Year
# 

# In[23]:


pd.crosstab(df.Year, df.Region).plot(kind='area',figsize=(20,10))
plt.title('Terrorist Activities By Region In Each Year')
plt.ylabel('Number of Attacks')
plt.show()


# In[24]:


df['Wounded'] = df['Wounded'].fillna(0).astype(int)
df['Killed'] = df['Killed'].fillna(0).astype(int)
df['Casualities'] = df['Killed'] + df['Wounded']


# In[25]:


# Top 50 worst terrorist attacks
df1 = df.sort_values(by='Casualities',ascending=False).reset_index()
df1.head()


# In[26]:


df1.shape


# In[27]:


heat=df1.pivot_table(index='Country',columns='Year',values='Casualities')
heat.fillna(0,inplace=True)
heat.head()


# In[28]:


heat.shape


# Top Countries Affected By Terrorist Attacks
# 

# In[29]:


plt.subplots(figsize=(20,10))
sns.barplot(df['Country'].value_counts()[:20].index,df['Country'].value_counts()[:20].values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.show()


# 4. Data Analysis:

# Terrorist Attacks of a Particular year and their Locations
# 

# In[30]:


filterYear = df[df['Year'] == 2001].reset_index(drop=True)
filterYear


# In[32]:


filterYear.info()


# In[33]:


reqFilterData = filterYear.loc[:,'city':'longitude'] # get the required fields
reqFilterData = reqFilterData.dropna() # drop NaN values in latitude and longitude
reqFilterDataList = reqFilterData.values.tolist()
reqFilterDataList[:20]


# Terrorist's Origanizations Operations In Each Country
# 

# In[34]:


df.Group.value_counts()


# In[35]:


test = df[df.Group.isin(['Shining Path (SL)','Taliban','Islamic State of Iraq and the Levant (ISIL)'])]


# In[36]:


test.Country.unique()


# In[37]:


df_df_group = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)
df_df_group.head()


# In[38]:


df_df_group = df_df_group.drop_duplicates(subset=['Country','Group']).reset_index(drop=True)
df_df_group


# In[39]:


terrorist = df.Group.value_counts()[1:8].index.tolist()
df_df_group = df_df_group.loc[df_df_group.Group.isin(terrorist)]
print(df_df_group.Group.unique())
terrorist


# In[40]:


df.head()


# In[41]:


# Total Number of people killed in terror attack
killData = df.loc[:,'Killed']
print('Number of people killed by terror attack:', int(sum(killData.dropna())))# drop the NaN values


# In[42]:


# Let's look at what types of attacks these deaths were made of.
attackData = df.loc[:,'AttackType']
# attackData
typeKillData = pd.concat([attackData, killData], axis=1)


# In[43]:


typeKillData.head()


# In[44]:


typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData


# In[45]:


typeKillFormatData.info()


# In[46]:


# Number of Killed in Terrorist Attacks by Countries
countryData = df.loc[:,'Country']
# countyData
countryKillData = pd.concat([countryData, killData], axis=1)
countryKillData.info()


# In[47]:


countryKillFormatData = countryKillData.pivot_table(columns='Country', values='Killed', aggfunc='sum')
countryKillFormatData


# In[48]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size


# In[49]:


labels = countryKillFormatData.columns.tolist()
labels = labels[:50]  #50 bar provides nice view
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[:50]
values = [int(i[0]) for i in values] # convert float to int
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange'] # color list for bar chart bar color 
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=30)
plt.xlabel('Countries', fontsize = 30)
plt.xticks(index, labels, fontsize=20, rotation=90)
plt.title('Number of People Killed By Countries', fontsize = 30)
plt.show()


# In[50]:


labels = countryKillFormatData.columns.tolist()
labels = labels[50:101]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[50:101]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=20
fig_size[1]=20
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=20, rotation=90)
plt.title('Number Of People Killed By Countries', fontsize = 30)
plt.show()


# In[51]:


labels = countryKillFormatData.columns.tolist()
labels = labels[152:206]
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[152:206]
values = [int(i[0]) for i in values]
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=20, rotation=90)
plt.title('Number Of people Killed By Countries', fontsize = 20)
plt.show()


# In[52]:


df['Group'].value_counts().drop('Unknown').head(20).plot(kind='pie',autopct='%1.1f%%',figsize=(40,40),startangle=180,fontsize='50')
plt.title("Groupname attacks in total",fontsize=70)
plt.show()


# In[53]:


df['AttackType'].value_counts().plot(kind='pie',figsize=(20,20),autopct='%1.1f%%',fontsize='30')
plt.title("Attacktype attacks in total",fontsize=40)
plt.show()


# In[54]:


df = df[['Country','Group','Killed']].groupby(['Group','Country']).sum().sort_values('Killed',ascending=False).drop('Unknown').head(20)
df


# In[ ]:




