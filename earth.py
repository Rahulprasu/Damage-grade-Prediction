import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('train_values.csv')

df1=pd.read_csv('train_labels.csv')

df1=df1.drop('building_id',axis=1)

df=pd.concat([df,df1],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df[['land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration','legal_ownership_status']]=df[['land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration','legal_ownership_status']].apply(le.fit_transform)

df=df.drop(['count_families','has_superstructure_cement_mortar_brick','position','other_floor_type','ground_floor_type','has_superstructure_mud_mortar_stone','plan_configuration',
'has_superstructure_adobe_mud',
'legal_ownership_status',
'has_superstructure_mud_mortar_brick',
'has_superstructure_rc_non_engineered',
'has_superstructure_stone_flag',
'has_superstructure_timber',
'has_secondary_use',
'has_superstructure_other',
'has_superstructure_bamboo',
'has_superstructure_cement_mortar_stone',
'has_secondary_use_agriculture',
'has_superstructure_rc_engineered',
'has_secondary_use_other',
'has_secondary_use_industry',
'has_secondary_use_institution',
'has_secondary_use_school',
'has_secondary_use_gov_office',
'has_secondary_use_health_post',
'has_secondary_use_use_police','building_id','geo_level_1_id','geo_level_2_id','geo_level_3_id'],axis=1)




p1=df.age.quantile(0.25)
p3=df.age.quantile(0.99)
iqr=p3-p1
ll=p1-1.5*iqr
ul=p3+1.5*iqr
df1=df[df.age<ul]

p1=df1.area_percentage.quantile(0.25)
p3=df1.area_percentage.quantile(0.94)
iqr=p3-p1
ul=p3+1.5*iqr
df2=df1[df1.area_percentage<ul]

x=df2.drop(['damage_grade'],axis=1)
y=df2.damage_grade

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='gini',max_depth=8)
dt.fit(xtrain,ytrain)

filename='earthproject.pkl'
pickle.dump(dt,open(filename,'wb'))
