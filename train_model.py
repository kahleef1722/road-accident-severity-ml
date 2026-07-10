import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
df = pd.read_csv("RTA Dataset.csv")

df['Educational_level']= df['Educational_level'].map({
    'Illiterate':0,
    'Elementary school':1,
    'Junior high school':2,
    'High school':3,
    'Above high school':4})

df['Educational_level'] = df['Educational_level'].fillna(df['Educational_level'].mode()[0])



df['Sex_of_driver'].unique()
df['Sex_of_driver']=df['Sex_of_driver'].map({'Male':0,'Female':1,})
df['Sex_of_driver'] = df['Sex_of_driver'].fillna(df['Sex_of_driver'].mode()[0])



df['Age_band_of_driver'].unique()
df['Age_band_of_driver']= df['Age_band_of_driver'].map(
    {
    'Under 18':0,
    '18-30':1,
    '31-50':2,
    'Over 51':3,

}
)
df['Age_band_of_driver'] = df['Age_band_of_driver'].fillna(df['Age_band_of_driver'].mode()[0])

# Map Driving_experience to numerical values
df['Driving_experience']=df['Driving_experience'].map({
    'Below 1yr':0,
    '1-2yr':1,
    '2-5yr':2,
    '5-10yr':3,
    'Above 10yr':4
})
df['Driving_experience'] = df['Driving_experience'].fillna(df['Driving_experience'].mode()[0])



df['Type_of_vehicle']=df['Type_of_vehicle'].map({
    'Bicycle':0,
    'Special vehicle':1,
    'Motorcycle':2,
    'Turbo':3,
    'Automobile':4,
    'Public (> 45 seats)':5,
    'Lorry (41?100Q)':6,
    'Public (13?45 seats)':7,
    'Lorry (11?40Q)':8,
    'Long lorry':9,
   'Public (12 seats)':10,
    'Taxi':11,
    'Pick up upto 10Q':12,
    'Stationwagen':13,
    'Ridden horse':14,
    })
df['Type_of_vehicle'] = df['Type_of_vehicle'].fillna(df['Type_of_vehicle'].mode()[0])
#get dummies#

df['Area_accident_occured']=df['Area_accident_occured'].map({
    'Residential areas':0,
    'Office areas':1,
    'Recreational areas':2,
    'Industrial areas':3,
    'Church areas':4,
    'Market areas':5,
    'Rural village areas':6,
    'Outside rural areas':7,
    'Hospital areas':8,
    'School areas':9,
    'Rural village areasOffice areas':9,
    'Recreational areas':10

})
df['Area_accident_occured'] = df['Area_accident_occured'].fillna(df['Area_accident_occured'].mean())

df['Types_of_Junction']=df['Types_of_Junction'].map({
    'No junction':0,
    'Y Shape':1,
    'Crossing':2,
    'O Shape':3,
    'Unknown':4,
    'T Shape':5,
    'X Shape':6
    })
df['Types_of_Junction'] = df['Types_of_Junction'].fillna(df['Types_of_Junction'].mean())

df['Light_conditions']=df['Light_conditions'].map({
    'Daylight':0,
    'Darkness - lights lit':1,
    'Darkness - no lighting':2,
    'Darkness - lights unlit:':3
})
df['Light_conditions'] = df['Light_conditions'].fillna(df['Light_conditions'].mean())


df['Weather_conditions']=df['Weather_conditions'].map({
    'Normal':0,
    'Raining':1,
    'Raining and Windy':2,
    'Cloudy':3,
    'Windy':4,
    'Snow':5,
    'Fog or mist':6

})
df['Weather_conditions'] = df['Weather_conditions'].fillna(df['Weather_conditions'].mode()[0])


df['Type_of_collision']=df['Type_of_collision'].map({
    'Collision with roadside-parked vehicles':0,
    'Vehicle with vehicle collision':1,
    'Collision with roadside objects':2,
    'Collision with animals':3,
    'Rollover':4,
    'Fall from vehicles':5,
    'Collision with pedestrians':6,
    'With Train':7

})
df['Type_of_collision'] = df['Type_of_collision'].fillna(df['Type_of_collision'].mode()[0])

# Map 'Cause_of_accident' to numerical values
df['Cause_of_accident'] = df['Cause_of_accident'].map({
    'Moving Backward': 0,
    'Overtaking': 1,
    'Changing lane to the left': 2,
    'Changing lane to the right': 3,
    'Driving under the influence of drugs': 4,
    'No distancing': 5,
    'Driving carelessly': 6,
    'Driving at high speed': 7,
    'Overloading': 8,
    'Other': 9,
    'Improper parking': 10,
    'No priority to vehicle': 11,
    'No priority to pedestrian': 12,
    'Fatigue': 13,
    'Drunk driving': 14,
    'Bad Road': 15,
    'Bad weather': 16
})
df['Cause_of_accident'] = df['Cause_of_accident'].fillna(df['Cause_of_accident'].mode()[0])

df['Accident_severity']=df['Accident_severity'].map({
    'Slight Injury':0,
    'Serious Injury':1,
    'Fatal injury':2

})
df = df.dropna(subset=['Accident_severity'])



df['Day_of_week']=df['Day_of_week'].map({
    'Monday':1,
    'Sunday':0,
    'Friday':5,
    'Wednesday':3,
    'Saturday':6,
    'Thursday':4,
    'Tuesday':2

})
df['Day_of_week'] = df['Day_of_week'].fillna(df['Day_of_week'].mean())
features= df[[
    'Day_of_week',
    'Sex_of_driver',
    'Age_band_of_driver',
    'Educational_level',
    'Driving_experience',
    'Type_of_vehicle',
    'Area_accident_occured',
    'Types_of_Junction',
    'Light_conditions',
    'Weather_conditions',
    'Type_of_collision',
    'Number_of_casualties',
    'Cause_of_accident'
    ]]
target = df['Accident_severity']
x = features
y = target
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42, stratify=target)
import numpy as np

weight_map = {
    0: 1.0,
    1: 4.0,
    2: 6.0
}

sample_weights = np.array(
    [weight_map[int(label)] for label in y_train]
)
print("training class distribution ")
print(y_train.value_counts())
print("\nsample weight example ")
print(sample_weights[:10])
xgb_model = XGBClassifier(
    n_estimators= 300,
    max_depth =5,
    learning_rate= 0.05,
     subsample=0.8,
    colsample_bytree=0.8,

    objective='multi:softprob',
    num_class=3,

    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1

)

xgb_model.fit(
    x_train,
    y_train,
    sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_xgb)
print(f"ACCURACY: \n {accuracy}")
print("\nClassification Report:") 
print( classification_report( y_test, y_pred_xgb, zero_division=0 ) ) 
print("\nConfusion Matrix:") 
print( confusion_matrix( y_test, y_pred_xgb))
import joblib

import joblib

joblib.dump(
    xgb_model,
    "accident_model.pkl"
)

print("Model saved successfully")