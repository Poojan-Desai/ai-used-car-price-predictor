import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('data/cars.csv')

X = df[['year','miles','make','model','trim','condition']]
y = df['price']

num_features = ['year','miles','condition']
cat_features = ['make','model','trim']

pre = ColumnTransformer([
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

model = RandomForestRegressor(n_estimators=200, random_state=42)
pipe = Pipeline([('pre', pre), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

joblib.dump(pipe, 'model.pkl')
with open('metrics.json','w') as f:
    json.dump({'r2': float(r2), 'mae': float(mae)}, f, indent=2)

print('Saved model.pkl. R^2=', r2, 'MAE=', mae)
