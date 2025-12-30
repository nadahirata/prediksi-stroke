import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load dataset
df = pd.read_csv("stroke.csv")

# 2. Pilih kolom yang kita pakai
df = df[['gender', 'age', 'hypertension', 'heart_disease',
         'avg_glucose_level', 'bmi', 'stroke']]

# 3. Tangani data kosong
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# 4. Encode gender
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# 5. Pisahkan fitur & target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Simpan model
joblib.dump(model, "model_stroke.pkl")

print("MODEL BERHASIL DISIMPAN")
