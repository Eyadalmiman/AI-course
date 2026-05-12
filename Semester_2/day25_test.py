import numpy as np
import pandas as pd
from tensorflow import keras         # المكتبة الرئيسية لبناء الشبكات العصبية
from tensorflow.keras import layers  # بنستخرج منها أنواع الطبقات


value1 = float(input("Enter Flower Tall | Float: "))
value2 = float(input("Enter Flower Size | Float: "))
value3 = input("Enter Flower Color | Red-Blue-Orange-Purple: ")
value4 = float(input("Enter Flower Levs | Float: "))

if value3.lower() == "red":
    value3 = 1.0
elif value3.lower() == 'blue':
    value3 = 2.0
elif value3.lower() == 'orange':
    value3 = 4.0
elif value3.lower() == 'purple':
    value3 = 5.0
else:
    value3 = 0.0

# عينة اختبارية قيمة 4 مزايا للزهرة مثل
# لونها
# حجمها
# طول ورقة الزهره
# طول الورقة الخضراء لغصن الزهره
test_sample = np.array([[value1, value2, value3, value4]])

# قاموس يربط رقم الفئة باسم النبتة 
flower_names = {0: "Yasmen", 1: "NightFlower", 2: "SunFlower"}

# نقرأ ملف البيانات ونحوله إلى جدول (DataFrame)
data = pd.read_csv('Semester_2/Data/flowers_data.csv')

# X = المزايا 4 اعمده نبغا النموذج يتعلم منها
X = data[['Flower_Tall', 'Flower_Size', 'Flower_Color', 'Flower_Levs']].values

# y = العمود الي فيه الاجابه الصحيحه 0 او 1
y = data['label'].values

# بناء النموذج: طبقات الشبكة العصبية مرتبة على التوالي
model = keras.Sequential([
    keras.Input(shape=(4,)),                    # طبقة الإدخال: 4 ميزات لكل عينة
    layers.Dense(128, activation='relu'),       # طبقة مخفية أولى: 128 عصبون مع تنشيط ReLU
    layers.Dense(64, activation='relu'),        # طبقة مخفية ثانية: 64 عصبون مع تنشيط ReLU
    layers.Dense(3, activation='softmax'),      # 3 انواع زهور
])

# تجهيز النموذج للتدريب: نخبره كيف يتعلم
model.compile(
    optimizer='adam',                           # الي يحسن النموذج: الخوارزمية التي تعدل الأوزان لتقليل الخطأ
    loss='sparse_categorical_crossentropy',     # دالة الخسارة: تقيس كم النموذج مخطئ (للتصنيف متعدد الفئات)
    metrics=['accuracy']                        # المقاييس: تعرض لنا نسبة الإجابات الصحيحة أثناء التدريب
)

# التدريب الفعلي: النموذج يشاهد البيانات ويعدّل أوزانه ليتعلم
model.fit(
    X, y,                                       # X المدخلات، y الإجابات الصحيحة
    epochs=30,                                  # عدد المرات التي يمر فيها على كل البيانات (30 جولة)
    verbose=1                                   # 1 = اعرض تفاصيل التدريب، 0 = اعرض بصمت
)

# طباعة ملخص النموذج
model.summary()

# اختبار النموذج على العينة الجديدة
prediction = model.predict(test_sample)

# طباعة الاحتمالات
print(f"Predictions: {prediction[0]}")

# argmax يرجع رقم الفئة صاحبة أعلى احتمال (0 أو 1)
category_number = np.argmax(prediction[0])              # خزنا الرقم في متغير
print(f"Category: {category_number}")
print(f"Flower Name : {flower_names[category_number]}")    # نطبع الاسم من القاموس