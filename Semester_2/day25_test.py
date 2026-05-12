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