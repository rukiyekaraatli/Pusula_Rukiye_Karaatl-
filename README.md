# Pusula_Rukiye_Karaatli
RUKİYE KARAATLI
rukiyekaraatli@gmail.com

Veri seti, kullanıcıların ilaç kullanımı ve yan etkilerini analiz etmek için çeşitli makine öğrenmesi projeleri geliştirmeye olanak tanır.
RUKİYE KARAATLI
rukiyekaraatli@gmail.com

Kütüphanelerin İçe Aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

Veri Setini Yükleme ve İnceleme
Kod:
# Verisetini yükleyelim.
sample_data = pd.read_excel("side_effect_data 1.xlsx")

# İlk satırlara göz atmak için:
print(sample_data.head())

# Veri yapısını incelemek için:
print(sample_data.info())

# Eksik veri kontrolü sağalamak için:
print(sample_data.isnull().sum())

Çıktı:
   Kullanici_id Cinsiyet Dogum_Tarihi    Uyruk         Il  \
0           107     Male   1960-03-01  Turkiye  Canakkale   
1           140     Male   1939-10-12  Turkiye    Trabzon   
2             2   Female   1976-12-17  Turkiye  Canakkale   
3            83     Male   1977-06-17  Turkiye      Adana   
4             7   Female   1976-09-03  Turkiye      Izmir 
    Ilac_Adi Ilac_Baslangic_Tarihi Ilac_Bitis_Tarihi  \

0                 trifluoperazine            2022-01-09        2022-03-04   
1                fluphenazine hcl            2022-01-09        2022-03-08   
2                 warfarin sodium            2022-01-11        2022-03-12   
3                   valproic acid            2022-01-04        2022-03-12   
4  carbamazepine extended release            2022-01-13        2022-03-06  

                Yan_Etki Yan_Etki_Bildirim_Tarihi Alerjilerim  \
0               Kabizlik      2022-02-19 18:28:43       Ceviz   
1              Yorgunluk      2022-02-03 20:48:17         Toz   
2               Carpinti      2022-02-04 05:29:20         Muz   
3             Sinirlilik      2022-02-08 01:01:21      Pancar   
4  Agizda Farkli Bir Tat      2022-02-12 05:33:06         NaN   

              Kronik Hastaliklarim  Baba Kronik Hastaliklari  \
0  Hipertansiyon, Kan Hastaliklari      Guatr, Hipertansiyon   
1                              NaN              Guatr, Diger   
2       Kalp Hastaliklari, Diyabet             Diyabet, KOAH   
3                   Diyabet, Diger  Kalp Hastaliklari, Diger   
4       Diyabet, Kalp Hastaliklari  Alzheimer, Hipertansiyon   

           Anne Kronik Hastaliklari    Kiz Kardes Kronik Hastaliklari  \
0                              KOAH  Kemik Erimesi, Kalp Hastaliklari   
1  Hipertansiyon, Kalp Hastaliklari                                     
2            Kemik Erimesi, Diyabet            Diyabet, Kemik Erimesi   
3                               NaN                             Astim   
4   Kan Hastaliklari, Kemik Erimesi                    Diyabet, Diger   

  Erkek Kardes Kronik Hastaliklari Kan Grubu   Kilo    Boy  
0             Kemik Erimesi, Guatr     B RH-  103.0  191.0  
1                    KOAH, Diyabet       NaN   81.0  181.0  
2                            Diger     B RH-   93.0  158.0  
3        Kalp Hastaliklari, Kanser    AB RH-    NaN  165.0  
4         Alzheimer, Hipertansiyon    AB RH-   99.0  172.0  
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2357 entries, 0 to 2356
Data columns (total 19 columns):
 #   Column                            Non-Null Count  Dtype         
---  ------                            --------------  -----         
 0   Kullanici_id                      2357 non-null   int64         
 1   Cinsiyet                          1579 non-null   object        
 2   Dogum_Tarihi                      2357 non-null   datetime64[ns]
 3   Uyruk                             2357 non-null   object        
 4   Il                                2130 non-null   object        
 5   Ilac_Adi                          2357 non-null   object        
 6   Ilac_Baslangic_Tarihi             2357 non-null   datetime64[ns]
 7   Ilac_Bitis_Tarihi                 2357 non-null   datetime64[ns]
 8   Yan_Etki                          2357 non-null   object        
 9   Yan_Etki_Bildirim_Tarihi          2357 non-null   datetime64[ns]
 10  Alerjilerim                       1873 non-null   object        
 11  Kronik Hastaliklarim              1965 non-null   object        
 12  Baba Kronik Hastaliklari          2201 non-null   object        
 13  Anne Kronik Hastaliklari          2140 non-null   object        
 14  Kiz Kardes Kronik Hastaliklari    2260 non-null   object        
 15  Erkek Kardes Kronik Hastaliklari  2236 non-null   object        
 16  Kan Grubu                         2010 non-null   object        
 17  Kilo                              2064 non-null   float64       
 18  Boy                               2243 non-null   float64       
dtypes: datetime64[ns](4), float64(2), int64(1), object(12)
memory usage: 350.0+ KB
None
Kullanici_id                          0
Cinsiyet                            778
Dogum_Tarihi                          0
Uyruk                                 0
Il                                  227
Ilac_Adi                              0
Ilac_Baslangic_Tarihi                 0
Ilac_Bitis_Tarihi                     0
Yan_Etki                              0
Yan_Etki_Bildirim_Tarihi              0
Alerjilerim                         484
Kronik Hastaliklarim                392
Baba Kronik Hastaliklari            156
Anne Kronik Hastaliklari            217
Kiz Kardes Kronik Hastaliklari       97
Erkek Kardes Kronik Hastaliklari    121
Kan Grubu                           347
Kilo                                293
Boy                                 114
dtype: int64

Yorum:
Bu veri setinde, özellikle kronik hastalıklar, cinsiyet, kan grubu, boy ve kilo gibi bazı kritik bilgilerin eksik olduğunu görüyorum. Eksik verilerin bu kadar yüksek oranda olduğu durumlarda, eksik değerleri doldurma (imputation), eksik verileri çıkarma veya farklı tekniklerle veri kalitesini artırma gibi adımlar uygulamak gerekir. Özellikle makine öğrenmesi modelleri için eksik değerlerle başa çıkmak önemli olacaktır.
 Veri Görselleştirme
Kod:
# Sayısal değişkenlerin dağılımını görmek için histogramları kullanalım.
sample_data.hist(bins=50, figsize=(20,15))
plt.show()

Çıktı: ![image](https://github.com/user-attachments/assets/3543eebb-2306-4c5a-95fd-e28bae445c33)


Yorum: 
Genel olarak sayısal değişkenlerde dengeli bir dağılım gözlemledim.Ama genel bir bakış atmak gerekirse :
100 den fazla kişi 1970-1980  tarihleri arasında doğmuş,
En yoğun yan etki bildirim tarihi 02-19 olarak gözüküyor.
80-85 kg kilo ya sahip kişi sayısı fazla ve 180-185 cm arası boya sahip insan yoğunlukta görülüyor.
Kod:
# Kategorik değişkenler için bar grafikleri kullanalım.
categorical_columns = ['Cinsiyet', 'Il', 'Kan Grubu']
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=sample_data, x=col)
    plt.title(f'{col} Dağılımı')
    plt.show()

Çıktı: ![image](https://github.com/user-attachments/assets/4e95dd3a-2d70-45ac-9b52-7818ef436973)

![image](https://github.com/user-attachments/assets/198353cc-9d00-4089-b258-a655fb1c750a)

![image](https://github.com/user-attachments/assets/2050703e-da34-4658-8153-7e25094a1b47)


Yorum:
Kadın sayısı erkeğe göre fazla, 200 den fazla kişi Adana yaşıyor ve en yoğun kan grubu AB RH- olarak görülüyor.
Kod:
# Isı haritası ile korelasyon analizi yapalım.
# Önce Sayısal sütunları seçelim
numerical_data = sample_data.select_dtypes(include=['float64', 'int64'])

# Korelasyon matrisi oluşturma ve ısı haritasını çizme
plt.figure(figsize=(10,8))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()
 
Çıktı: ![image](https://github.com/user-attachments/assets/83b419ec-febd-46f6-af74-9b6884c603f5)


Eksik Verilerin İşlenmesi
Kod:
# Kategorik değişkenlerde eksik verileri en sık görülen değer ile dolduralım
imputer_cat = SimpleImputer(strategy='most_frequent')
sample_data[categorical_columns] = imputer_cat.fit_transform(sample_data[categorical_columns])

# Sayısal değişkenlerde eksik verileri ortalama ile dolduralım
numerical_columns = ['Kilo', 'Boy']
imputer_num = SimpleImputer(strategy='mean')
sample_data[numerical_columns] = imputer_num.fit_transform(sample_data[numerical_columns])

Yorum:
Kategorik Değişkenler için Eksik Verilerin Doldurulması:
SimpleImputer sınıfı, eksik verileri doldurmak için kullanılan bir araçtır. Burada, strategy='most_frequent' parametresi ile kategorik değişkenlerde eksik veriler en sık görülen değer ile doldurulmaktadır. sample_data[categorical_columns], kategorik değişkenlerin bulunduğu DataFrame'in bir dilimidir. fit_transform metodu, bu dilimdeki eksik verileri doldurmak için en sık görülen değeri hesaplar ve eksik verileri bu değer ile değiştirir. 
Sayısal Değişkenler için Eksik Verilerin Doldurulması:
numerical_columns listesi, sayısal değişkenlerin adlarını içermektedir. Burada, 'Kilo' ve 'Boy' değişkenleri yer almaktadır. SimpleImputer yine kullanılarak, bu sayısal değişkenlerde eksik veriler ortalama (strategy='mean') ile doldurulmaktadır. sample_data[numerical_columns] kısmı, sayısal değişkenlerin bulunduğu DataFrame'in bir dilimidir ve fit_transform metodu ile eksik veriler ortalama ile doldurulmaktadır. Bu işlem, veri setindeki eksik verileri temizlemek ve daha sonrasında yapılacak analizler veya modelleme için verilerin hazır olmasını sağlamak amacıyla yapılmaktadır.
Kategorik Değişkenlerin Kodlanması
Kod: # Cinsiyet değişkenini LabelEncoder ile kodlama yapalım
le = LabelEncoder()
sample_data['Cinsiyet'] = le.fit_transform(sample_data['Cinsiyet'])

# Mevcut sütun adlarını kontrol edelim
print(sample_data.columns)

# Kan Grubu ve diğer çok sınıflı kategorik değişkenler için OneHotEncoder kullanımını gerçekleştirelim
ohe_columns = ['Kan Grubu', 'Ilac_Adi', 'Yan_Etki']
# Hatalı sütunları kontrol edip doğru isimlerle güncelleyelim
if all(col in sample_data.columns for col in ohe_columns):
    ohe = OneHotEncoder(sparse_output=False, drop='first')  # sparse parametresi güncellendi
    ohe_encoded = pd.DataFrame(ohe.fit_transform(sample_data[ohe_columns]), columns=ohe.get_feature_names_out(ohe_columns))

    # Orijinal sütunları düşürüp kodlanmış yeni sütunları ekleyelim
    sample_data = sample_data.drop(columns=ohe_columns).reset_index(drop=True)
    sample_data = pd.concat([sample_data, ohe_encoded], axis=1)
else:
    print("Eksik sütunlar: ", [col for col in ohe_columns if col not in sample_data.columns])

Çıktı: Index(['Kullanici_id', 'Cinsiyet', 'Dogum_Tarihi', 'Uyruk', 'Il',
       'Ilac_Baslangic_Tarihi', 'Ilac_Bitis_Tarihi',
       'Yan_Etki_Bildirim_Tarihi', 'Alerjilerim', 'Kronik Hastaliklarim',
       ...
       'Yan_Etki_Kabizlik', 'Yan_Etki_Karin Agrisi', 'Yan_Etki_Kas Agrisi',
       'Yan_Etki_Mide Bulantisi', 'Yan_Etki_Sinirlilik',
       'Yan_Etki_Tansiyon Dusuklugu', 'Yan_Etki_Tansiyon Yukselme',
       'Yan_Etki_Terleme', 'Yan_Etki_Uykululuk Hali', 'Yan_Etki_Yorgunluk'],
      dtype='object', length=194)
Eksik sütunlar:  ['Kan Grubu', 'Ilac_Adi', 'Yan_Etki']
Sayısal Verilerin Normalizasyonu/Standardizasyonu
Kod:
# Sayısal verileri standardizasyon (z-skoru) işlemini gerçekleştirelim
scaler = StandardScaler()
sample_data[numerical_columns] = scaler.fit_transform(sample_data[numerical_columns])

# Son veri setine göz atalım
print(sample_data.head())
Çıktı.
   Kullanici_id  Cinsiyet Dogum_Tarihi    Uyruk         Il  \
0           107         1   1960-03-01  Turkiye  Canakkale   
1           140         1   1939-10-12  Turkiye    Trabzon   
2             2         0   1976-12-17  Turkiye  Canakkale   
3            83         1   1977-06-17  Turkiye      Adana   
4             7         0   1976-09-03  Turkiye      Izmir   

  Ilac_Baslangic_Tarihi Ilac_Bitis_Tarihi Yan_Etki_Bildirim_Tarihi  \
0            2022-01-09        2022-03-04      2022-02-19 18:28:43   
1            2022-01-09        2022-03-08      2022-02-03 20:48:17   
2            2022-01-11        2022-03-12      2022-02-04 05:29:20   
3            2022-01-04        2022-03-12      2022-02-08 01:01:21   
4            2022-01-13        2022-03-06      2022-02-12 05:33:06   

  Alerjilerim             Kronik Hastaliklarim  ... Yan_Etki_Kabizlik  \
0       Ceviz  Hipertansiyon, Kan Hastaliklari  ...               1.0   
1         Toz                              NaN  ...               0.0   
2         Muz       Kalp Hastaliklari, Diyabet  ...               0.0   
3      Pancar                   Diyabet, Diger  ...               0.0   
4         NaN       Diyabet, Kalp Hastaliklari  ...               0.0   

  Yan_Etki_Karin Agrisi Yan_Etki_Kas Agrisi Yan_Etki_Mide Bulantisi  \
0                   0.0                 0.0                     0.0   
1                   0.0                 0.0                     0.0   
2                   0.0                 0.0                     0.0   
3                   0.0                 0.0                     0.0   
4                   0.0                 0.0                     0.0   

   Yan_Etki_Sinirlilik  Yan_Etki_Tansiyon Dusuklugu  \
0                  0.0                          0.0   
1                  0.0                          0.0   
2                  0.0                          0.0   
3                  1.0                          0.0   
4                  0.0                          0.0   

   Yan_Etki_Tansiyon Yukselme  Yan_Etki_Terleme  Yan_Etki_Uykululuk Hali  \
0                         0.0               0.0                      0.0   
1                         0.0               0.0                      0.0   
2                         0.0               0.0                      0.0   
3                         0.0               0.0                      0.0   
4                         0.0               0.0                      0.0   

   Yan_Etki_Yorgunluk  
0                 0.0  
1                 1.0  
2                 0.0  
3                 0.0  
4                 0.0  

[5 rows x 194 columns]

YORUM:
Bu veri seti, kullanıcıların ilaç kullanımı ile ilgili çeşitli analizler yapmak için uygundur.Örnek olarak;Yan etkilerin analizi,Cinsiyet veya diğer demografik özelliklere göre yan etki deneyimlerinin karşılaştırılması,Kullanıcıların sağlık geçmişleri ile yan etkiler arasındaki ilişkiler incelenebilir.
MAKİNE ÖĞRENMESİ PROJESİ GELİŞTİRECEK OLSAM ;

Yan Etki Tahmini Modeli
Amaç: Kullanıcıların belirli ilaçları kullanırken yaşayabilecekleri yan etkileri tahmin etmek.
Aşamalar: Veri ön işleme, öznitelik seçimi, sınıflandırma algoritmaları ile model geliştirme ve değerlendirme.
Kullanıcı Profiline Dayalı Öneri Sistemi
Amaç: Kullanıcıların geçmiş sağlık verilerine dayanarak uygun ilaçları önermek.
Aşamalar: Veri ön işleme, öneri modelleme ve kullanıcı geri bildirimleri toplama.

Bu modelleri kullanabilirdim.
