print(df.shape)

df['name'].apply(lambda x: x.split(' ')[0])

df.replace({'manufacturer' : 
            {'MAZDA' : 'Mazda', 'JAGUAR':'Jaguar', 'AUDI' : 'Audi', 'NISSAN': 'Nissan', 'MINI': 'Mini', 
             'VOLKSWAGEN':'Volkswagen', 'VAUXHALL':'Vauxhall', 'TOYOTA':'Toyota', 'SKODA':'Skoda', 'FORD':'Ford',
             'Bmw':'BMW','SUZUKI' : 'Suzuki', 'RENAULT':'Renault', 'PEUGEOT':'Peugeot', 'CITROEN':'Citroen',
             'VOLVO':'Volvo', 'FIAT':'Fiat', 'Ds':'DS', 'DACIA':'Dacia', 'ABARTH':'Abarth', 'SMART':'Smart', 
             'smart':'Smart','SEAT':'Seat', 'MITSUBISHI':'Mitsubishi', 'KIA':'Kia', 'HYUNDAI':'Hyundai',
             'HONDA':'Honda','MASERATI':'Maserati', 'PORSCHE':'Porsche', 'INFINITI':'Infiniti', 'Alfa':'Alfa-Romero',
             'Mercedes': 'Mercedes-Benz', 'MERCEDES-BENZ': 'Mercedes-Benz', 'Mercedes-benz': 'Mercedes-Benz',
             'Land':'Land-Rover', 'LAND': 'Land-Rover'}
             })

import numpy as np

# Assuming 'data' is your DataFrame with a 'price' column
df['price'] = df['price'].str.replace(',', '').str.replace('£', '').str.replace('$', '').astype(np.int64)


print(df.columns)

# Assuming 'df' is your DataFrame with a 'mileage' column
df['mileage'] = df['mileage'].str.replace(',', '').str.replace('miles', '')

# Replace empty strings and non-finite values with a default value
default_value = -1  # Choose an appropriate default value
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce', downcast='float').fillna(default_value).astype(np.int64)

df = df.replace({'engine' : 
                     {'Petrol hybrid': 'Hybrid', 'Petrol hybrid': 'Hybrid', 
                      'Petrol / electric hy' : 'Hybrid', 'Petrol plug-in hybri': 'Plug_in_hybrid',
                     'Petrol/electric' : 'Hybrid'}
                     })

df.replace({'transmission' : 
                     {'Semi auto': 'Semiautomatic', 'Semiauto': 'Semiautomatic',
                      'Semi automatic': 'Semiautomatic', 'Manual ': 'Manual',
                      'Semi-automatic': 'Semiautomatic', 'G-tronic+': 'Automatic',
                      'Cvt': 'Automatic'}
                     
                     })

