#the most expensive cars in the collection
df.nlargest(5, 'price')

#cars with the most miles travelled in the collection
df.nlargest(5, 'mileage')

#oldest cars in the collection
# df.nlargest(5, 'age')

#numbers of cars under the various car manufacturers
df.name.value_counts()

#count of the manufacture years 
# df.year.value_counts()

corr = df.corr()
corr

