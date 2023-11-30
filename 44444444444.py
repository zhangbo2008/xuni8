import pickle

# Take a dictionary
# marks = { 'Alex': 87, 'Lini': 92, 'Kiku': 90 }
# # Create a pickle file
# picklefile = open('marks', 'wb')
# # Pickle the dictionary and write it to file
# pickle.dump(marks, picklefile)
# # Close the file
# picklefile.close()





import numpy
picklefile = open('marks', 'rb')
a=pickle.load(picklefile)
print(1)