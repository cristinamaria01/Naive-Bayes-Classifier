import numpy as np
import pandas
from NBClassifier import NBClassifier


def get_data(file_name):
    reader = pandas.read_csv(file_name, iterator=True, delimiter=', ', engine="python")
    df_train = pandas.concat(reader, ignore_index=True)  # Creating the dataframe

    data = np.asarray(df_train, dtype=str)
    X = data[:, 0:14]
    y = data[:, 14:15]
    return X, y


predictions = []

# Get train and validation data
X_train, y_train = get_data('adult_train.csv')
X_val, y_val = get_data('adult_test.csv')

y_train = y_train.tolist()
X_val = X_val.tolist()
y_val = y_val.tolist()

train_labels = [label[0] for label in y_train]
val_labels = [label[0] for label in y_val]

classifier = NBClassifier()
classifier.train(X_train, train_labels)

for item in X_val:
    result = classifier.predict(item)
    predictions.append(str(result))

correctly_classified_counter = 0
for index, prediction in enumerate(predictions):
    # Each label from validation dataset contains a '.' char at the end
    # We replace that char in order to match with values from predictions list
    if prediction == val_labels[index].replace('.', ''):
        correctly_classified_counter += 1

accuracy = correctly_classified_counter / len(y_val)
print("Model accuracy on validation data is: {}".format(accuracy))



