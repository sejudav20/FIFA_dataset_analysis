"""
Neel Jog
Mark Ryman
Davin Seju
CSE 163+
ml_train.py file contains multiple methods
to train the models value overall
model and attribute model.
All methods use the data set
at FIFA2019.csv
If the module is run on main the module
will train both models.
"""
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error


def _convert_to_numeric(value):
    """_convert_to_numeric is a private method that takes
    in a string value as a parameter and returns a float.
    _convert_to_numeric is used to convert the player
    value column in the data from strings to more usable
    floats.
    """
    value = value[1:]
    if value[-1] == "M":
        value = float(value[:-1]) * (10**6)
    elif value[-1] == "K":
        value = float(value[:-1]) * (10**3)
    else:
        value = float(value)
    return value


def _convert_weight(value):
    """_convert_weight takes a string weight and returns
    it as a float. This method is a helper method
    to convert the weight column from the data to
    floats by taking out the units from each sample"""
    value2 = float(value.strip("lbs"))
    return value2


def _convert_height(value):
    """_convert_height takes a string height and returns
    it as a float. The method converts the data's height
    column from a list of heights in ft'inch format
    and returns the height in inches"""
    nums = value.split("'")

    return (int(nums[0])*12 + int(nums[1]))


def get_prepared_data_set():
    """get_prepared_data_set uses the previous methods
    to make the data from FIFA2019.csv in a numerical save_format
    that can be used with the model. The method takes away unnecessary
    columns and converts columns with strings to usable floats.
    The methods takes no parameters but returns
    the prepared data frame."""
    df = pd.read_csv('FIFA2019.csv')
    dont_need = ["Photo", "Flag", "ID", "Club Logo",
                 "Special", "International Reputation",
                 "Preferred Foot", "Body Type",
                 "Real Face", "Jersey Number",
                 "Joined", "Loaned From",
                 "Contract Valid Until", "Release Clause",
                 "Weak Foot"]
    do_need = df.drop(columns=dont_need)
    do_need = do_need.dropna()
    num_list = list()
    for val in do_need["Value"]:
        num_list.append(_convert_to_numeric(str(val)))
    do_need["Numerical Value"] = num_list

    weight_list = list()
    for weight in do_need["Weight"]:
        weight_list.append(_convert_weight(str(weight)))
    do_need["Numerical Weight"] = weight_list
    height_list = list()
    for height in do_need["Height"]:
        height_list.append(_convert_height(str(height)))
    do_need["Numerical Height"] = height_list
    # For dribblers and dribbling
    dribble_list = ["Dribbling", "BallControl", "Acceleration", "Agility",
                    "Reactions", "Balance", "Positioning", "FKAccuracy",
                    "Vision"]
    do_need["Overall Skill"] = do_need.loc[:, dribble_list].sum(axis=1) /\
        len(dribble_list)

    # For shooters
    shooting_list = ["Finishing", "Volleys", "Curve", "ShotPower",
                     "LongShots", "Composure", "Penalties"]
    do_need["Shooting Overall"] = do_need.loc[:, shooting_list].sum(axis=1) /\
        len(shooting_list)

    # For passing
    passing_list = ["Crossing", "ShortPassing", "Curve", "LongPassing",
                    "Vision", "FKAccuracy"]
    do_need["Passing Overall"] = do_need.loc[:, passing_list].sum(axis=1) /\
        len(passing_list)

    # For defending
    defending_list = ["HeadingAccuracy", "Jumping", "Strength",
                      "Aggression", "Interceptions", "Marking",
                      "StandingTackle", "SlidingTackle"]
    do_need["Defending Overall"] = do_need.loc[:, defending_list].sum(axis=1)\
        / len(defending_list)

    return do_need


def train_save_overall_model(path):
    """train_save_attribute_model takes in a file path.
    The method takes the prepared
    data set and trains a neural network with the data.
    The models weights are saved at the path specified
    so the model can be used without retraining. The function
    returns the training error of the model, the test error
    of the model, the scaler used to scale the features,
    and the scaler used on the values. These are all returned
    in a tuple. The model calculates error using mean_squared_error
    and thus the errors are floats."""
    data = get_prepared_data_set()

    features_list = ["Overall Skill", "Shooting Overall",
                     "Passing Overall", "Defending Overall"]

    features = data.loc[:, features_list]
    labels = data["Numerical Value"]
    x_train, x_test, y_train, y_test =\
        train_test_split(features.values, labels.values,
                         test_size=0.3, random_state=0)

    sc_x = StandardScaler()

    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
    y_test = sc_y.transform(y_test.reshape(-1, 1))

    # build and fit model

    value_model = Sequential()
    value_model.add(Dense(units=50, activation='relu',
                    input_shape=(x_test.shape[1],)))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=4, activation='relu'))
    value_model.add(Dense(units=4, activation='relu'))
    value_model.add(Dense(units=1))
    value_model.compile(optimizer='adam', loss="mean_squared_error")

    value_model.fit(x_train, y_train, batch_size=32, epochs=100)
    # Predict with model

    y_pred_test = value_model.predict(x_test)
    y_pred_train = value_model.predict(x_train)

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    # "C:\\Users\\davse\\Documents\\Machine Learnin\\valueModel.h5"
    value_model.save(path, save_format='h5')
    return (train_error, test_error, sc_x, sc_y)


def train_save_attribute_model(path):
    """train_save_attribute_model takes in a file path.
    The method takes the prepared
    data set and trains a neural network with the data.
    The models weights are saved at the path specified
    so the model can be used without retraining. The function
    returns the training error of the model, the test error
    of the model, the scaler used to scale the features,
    and the scaler used on the values. These are all returned
    in a tuple. The model calculates error using mean_squared_error
    and thus the errors are floats."""
    data = get_prepared_data_set()
    features_list = ["Crossing", "Finishing", "HeadingAccuracy",
                     "ShortPassing", "Volleys", "Dribbling", "Curve",
                     "FKAccuracy", "LongPassing", "BallControl",
                     "Acceleration", "SprintSpeed", "Agility", "Reactions",
                     "Balance", "ShotPower", "Jumping", "Stamina", "Strength",
                     "LongShots", "Aggression", "Interceptions",
                     "Positioning", "Vision", "Penalties", "Composure",
                     "Marking", "StandingTackle", "SlidingTackle",
                     "GKDiving", "GKHandling", "GKKicking", "GKPositioning",
                     "GKReflexes", "Age", "Skill Moves",
                     "Numerical Height", "Numerical Weight"]

    features = data.loc[:, features_list]
    labels = data["Numerical Value"]
    x_train, x_test, y_train, y_test =\
        train_test_split(features.values, labels.values,
                         test_size=0.3, random_state=0)

    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
    y_test = sc_y.transform(y_test.reshape(-1, 1))
    # build and fit model
    value_model = Sequential()
    value_model.add(Dense(units=110,
                    activation='relu', input_shape=(x_test.shape[1],)))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=50, activation='relu'))
    value_model.add(Dense(units=1))
    value_model.compile(optimizer='adam', loss="mean_squared_error")

    value_model.fit(x_train, y_train, batch_size=32, epochs=100)
    # Predict with model

    y_pred_test = value_model.predict(x_test)
    y_pred_train = value_model.predict(x_train)

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    # "C:\\Users\\davse\\Documents\\Machine Learnin\\valueModel.h5"
    value_model.save(path, save_format='h5')
    return (train_error, test_error, sc_x, sc_y)


def train_save_both_models():
    """train_save_both_models() takes no paramaters.
    The method trains both the overall model, which is saved
    to the absolute path valueOverallModel.h5, and
    the attribute model which is saved at
    valueAttributeModel.h5. These two file paths
    are used by ml_visualization.py in visualizeModels().
    The function returns a tuple containing
    the error in dollars of overall model
    and attribute model in that order."""
    train_error, test_error, sc_x, sc_y =\
        train_save_overall_model("valueOverallModel.h5")
    overall_test_error = sc_y.inverse_transform(np.array([test_error]))[0]
    train_error, test_error, sc_x, sc_y =\
        train_save_attribute_model("valueAttributeModel.h5")
    attribute_test_error = sc_y.inverse_transform(np.array([test_error]))[0]
    return overall_test_error, attribute_test_error


def get_models():
    """ get_models gets the saved models that were created by
    train_save_both_models. The function does not take any
    parameters but returns the Sequential models in a tuple
    with overall_model, and attribute model next"""
    overall_model = keras.models.load_model("valueOverallModel.h5")
    attribute_model = keras.models.load_model("valueAttributeModel.h5")
    return overall_model, attribute_model


def main():
    errors = train_save_both_models()
    print(errors)


if __name__ == "__main__":
    main()
