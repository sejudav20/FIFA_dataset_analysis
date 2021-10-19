"""
Neel Jog
Mark Ryman
Davin Seju
CSE 163+
ml_visualization.py file contains multiple methods
to display visualizations for models
created and saved in ml_train.py.
If the module is run on main the module
will create the visualizations described in the
following methods for both attribute and overall
models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from ml_train import get_prepared_data_set
from ml_train import get_models
import seaborn as sns
sns.set()
plt.close('all')

_features_max_value = dict()


def get_scalers(features_list):
    """get_scalers takes a list of strings features_list
    and returns a tuple (scaler for features, scaler for labels).
    The function reads all the data from the FIFA2019.csv on
    the local directory. The method then takes the features
    listed and the labels using two respective StandardScaler
    objects that are returned by the method as a tuple"""
    do_need = get_prepared_data_set()
    features = do_need.loc[:, features_list]
    labels = do_need["Numerical Value"]
    x_train, x_test, y_train, y_test =\
        train_test_split(features.values, labels.values, test_size=0.3)

    sc_x = StandardScaler()

    x_train = sc_x.fit_transform(x_train)

    x_test = sc_x.transform(x_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

    return (sc_x, sc_y)


def _graph_feature(model, features_test, index, ax,
                   finalValue, sc_x, sc_y, feature_name):
    """_graph_features is an helper method that takes
    a keras sequential model, the features, index of
    which feature is not constant, ax the subplot
    where the graph will be plotted, the highest value
    the independent feature will be increased, the two
    scalers one for features and one for the label, and the
    name of the independent feature being changed. The method
    takes the features and increases the feature at the index
    specified and plots a graph with the specified feature
    on the x axis and the model predicted value
    on the y axis. The graph is saved to the subplot
    at ax. The function returns none."""
    initialValue = features_test[index]
    feature_increments = range(initialValue, finalValue + 1)
    values = [0]*len(feature_increments)
    i = 0
    for feature_num in feature_increments:
        features_test[index] = feature_num
        features_scaled = sc_x.transform(features_test.reshape(1, -1))
        pred_labels = model.predict(features_scaled)
        values[i] = sc_y.inverse_transform(pred_labels)
        i += 1
    _features_max_value[feature_name] = values[-1][0][0]
    values = np.array(values).reshape(len(feature_increments), 1)
    feature_increments = np.array(feature_increments)\
        .reshape(len(feature_increments), 1)

    data = np.concatenate((feature_increments, values), axis=1)
    df = pd.DataFrame(data=data, columns=["feature", "value"])
    sns.lineplot(data=df, x="feature", y="value", ax=ax)
    ax.set_xlabel(feature_name + "")
    ax.set_label("Value $")
    ax.set_title(f"Impact of {feature_name} on Value")


def graph_overall_features(model_path, save_path):
    """graph_overall_features takes the path of the overall model
    saved in ml_train.py and a save_path for the diagram
    this method will produce. The method
    creates graphs demonstrating the affect of
    changing each feature in the overall model on the value
    of the player while keeping the other features constant.
    The function makes one figure containing the respective graphs
    of all 5 features and returns None."""
    model = keras.models.load_model(model_path)
    features_list = ["Overall Skill", "Shooting Overall", "Passing Overall",
                     "Defending Overall"]

    sc_x, sc_y = get_scalers(features_list)
    data_list = ([60] * 4)
    features_test = np.array(data_list)
    fig, axs = plt.subplots(4, 1, figsize=(14, 14))

    for i in range(0, len(data_list)):
        _graph_feature(model, features_test, i, axs[i],
                       90, sc_x, sc_y, features_list[i])
    fig.tight_layout()
    fig.savefig(save_path)


def graph_attribute_features(model_path, save_path):
    """graph_attribute_features takes the path of the attribute model
    saved in ml_train.py and a save_path for the diagram
    this method will produce. The method
    creates graphs demonstrating the affect of
    changing each feature in the attribute model on the value
    of the player while keeping the other features constant.
    The function makes one figure containing the respective graphs
    of all 38 features and returns None."""
    model = keras.models.load_model(model_path)
    features_list = ["Crossing", "Finishing", "HeadingAccuracy",
                     "ShortPassing", "Volleys", "Dribbling", "Curve",
                     "FKAccuracy", "LongPassing", "BallControl",
                     "Acceleration", "SprintSpeed", "Agility",
                     "Reactions", "Balance", "ShotPower",
                     "Jumping", "Stamina", "Strength",
                     "LongShots", "Aggression", "Interceptions",
                     "Positioning", "Vision", "Penalties",
                     "Composure", "Marking", "StandingTackle",
                     "SlidingTackle", "GKDiving", "GKHandling",
                     "GKKicking", "GKPositioning", "GKReflexes",
                     "Age", "Skill Moves",
                     "Numerical Height", "Numerical Weight"]
    sc_x, sc_y = get_scalers(features_list)
    data_list = ([60]*34)
    data_list.extend([22])
    data_list.extend([1])
    data_list.extend([68])
    data_list.extend([160])
    range_list = {60: 90, 22: 30, 1: 4, 160: 175, 68: 74}
    features_test = np.array(data_list)
    fig, ax = plt.subplots(5, 2, figsize=(14, 14))
    fig2, ax2 = plt.subplots(5, 2, figsize=(14, 14))
    fig3, ax3 = plt.subplots(5, 2, figsize=(14, 14))
    fig4, ax4 = plt.subplots(4, 2, figsize=(14, 14))

    axs = np.concatenate((ax.reshape(10,),
                          ax2.reshape(10,), ax3.reshape(10,),
                          ax4.reshape(8,)))
    for i in range(0, len(data_list)):
        _graph_feature(model, features_test, i, axs[i],
                       range_list[data_list[i]], sc_x,
                       sc_y, features_list[i])
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig.savefig(save_path[0])
    fig2.savefig(save_path[1])
    fig3.savefig(save_path[2])
    fig4.savefig(save_path[3])


def _predict_pay(data, features, model):
    """_predict_pay takes a pandas dataframe,
    a list of features and the model itself.The method
    predicts the pay of one sample of the features
    provided in data using the model. This method
    can be used for both models but the data
    must have the amount of features required for the
    respective models. The method returns the value as
    a double."""
    sc_x, sc_y = get_scalers(features)
    m_data = sc_x.transform(data.values)
    value = model.predict(m_data)
    return sc_y.inverse_transform(value)


def test_model_on_players():
    """test_model_on_players takes no parameters
    and returns None. The method runs both attribute
    and overall model on certain players
    and outputs the results along with the name and actual
    value as a print statement"""
    o_model, a_model = get_models()
    df = get_prepared_data_set()

    attribute_list = ["Crossing", "Finishing", "HeadingAccuracy",
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

    overall_list = ["Overall Skill", "Shooting Overall",
                    "Passing Overall", "Defending Overall"]
    messi_mask = df['Name'] == "L. Messi"
    messi_o_features = df.loc[messi_mask, overall_list]
    messi_a_features = df.loc[messi_mask, attribute_list]
    messi_actual_value = df.loc[messi_mask, "Value"].values
    messi_o_value = _predict_pay(messi_o_features,
                                 overall_list, o_model)
    messi_a_value = _predict_pay(messi_a_features,
                                 attribute_list, a_model)
    print(f"Lionel Messi actual:{messi_actual_value} overall model:"
          f"{messi_o_value} attribute model: {messi_a_value}")


def visualize_models():
    """visualize_models is a method that takes no parameters
    and returns none. The method graphs the overall and
    attribute models and saves all the graphs for the
    attribute value at Attributes1.png, Attributes2.png
    , Attributes3.png, and Attributes.png.
    The graphs of the Overall model will be saved in Overall.png.
    This method just combines both graph_overall_features
    and graph_attribute_features using absolute file paths
    defined in ml_train.py."""
    attribute_model_path = "valueAttributeModel.h5"
    overall_model_path = "valueOverallModel.h5"
    attribute_file_save_path = ["Attributes1.png",
                                "Attributes2.png",
                                "Attributes3.png",
                                "Attributes4.png"]
    overall_file_path = "Overall.png"
    graph_overall_features(overall_model_path, overall_file_path)
    graph_attribute_features(attribute_model_path, attribute_file_save_path)


def main():
    visualize_models()
    test_model_on_players()
    # print(_features_max_value)
    features_in_order = sorted(_features_max_value.items(), key=lambda x:
                               x[1], reverse=True)
    print(features_in_order)


if __name__ == "__main__":
    main()
