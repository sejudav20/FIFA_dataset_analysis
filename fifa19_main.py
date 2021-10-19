'''
Neel Jog, Mark Ryman, Davin Seju
CSE 163 Winter 2021
This program runs all of the modules required to perform our analysis
and visualizations of the FIFA19 dataset referenced in our report
Ouput data saved as local .csv files as needed for analysis
'''
import geopandas as gpd
from fifa_lib import load_in_data, research_q1a, research_q1b
from fifa_lib import research_q3, research_q4, research_q5
# from ml_visualization import visualize_models
# from ml_train import train_save_both_models


def main():
    files = ["FIFA2019.csv",
             "england-premier-league-2018-to-2019.csv",
             "world_cup_2018_stats.csv",
             gpd.datasets.get_path('naturalearth_lowres')]
    df, prem, wc, world = load_in_data(files)
    do_need, top_players, goalkeepers = research_q1a(df)
    #  Create output files for 1a
    do_need.to_csv('q1a_do_need.csv', index=False, header=True)
    top_players[0].to_csv('q1a_top_dribblers.csv', index=False, header=True)
    top_players[1].to_csv('q1a_top_shooters.csv', index=False, header=True)
    top_players[2].to_csv('q1a_top_passers.csv', index=False, header=True)
    top_players[3].to_csv('q1a_top_defenders.csv', index=False, header=True)
    top_players[4].to_csv('q1a_top_gks.csv', index=False, header=True)

    top_of_prem, from_prem, best_of_prem = research_q1b(do_need, prem)
    #  Create output files for 1b
    top_of_prem.to_csv('q1b_top_of_prem.csv', index=False, header=True)
    from_prem.to_csv('q1b_from_prem.csv', index=False, header=True)
    best_of_prem.to_csv('q1b_best_of_prem.csv', index=False, header=True)

    # Must also un-comment 2 import statements above
    # train_save_both_models()  # un-comment this to train model as needed
    # visualize_models()  # un-comment to produce visuals once trained

    # Visualizations from visualize models provide q2 results

    research_q3(do_need, world, goalkeepers)
    #  Geospatial png files provide results for q3

    most_homogeneous = research_q4(from_prem)
    #  Create output file for 4
    most_homogeneous.to_csv('q4_most_homogeneous.csv', index=False,
                            header=True)

    top_of_wc, country_ratings = research_q5(wc, do_need)
    #  Create output files for 5
    top_of_wc.to_csv('q5_top_of_wc.csv', index=False, header=True)
    country_ratings.to_csv('q5_country_ratings.csv', index=True, header=True)


if __name__ == '__main__':
    main()
