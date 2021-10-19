'''
Neel Jog, Mark Ryman, Davin Seju
CSE 163 Winter 2021
This library contains most methods called in fifa19_main to perform our
analysis and visualizations of the FIFA19 dataset referenced in our report
'''
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_in_data(files):
    '''Loads in data from three different csv files to return
    DataFrame objects and one GeoPandas dataset file to return a
    GeoDataFrame object for analysis and visualizations
    '''
    df = pd.read_csv(files[0])
    prem = pd.read_csv(files[1])
    prem = prem.convert_dtypes()
    wc = pd.read_csv(files[2])
    wc = wc.convert_dtypes()
    wc = wc.iloc[::2]  # Remove redundant data in even rows
    world = gpd.read_file(files[3])
    world = world[["name", "geometry"]]
    return (df, prem, wc, world)


def research_q1a(df):
    '''Takes in the FIFA19 DataFrame (df) and returns a list of filtered
    DataFrames (top_players) listing top rated players for different
    skills categories, as well as a filtered DataFrame (do_need) required
    for subsequent methods
    '''
    dont_need = ["Photo", "Flag", "ID", "Club Logo", "Special",
                 "International Reputation", "Preferred Foot", "Body Type",
                 "Real Face", "Jersey Number", "Joined", "Loaned From",
                 "Contract Valid Until", "Release Clause", "Weak Foot"]
    do_need = df.drop(columns=dont_need)
    do_need = do_need.dropna()

    # For dribblers and dribbling
    dribble_list = ["Dribbling", "BallControl", "Acceleration", "Agility",
                    "Reactions", "Balance", "Positioning", "FKAccuracy",
                    "Vision"]
    do_need["Overall Skill"] = do_need.loc[:, dribble_list].sum(axis=1) /\
        len(dribble_list)
    top_dribblers = do_need.nlargest(5, "Overall Skill")
    top_dribblers.drop(top_dribblers.loc[:, "Work Rate":"GKReflexes"],
                       inplace=True, axis=1)
    sns.barplot(x='Name', y='Overall Skill', data=top_dribblers)
    plt.ylim(bottom=75, top=100)
    plt.title('Top Overall Skills')
    plt.xlabel('Player Name', style='italic')
    plt.ylabel('Overall Skill', style='italic')
    plt.savefig('q1a_skills_overall.png')
    plt.clf()

    # For shooters
    shooting_list = ["Finishing", "Volleys", "Curve", "ShotPower",
                     "LongShots", "Composure", "Penalties"]
    do_need["Shooting Overall"] = do_need.loc[:, shooting_list].sum(axis=1) /\
        len(shooting_list)
    top_shooters = do_need.nlargest(5, "Shooting Overall")
    top_shooters.drop(top_shooters.loc[:, "Work Rate":"GKReflexes"],
                      inplace=True, axis=1)
    sns.barplot(x='Name', y='Shooting Overall', data=top_shooters)
    plt.ylim(bottom=75, top=100)
    plt.title('Top Shooting Skills')
    plt.xlabel('Player Name', style='italic')
    plt.ylabel('Shooting Overall', style='italic')
    plt.savefig('q1a_shooting.png')
    plt.clf()

    # For passing
    passing_list = ["Crossing", "ShortPassing", "Curve", "LongPassing",
                    "Vision", "FKAccuracy"]
    do_need["Passing Overall"] = do_need.loc[:, passing_list].sum(axis=1) /\
        len(passing_list)
    top_passers = do_need.nlargest(5, "Passing Overall")
    top_passers.drop(top_passers.loc[:, "Work Rate":"GKReflexes"],
                     inplace=True, axis=1)
    sns.barplot(x='Name', y='Passing Overall', data=top_passers)
    plt.ylim(bottom=75, top=100)
    plt.title('Top Passing Skills')
    plt.xlabel('Player Name', style='italic')
    plt.ylabel('Passing Overall', style='italic')
    plt.savefig('q1a_passing.png')
    plt.clf()

    # For defending
    defending_list = ["HeadingAccuracy", "Jumping", "Strength",
                      "Aggression", "Interceptions", "Marking",
                      "StandingTackle", "SlidingTackle"]
    do_need["Defending Overall"] = do_need.loc[:, defending_list].sum(axis=1)\
        / len(defending_list)
    top_defenders = do_need.nlargest(5, "Defending Overall")
    top_defenders.drop(top_defenders.loc[:, "Work Rate":"GKReflexes"],
                       inplace=True, axis=1)
    sns.barplot(x='Name', y='Defending Overall', data=top_defenders)
    plt.ylim(bottom=75, top=100)
    plt.title('Top Defending Skills')
    plt.xlabel('Player Name', style='italic')
    plt.ylabel('Defending Overall', style='italic')
    plt.savefig('q1a_defending.png')
    plt.clf()

    # Top goalkeepers
    goalkeepers = df[df["Position"] == "GK"].copy()
    gk_list = ["GKDiving", "GKHandling", "GKKicking", "GKPositioning",
               "GKReflexes", "ShortPassing"]
    revised_goalkeeping = goalkeepers.loc[:, gk_list]
    goalkeepers.loc[:, "Goalkeeping"] = \
        revised_goalkeeping.loc[:, :].sum(axis=1) / len(gk_list)
    top_gks = goalkeepers.nlargest(5, "Goalkeeping")
    not_gk_relevant = top_gks.loc[:, "Work Rate":"GKReflexes"]
    top_gks.drop(not_gk_relevant, inplace=True, axis=1)
    sns.barplot(x='Name', y='Goalkeeping', data=top_gks)
    plt.ylim(bottom=75, top=100)
    plt.title('Top Goalkeeping Skills')
    plt.xlabel('Player Name', style='italic')
    plt.ylabel('Goalkeeping', style='italic')
    plt.savefig('q1a_goalkeeping.png')
    plt.clf()

    top_players = [top_dribblers, top_shooters, top_passers, top_defenders,
                   top_gks]
    return do_need, top_players, goalkeepers


def research_q1b(do_need, prem):
    '''Takes in the filtered fifa19 DataFrame (do_need) and the Premier League
    DataFrame (prem) and returns DataFrames for comparing best teams from the
    game and best performing Premier League teams in the real world stats
    '''
    relevant_prem = prem[["HomeTeam", "AwayTeam", "FTR"]]
    prem_results = dict()
    for index, row in relevant_prem.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        if home not in prem_results:
            prem_results[home] = 0
        if away not in prem_results:
            prem_results[away] = 0
        if result == "H":
            prem_results[home] += 3
        elif result == "A":
            prem_results[away] += 3
        else:
            prem_results[home] += 1
            prem_results[away] += 1
    prem_results["Manchester United"] = prem_results.pop("Man United")
    prem_results["Manchester City"] = prem_results.pop("Man City")
    prem_results["Cardiff City"] = prem_results.pop("Cardiff")
    prem_results["Huddersfield Town"] = prem_results.pop("Huddersfield")
    prem_results["Newcastle United"] = prem_results.pop("Newcastle")
    prem_results["Tottenham Hotspur"] = prem_results.pop("Tottenham")
    prem_results["Brighton & Hove Albion"] = prem_results.pop("Brighton")
    prem_results["Wolverhampton Wanderers"] = prem_results.pop("Wolves")
    prem_results["West Ham United"] = prem_results.pop("West Ham")
    prem_results["Leicester City"] = prem_results.pop("Leicester")

    top_of_prem = sorted(prem_results.items(), key=lambda x: x[1],
                         reverse=True)
    stats_best = pd.DataFrame(top_of_prem, columns=['Club', 'Points'])
    valid_clubs = prem_results.keys()
    in_prem = do_need["Club"].isin(valid_clubs)
    from_prem = do_need[in_prem]
    best_of_prem = from_prem.groupby("Club")["Overall"].mean()\
        .sort_values(ascending=False)
    best_df = best_of_prem.to_frame().reset_index()

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Club', y='Points', data=stats_best)
    plt.tight_layout
    plt.title('Top Premier League Teams', fontsize=15)
    plt.xlabel('Club Name', fontsize=15, style='italic')
    plt.xticks(rotation=75, fontsize=15)
    plt.ylabel('Final Points Tally', fontsize=15, style='italic')
    plt.savefig('q1b_stats_best.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Club', y='Overall', data=best_df)
    plt.tight_layout
    plt.title('Top Fifa 19 Teams', fontsize=15)
    plt.xlabel('Club Name', fontsize=15, style='italic')
    plt.xticks(rotation=75, fontsize=15)
    plt.ylabel('Aggregate Skill Rating', fontsize=15, style='italic')
    plt.savefig('q1b_best_of_prem.png', bbox_inches='tight')
    plt.clf()
    return stats_best, from_prem, best_of_prem


def research_q3(do_need, world, goalkeepers):
    '''Takes in the filtered FIFA19 DataFrame (do_need) and the GeoDataFrame
    (world) to perform analaysis of how players of different nationalities
    compare across key attributes and produces geospatial visualizations
    of the top five nationalities for each category, with the top ranked
    nationality highlighted.  Data visualizations are saved as local png
    files in local project diretory
    '''

    valid_countries = ["Belgium", "France", "Brazil", "England", "Portugal",
                       "Spain", "Argentina", "Uruguay", "Mexico", "Italy",
                       "Croatia", "Denmark", "Germany", "Netherlands",
                       "Colombia", "Switzerland", "Chile", "Wales", "Poland",
                       "Senegal", "United States"]
    valid_players = do_need["Nationality"].isin(valid_countries)
    valid_overall = do_need["Overall"] >= 75
    players_to_consider = do_need[valid_players & valid_overall]
    goalies_to_consider = goalkeepers[goalkeepers["Nationality"]
                                      .isin(valid_countries)]
    goalies_to_consider = goalies_to_consider[goalies_to_consider["Overall"]
                                              >= 75]
    is_attacker = ["LS", "ST", "RS", "LW", "RW", "LF", "CF", "RF", "LAM",
                   "CAM", "RAM"]
    is_midfielder = ["LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM",
                     "LAM", "CAM", "RAM", "LWB", "RWB"]
    is_defender = ["LWB", "LB", "LCB", "CB", "RCB", "RB", "RWB", "LDM",
                   "CDM", "RDM"]

    for_dribbling = players_to_consider[players_to_consider["Position"].isin(
        is_attacker+is_midfielder)]
    dribbling_by_country = for_dribbling.groupby("Nationality")[
        "Overall Skill"].mean().sort_values(ascending=False)
    for_shooting = players_to_consider[players_to_consider["Position"]
                                       .isin(is_attacker)]
    shooting_by_country = \
        for_shooting.groupby("Nationality")["Shooting Overall"].mean()\
        .sort_values(ascending=False)
    passing_by_country = \
        players_to_consider.groupby("Nationality")["Passing Overall"].mean()\
        .sort_values(ascending=False)
    for_defense = \
        players_to_consider[players_to_consider["Position"].isin(is_defender)]
    defense_by_country = \
        for_defense.groupby("Nationality")["Defending Overall"].mean()\
        .sort_values(ascending=False)

    gk_by_country = goalies_to_consider.groupby("Nationality")["Goalkeeping"]
    gk_by_country = gk_by_country.mean().sort_values(ascending=False)
    dribbling_by_country.to_csv('q3_skill.csv', index=True, header=True)
    shooting_by_country.to_csv('q3_shooting.csv', index=True, header=True)
    passing_by_country.to_csv('q3_passing.csv', index=True, header=True)
    defense_by_country.to_csv('q3_defense.csv', index=True, header=True)
    gk_by_country.to_csv('q3_gks.csv', index=True, header=True)

    #  Create a plot for each skill set of interest on global map
    skills = [(dribbling_by_country, "Overall Skill"),
              (shooting_by_country, "Shooting Overall"),
              (passing_by_country, "Passing Overall"),
              (defense_by_country, "Defending Overall"),
              (gk_by_country, "Goalkeeping")]

    skills_to_plot = []
    for i in range(len(skills)):
        this_tup = (world.merge(skills[i][0],
                    left_on='name',
                    right_on='Nationality').nlargest(5, skills[i][1]),
                    world.merge(skills[i][0],
                    left_on='name',
                    right_on='Nationality').nlargest(1, skills[i][1]))
        skills_to_plot.append(this_tup)

    titles = ["Top 5 Overall Skill by Nationality",
              "Top 5 Shooting by Nationality",
              "Top 5 Passing by Nationality",
              "Top 5 Defending by Nationality",
              "Top 5 Goalkeeping by Nationality"]

    file_names = ['best_dribbling.png', 'best_shooting.png',
                  'best_passing.png', 'best_defenders.png',
                  'best_goalkeepers.png']

    for i in range(len(titles)):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_ylim([-75, 75])
        ax.set_xlim([-200, 200])
        plt.title(titles[i], fontsize=15)
        world.plot(color='#EEEEEE', edgecolor='#FFFFFF', aspect='equal', ax=ax)
        skills_to_plot[i][0].plot(color='blue', edgecolor='black',
                                  aspect='equal', ax=ax)
        skills_to_plot[i][1].plot(color='gold', edgecolor='black',
                                  aspect='equal', ax=ax)
        plt.text(0.15, 0.35, 'Highest Ranked =', fontsize=12, color='black',
                 transform=plt.gcf().transFigure)
        plt.text(0.23, 0.35, '.', fontsize=70, color='gold',
                 transform=plt.gcf().transFigure)
        plt.text(0.15, 0.30, 'Other Top Five =', fontsize=12, color='black',
                 transform=plt.gcf().transFigure)
        plt.text(0.23, 0.30, '.', fontsize=70, color='blue',
                 transform=plt.gcf().transFigure)
        plt.savefig(file_names[i])


def research_q4(from_prem):
    '''Takes in a filtered DataFrame from the fifa19 dataset (from_prem) and
    analyzes clubs within that subset to determine a homogeneity index based
    on the makeup of players within each club from the same nationality.
    Returns all homogeneity results for clubs in the Premier leage, sorted
    from most homogenous to least, to provide grounds for analyzing possible
    correlation to real world club performance rankings.
    '''
    by_club = from_prem.groupby("Club")
    homogeneity_per_club = dict()
    for club, players in by_club:
        nationality_register = dict()
        for row_index, player in players.iterrows():
            nationality = player["Nationality"]
            if nationality not in nationality_register:
                nationality_register[nationality] = 0
            nationality_register[nationality] += 1
        max_possible_same = max(list(nationality_register.values()))
        num_players = sum(list(nationality_register.values()))
        homogeneity_per_club[club] = max_possible_same/num_players
    most_homogeneous = sorted(homogeneity_per_club.items(), key=lambda x: x[1],
                              reverse=True)
    most_homogeneous = pd.DataFrame(most_homogeneous, columns=['Clubs', 'HI'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x='Clubs', y='HI', data=most_homogeneous)
    plt.tight_layout
    plt.title('Homogeneity of Club Players Registered Nationality',
              fontsize=15)
    plt.xlabel('Club Name', fontsize=15, style='italic')
    plt.xticks(rotation=75, fontsize=15)
    plt.ylabel('Homogeneity Index Value', fontsize=15, style='italic')
    plt.savefig('q4_homogeneity.png', bbox_inches='tight')
    plt.clf()
    return most_homogeneous


def research_q5(wc, do_need):
    '''Takes in the World Cup 2018 DataFrame (wc) and the filtered fifa19
    DataFrame (do_need) and returns DataFrames for comparing if overall
    ratings of players in the game grouped by nationality correlate to
    performance of the respective nations in the World Cup tournament
    '''
    relevant_wc = wc[["Team", "Opponent", "WDL"]]
    wc_results = dict()
    for index, row in relevant_wc.iterrows():
        home = row["Team"]
        away = row["Opponent"]
        result = row["WDL"]
        if home not in wc_results:
            wc_results[home] = 0
        if away not in wc_results:
            wc_results[away] = 0
        if result == "W":
            wc_results[home] += 3
        elif result == "L":
            wc_results[away] += 3
        else:
            wc_results[home] += 1
            wc_results[away] += 1

    top_of_wc = sorted(wc_results.items(), key=lambda x: x[1], reverse=True)
    top_of_wc = pd.DataFrame(top_of_wc, columns=['Country', 'Points'])
    wc_countries = wc_results.keys()
    in_wc = do_need["Nationality"].isin(wc_countries)
    valid_for_wc = do_need[in_wc]
    probable_cutoffs = valid_for_wc.groupby("Nationality")["Overall"]\
        .nlargest(20)
    country_ratings = probable_cutoffs.groupby("Nationality").mean()\
        .sort_values(ascending=False)
    country_df = country_ratings.to_frame().reset_index()

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Country', y='Points', data=top_of_wc)
    plt.tight_layout
    plt.title('Best Teams in World Cup 2018', fontsize=15)
    plt.xlabel('Country', fontsize=15, style='italic')
    plt.xticks(rotation=75, fontsize=15)
    plt.ylabel('Total Points Tally', fontsize=15, style='italic')
    plt.savefig('q5_top_of_wc.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Nationality', y='Overall', data=country_df)
    plt.tight_layout
    plt.title('Best Teams by Nationality in FIFA 19', fontsize=15)
    plt.xlabel('Nationality', fontsize=15, style='italic')
    plt.xticks(rotation=75, fontsize=15)
    plt.ylabel('Aggregate Skills Rating', fontsize=15, style='italic')
    plt.savefig('q5_country_ratings.png', bbox_inches='tight')
    plt.clf()

    return top_of_wc, country_ratings
