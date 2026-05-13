# data, strucuture and maths
# time
import datetime as datetime
import glob
import itertools
import json
import math
import os
import random
import re
import ssl
import string
import threading
import time
from ast import literal_eval
from time import sleep

import emoji
import numpy as np
import pandas as pd

# text processing / regex
import regex

# dataviz and look/feel
import seaborn as sns
from IPython.display import HTML, clear_output, display
from more_itertools import unique_everseen

# pre-processing
from sklearn import preprocessing

# progress,performance and management
from tqdm import tqdm_notebook

display(HTML("<style>.container { width:100% !important; }</style>"))
sns.set(style="white", context="talk")
# %matplotlib inline # Removed for standalone compatibility if needed, but keeping original intent
sns.set_style("whitegrid")

# language & NLP
import warnings

import langdetect as ld

# network libraries and data viz
import networkx as nx
import plotly
import plotly.express as px
import plotly.graph_objs as go
import spacy
from networkx.algorithms import community

# plotly offline rendering
from plotly.offline import download_plotlyjs, iplot, plot

warnings.simplefilter("ignore")


class InstagramGraph:
    def __init__(
        self,
        csv,
        source_col="searched_for",
        post_col="post_caption",
        user_col="user_name",
        hashtag_col="post_hashtags",
        encoding="utf-8-sig",
    ):
        """
        Initializes the InstagramGraph class with a CSV file.

        Args:
            csv (str): Path to the input CSV file.
            source_col (str): Column name for the searched term (hashtag/profile).
            post_col (str): Column name for the post text/caption.
            user_col (str): Column name for the username.
            hashtag_col (str): Column name for the pre-extracted hashtags (list as string).
            encoding (str): Encoding of the CSV file.
        """
        try:
            self.df = pd.read_csv(csv, encoding=encoding).head(750)
        except UnicodeDecodeError:
            # Fallback to latin for older files
            self.df = pd.read_csv(csv, encoding="latin").head(750)

        self.post_col = post_col
        self.user_col = user_col
        self.hashtag_col = hashtag_col
        self.source_col = source_col

        self.nlp = spacy.load("en_core_web_lg")
        self.lemma_count = 0
        self.hashtag_count = 0

        # Handle searched_for missing in some edge cases
        if self.source_col in self.df.columns:
            self.source = self.df[self.source_col].unique()
        else:
            self.source = ["unknown"]

        self.default_stopwords = [
            "photooftheday",
            "picoftheday",
            "like4likes",
            "like4like",
            "instagood",
            "likeforlikes",
            "l4l",
            "likeforlike",
            "instagram",
            "follow4follow",
            "followforfollow",
            "instadaily",
            "instagrammers",
            "instalike",
            "follow",
            "likeforfollow",
            "like4follow",
            "instamood",
            "instafollow",
            "bestoftheday",
            "like",
            "followme",
            "instapic",
            "repost",
            "bhfyp",
        ]

        # Parse hashtag_col if it exists and contains stringified lists
        if self.hashtag_col in self.df.columns:

            def _parse_hashtags(val):
                tags = []
                if isinstance(val, str) and val.startswith("["):
                    try:
                        tags = literal_eval(val)
                    except:
                        tags = []
                elif isinstance(val, list):
                    tags = val
                
                # Solution 1: Strip '#' and ensure consistency (lowercase and remove empty)
                return [str(t).strip('#').lower() for t in tags if str(t).strip('#')]

            self.df[self.hashtag_col] = self.df[self.hashtag_col].apply(_parse_hashtags)

    # cleans and formats dataframe
    def cleaning(self):
        """
        Cleans and formats the dataframe.
        """
        # drop nulls on post column
        self.df.dropna(subset=[self.post_col], inplace=True)

        # convert any posts to string
        self.df[self.post_col] = self.df[self.post_col].map(lambda x: str(x))

        # remove emojis
        self.df[self.post_col] = self.df[self.post_col].map(
            lambda x: x.encode("ascii", "ignore").decode("ascii")
        )

        return self.df

    # extracts hashtags from any string returning list of hashtags
    def getHashtag(self, _string):

        # splits string into list and appends unique hashtags into a new list
        hashtags = [
            hashtag
            for hashtag in set(
                [token for token in _string.split() if token.startswith("#")]
            )
        ]

        # if there are hashtags in the string we process them further..
        if len(hashtags) > 0:
            # this will break up any hashtags that haven't been seperated by a space
            hashtags_seperated = [
                i for i in "".join(hashtags).strip().split("#") if len(i) > 0
            ]

            # this will remove any punctuation
            hashtags_clean = [
                hashtag.translate(str.maketrans("", "", string.punctuation))
                for hashtag in hashtags_seperated
            ]

            hashtags_clean = [i.lower() for i in hashtags_clean]

            # returns unique, cleaned hashtags without the
            return list(set(hashtags_clean))

        else:
            return np.nan

    # converts list of strings to lemma (if applicable) returning list of lemmas
    def getHashtagLemma(self, hashtags):

        # create a spacy document using hashtags as an argument
        doc = self.nlp(" ".join(hashtags))

        # empty list for lemmas
        tokens = []

        # loop through each token,
        for token in doc:
            if token.lemma_ != "-PRON-":
                tokens.append(token.lemma_)

                if str(token.text) != str(token.lemma_):
                    self.lemma_count += 1

        self.hashtag_count += len(tokens)

        return list(set(tokens))

    # gets a users post count that exists in the data
    def getUserpostcount(self, user):

        return self.user_count_dict.get(user, 0)

    # gets a users median hashtag use in the data
    def getUserhashtagcount(self, user):

        return self.user_hashtag_count_dict.get(user, 0)

    # gets the language of the string
    def getLanguage(self, _string):

        try:
            return ld.detect(_string)
        except:
            return np.nan

    def eda(self):
        """
        Language split
        """
        if hasattr(self, "translate") and self.translate == True:
            language_frame = pd.DataFrame(list(self.language_split.items()))

            language_frame.columns = ["language", "incidence"]

            language_frame.incidence = language_frame.incidence.map(
                lambda x: x / sum(language_frame.incidence)
            )

            language_low_incidence = language_frame[language_frame.incidence < 0.05]

            language_frame_summary = language_frame.replace(
                language_low_incidence.language.values, "other"
            )

            language_frame_summary = language_frame_summary.groupby("language")[
                "incidence"
            ].sum()

            language_frame_summary.sort_values(ascending=False, inplace=True)

            fig = go.Figure(
                [
                    go.Bar(
                        x=language_frame_summary.index,
                        y=language_frame_summary.values,
                        name="Secondary Product",
                    )
                ]
            )

            fig.update_layout(
                xaxis_tickangle=-45,
                title="Incidence of language by post for #" + self.source[0],
                xaxis_title="Language",
                yaxis_title="Incidence",
            )

            fig.show()

        def _histogram(metric, metric_label):

            fig = go.Figure(
                data=[go.Histogram(x=metric, histnorm="probability density")]
            )

            fig.update_layout(
                title=f"Distribution of {metric_label} for #" + self.source[0],
                xaxis_title=f"Count of {metric_label}",
                yaxis_title="Frequency",
            )

            return fig.show()

        if "user_post_count" in self.df.columns:
            _histogram(self.df.user_post_count, "user post frequency")

        if "hashtag_count" in self.df.columns:
            _histogram(self.df.hashtag_count, "hashtags by post")

        # get each user's posting frequency
        df_count = pd.DataFrame(self.df[self.user_col].value_counts())

        # col labels
        df_count.columns = ["post_freq"]

        # normalise column
        df_count["post_freq_norm"] = df_count["post_freq"].map(
            lambda x: int(x) / df_count["post_freq"].sum()
        )

        # cumulative sum on posts
        cum_sum_posts = np.cumsum(df_count["post_freq_norm"])

        users = []
        count = 1

        for i in range(self.df[self.user_col].nunique()):
            users.append(count)
            count += 1

        # normalise users
        if users:
            users_ = [i / users[-1] for i in users]

            # growth = pd.DataFrame(zip(users,cum_sum_posts))
            fig = go.Figure(data=go.Scatter(x=users_, y=cum_sum_posts))

            fig.update_layout(
                title=f"User post contribution for #" + self.source[0],
                xaxis_title="Normalised User Base",
                yaxis_title="Normalised Post Contribution",
            )

            fig.show()

        return

    # gets a list of hashtag lists from the dataset
    def getBatches(self, additional_stopwords=[]):

        # if no extra stopwords are specificed we use the defalut stop word list
        if len(additional_stopwords) == 0:
            self.current_stopwords = self.default_stopwords

        # append new stopwords to default stopword list
        else:
            self.current_stopwords = self.default_stopwords + additional_stopwords

        # function that iterates through list input and removes any stopwords
        def _removestop(words):
            if not isinstance(words, list):
                return []

            cleaned_words = words.copy()
            for stop_word in self.current_stopwords:
                try:
                    while stop_word in cleaned_words:
                        cleaned_words.remove(stop_word)
                except:
                    pass

            return cleaned_words

        # apply function to hashtag column
        df_nostop = self.target[self.target.columns[0]].map(_removestop)

        # create new list of lists containing hashtags
        batch = [[df_nostop.iloc[i]][0] for i in range(len(df_nostop.index))]

        return batch

    # calculates the edges and nodes that exist in the list of hashtag lists
    def getEdgesNodes(self, batches, min_frequency):

        # ranks hashtags in alphabetical order
        def _ranked_topics(batches):

            batches.sort()

            return batches

        # finds all possible unique combinations of topics
        def _unique_combinations(batches):
            return list(itertools.combinations(_ranked_topics(batches), 2))

        # adds each combination to a dictionary, if combination already exists value of key increases by one
        def _add_unique_combinations(_unique_combinations, _dict):

            for combination in _unique_combinations:
                if combination in _dict:
                    _dict[combination] += 1

                else:
                    _dict[combination] = 1

            return _dict

        edge_dict = {}

        source = []

        target = []

        edge_frequency = []

        # execute functions as above looping through each list, finding all unique combinations in each list
        # and adding them to dict object
        for batch in batches:
            if len(batch) >= 2:
                edge_dict = _add_unique_combinations(
                    _unique_combinations(batch), edge_dict
                )

        # create edge dataframe
        for key, value in edge_dict.items():
            source.append(key[0])

            target.append(key[1])

            edge_frequency.append(value)

        if not source:
            print("Warning: No edges found. Increase data or decrease min_frequency.")
            self.edge_df = pd.DataFrame(columns=["source", "target", "edge_frequency"])
            self.node_df = pd.DataFrame(columns=["id", "id_code"])
            return

        edge_df = pd.DataFrame(
            {"source": source, "target": target, "edge_frequency": edge_frequency}
        )

        edge_df.sort_values(by="edge_frequency", ascending=False, inplace=True)

        edge_df.reset_index(drop=True, inplace=True)

        # mask edge dataframe, only retinaing edges that occur n times
        edge_df = edge_df[edge_df["edge_frequency"] > min_frequency]

        if edge_df.empty:
            print("Warning: Edge dataframe empty after min_frequency filter.")
            self.edge_df = edge_df
            self.node_df = pd.DataFrame(columns=["id", "id_code"])
            return

        # create node dataframe
        node_df = pd.DataFrame(
            {"id": list(set(list(edge_df["source"]) + list(edge_df["target"])))}
        )

        labels = [i for i in range(len(node_df["id"]))]

        node_df["id_code"] = node_df.index

        # create a dictionary of all the nodes
        node_dict = dict(zip(node_df["id"], labels))

        # add relevant id's to each node in the edge dataframe
        edge_df["source_code"] = edge_df["source"].apply(lambda x: node_dict[x])

        edge_df["target_code"] = edge_df["target"].apply(lambda x: node_dict[x])

        # retain some attributes for the instance
        self.edge_df = edge_df

        self.node_df = node_df

        self.node_dict = node_dict

        self.edge_dict = edge_dict

        return

    # build the graph using the edge and node data
    def getGraph(self):

        if not hasattr(self, "edge_df") or self.edge_df.empty:
            return nx.Graph()

        # function that loops through and appends edge tuples to list
        def _extract_edges(edge_df):

            tuple_out = []

            # Using itertuples for faster iteration and to avoid index issues
            for row in edge_df.itertuples():
                tuple_out.append((row.source_code, row.target_code))

            return tuple_out

        # instantiate an instance of a Networkx graph
        G = nx.Graph()

        # add the nodes to the instance
        G.add_nodes_from(self.node_df.id_code)

        # extract the edges
        edge_tuples = _extract_edges(self.edge_df)

        # loop through and add each edge to the instance
        for i in edge_tuples:
            G.add_edge(i[0], i[1])

        return G

    """
    Pipeline of all methods
    """

    # generate all the features we need
    def getFeatures(self, translate=False, lemma=False):

        self.translate = translate

        if self.translate == True:
            print("Attempting to identify language...")

            # detect language using getLanguage method
            self.df["language"] = self.df[self.post_col].map(self.getLanguage)

            # get language split as a class dictionary attribute
            self.language_split = dict(self.df["language"].value_counts())
            print("Languages identified...")

        # call cleaning method
        self.df = self.cleaning()
        print("Data cleaned...")

        print("Attempting to extract hashtags...")
        # If hashtag_col was provided and exists, use it. Otherwise, extract from post_col.
        if self.hashtag_col in self.df.columns:
            self.df["hashtags"] = self.df[self.hashtag_col]
        else:
            self.df["hashtags"] = self.df[self.post_col].map(self.getHashtag)

        # drop any rows in the dataframe that don't have any hashtags
        self.df.dropna(subset=["hashtags"], inplace=True)

        # count of hashtags by post as new columns
        self.df["hashtag_count"] = self.df["hashtags"].map(lambda x: len(x))

        if lemma:
            print("Attempting to lemmatise hashtags...")
            # lemmatise any hashtags to new column
            self.df["hashtags_lemma"] = self.df["hashtags"].map(self.getHashtagLemma)

            if self.hashtag_count > 0:
                lemma_conversion = self.lemma_count / self.hashtag_count
                print(
                    f"Of {str(self.hashtag_count)} hashtags, {str(self.lemma_count)} hashtags were successfully lemmatised ({str(lemma_conversion)})"
                )
        else:
            # Use original hashtags (already cleaned and lowercase)
            self.df["hashtags_lemma"] = self.df["hashtags"]

        # get user post frequency as class attribute
        self.user_count_dict = dict(self.df[self.user_col].value_counts())

        # get median post count for each user as a class attribute
        self.user_hashtag_count_dict = dict(
            self.df.groupby(self.user_col)["hashtag_count"].median()
        )

        # get user post count as new column
        self.df["user_post_count"] = self.df[self.user_col].map(
            lambda x: self.getUserpostcount(x)
        )

        # get median user post count as new column
        self.df["user_median_hashtag_count"] = self.df[self.user_col].map(
            lambda x: self.getUserhashtagcount(x)
        )

        print("Running EDA and generating plots...")

        self.eda()

        return

    # select the data we want to include
    def selectData(self, english=True, remove_verified=True, max_posts=3, lemma=False):

        # retain some attributes
        self._filterenglish = english

        self._filterverified = remove_verified

        self._filterpostcount = max_posts

        self.df_edit = self.df.copy()

        if hasattr(self, "translate") and self.translate == True:
            # filter dataset to only include english if arg is true (default)
            if self._filterenglish == True:
                self.df_edit = self.df_edit[self.df_edit["language"] == "en"]

        # filter dataset to only include unverified accounts if arg is true (default)
        if (
            self._filterverified == True
            and "user_verified_status" in self.df_edit.columns
        ):
            self.df_edit = self.df_edit[self.df_edit["user_verified_status"] == False]

        # filter dataset to only include users who have posted under a threshold number of posts - gets rids of high volume posters
        if "user_post_count" in self.df_edit.columns:
            self.df_edit = self.df_edit[
                self.df_edit["user_post_count"] <= self._filterpostcount
            ]

        # retains the target column as an attribute of either hashtags that have been lemmatised or not
        if lemma == True:
            self.target = self.df_edit[["hashtags_lemma"]]
        else:
            self.target = self.df_edit[["hashtags"]]

        print("Data Selected.")

        return

    # create edges and nodes and add these to an instance of a graph object
    def buildGraph(self, additional_stopwords=[], min_frequency=5):

        # call getBatches method passing any contextual stop words as an arg
        batches = self.getBatches(additional_stopwords)

        # call getEdgesNodes mnethod taking max frequency as an arg
        self.getEdgesNodes(batches, min_frequency)

        # call the getGraph method and build the graph
        self.G = self.getGraph()

        if self.G.number_of_nodes() == 0:
            print("Graph build failed: no nodes/edges.")
            return

        print("Graph successfully built.")
        print("Node and Edge dataframes created.")

        """
        save a number of attributes to the instance of the class
        """
        # retain graph object adjacencies
        self.adjacencies = dict(self.G.adjacency())

        # retain graph object node betweeness centrality
        self.betweeness = nx.betweenness_centrality(self.G)

        # retain graph object clustering coefficients
        self.clustering_coeff = nx.clustering(self.G)

        """
        add these attributes as columns on the node dataframe
        """

        self.node_df["adjacency_frequency"] = self.node_df["id_code"].map(
            lambda x: len(self.adjacencies[x])
        )

        self.node_df["betweeness_centrality"] = self.node_df["id_code"].map(
            lambda x: self.betweeness[x]
        )

        self.node_df["clustering_coefficient"] = self.node_df["id_code"].map(
            lambda x: self.clustering_coeff[x]
        )

        # identify communities in instance of graph object and retain as attribute
        self.communities = community.greedy_modularity_communities(self.G)

        """
        assign each node to its community and add as column to node dataframe
        """
        self.communities_dict = {}

        nodes_in_community = [list(i) for i in self.communities]

        for i in nodes_in_community:
            self.communities_dict[nodes_in_community.index(i)] = i

        def community_allocation(source_val):
            for k, v in self.communities_dict.items():
                if source_val in v:
                    return k

        self.node_df["community"] = self.node_df["id_code"].map(
            lambda x: community_allocation(x)
        )

        print("Communities calculated.")
        return

    # plot the graph using plotly
    def plotGraph(
        self,
        sizing=75,
        node_size="adjacency_frequency",
        layout=nx.kamada_kawai_layout,
        light_theme=True,
        colorscale="Viridis",
        community_plot=False,
    ):

        if not hasattr(self, "G") or self.G.number_of_nodes() == 0:
            print("Nothing to plot.")
            return

        # formatting options for plot - dark vs. light theme
        if light_theme:
            back_col = "#ffffff"
            edge_col = "#ece8e8"

        else:
            back_col = "#000000"
            edge_col = "#2d2b2b"

        """
        normalise all graph metrics
        """
        # subset graph metrics
        X = self.node_df[self.node_df.columns[2:5]]

        # get columns labels
        cols = self.node_df.columns[2:5]

        # instantiate instance of MinMaxScaler class
        min_max_scaler = preprocessing.MinMaxScaler()

        # transform graph metrics
        X_scaled = min_max_scaler.fit_transform(X)

        # create new dataframe of scaled metrics
        plot_df = pd.DataFrame(X_scaled)

        plot_df.columns = cols

        for i in plot_df.columns:
            plot_df[i] = plot_df[i].apply(lambda x: x * sizing)

        # extract graph x,y co-ordinates from G instance
        pos = layout(self.G)

        # add position of each node from G to 'pos' key
        for node in self.G.nodes:
            self.G.nodes[node]["pos"] = list(pos[node])

        stack = []

        index = 0

        # add edges to Plotly go.Scatter object
        for edge in self.G.edges:
            x0, y0 = self.G.nodes[edge[0]]["pos"]

            x1, y1 = self.G.nodes[edge[1]]["pos"]

            weight = 0.5

            trace = go.Scatter(
                x=tuple([x0, x1, None]),
                y=tuple([y0, y1, None]),
                mode="lines",
                line={"width": weight},
                marker=dict(color=edge_col),
                line_shape="spline",
                opacity=1,
            )

            # append edge traces
            stack.append(trace)

            index = index + 1

        # conditionals for either showing a plot where formatting denotes community or not
        if community_plot == True:
            # make a partly empty dictionary for the nodes
            marker = {"size": [], "line": dict(width=0.5, color=edge_col), "color": []}

        else:
            # make a partly empty dictionary for the nodes
            marker = {
                "colorscale": colorscale,
                "size": [],
                "line": dict(width=0.5, color=edge_col),
                "color": [],
                "colorbar": dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
            }

        # initialise a go.Scatter object for the nodes
        node_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            text=[],
            mode="markers",
            textposition="bottom center",
            hoverinfo="text",
            marker=marker,
        )

        index = 0

        # add nodes to Plotly go.Scatter object
        for node in self.G.nodes():
            x, y = self.G.nodes[node]["pos"]

            node_trace["x"] += tuple([x])

            node_trace["y"] += tuple([y])

            node_trace["text"] += tuple(
                [self.node_df["id"].iloc[index]]
            )  # Use iloc for safety

            if community_plot == True:
                node_trace["marker"]["color"] += tuple(
                    [self.node_df.community.iloc[index]]
                )

                node_trace["marker"]["size"] += tuple([plot_df[node_size].iloc[index]])

            else:
                node_trace["marker"]["color"] += tuple(
                    [self.node_df.adjacency_frequency.iloc[index]]
                )

                node_trace["marker"]["size"] += tuple(
                    [self.node_df.adjacency_frequency.iloc[index]]
                )

            index = index + 1

        # append node traces
        stack.append(node_trace)

        # set up axis for plot
        axis = dict(
            showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        )

        # set up figure for plot
        figure = {
            "data": stack,
            "layout": go.Layout(
                title=str(self.source[0] + " is.."),
                font=dict(family="Arial", size=20),
                width=1100,
                height=1100,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                    l=40,
                    r=40,
                    b=85,
                    t=100,
                    pad=0,
                ),
                hovermode="closest",
                plot_bgcolor=back_col,  # set background color
            ),
        }

        # retain plot figure as attribute
        self.graph_plot = figure

        # plot the figure
        iplot(self.graph_plot)

        return

    # sunburst that plots communities and relevant hahstags
    def plotCommunity(self, colorscale=False):

        if not hasattr(self, "node_df") or self.node_df.empty:
            print("Nothing to plot.")
            return

        # make copy of node dataframe
        df_temp = self.node_df.copy()

        # change community label to string (needed for plot)
        df_temp["community"] = df_temp["community"].map(lambda x: str(x))

        # conditionals for plot type
        if colorscale == False:
            fig = px.sunburst(
                df_temp,
                path=["community", "id"],
                values="adjacency_frequency",
                color="community",
                hover_name=None,
                hover_data=None,
            )
        else:
            fig = px.sunburst(
                df_temp,
                path=["community", "id"],
                values="adjacency_frequency",
                color="betweeness_centrality",
                hover_data=None,
                color_continuous_scale="blugrn",
                color_continuous_midpoint=np.average(
                    df_temp["betweeness_centrality"],
                    weights=df_temp["betweeness_centrality"],
                ),
            )

        # add margin to plot
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

        # retain community plot as attribute
        self.community_plot = fig

        # offline sunburst plot
        iplot(self.community_plot)

        return

    # save map / sunburst plot locally as html file
    def savePlot(self, plot="map"):

        # get current time
        date = str(pd.to_datetime(datetime.datetime.now())).split(" ")[0]

        if plot == "map":
            if not hasattr(self, "graph_plot"):
                print("Plot 'map' not generated yet.")
                return

            plot_save = self.graph_plot

            filename = date + "_" + str(self.source[0]) + "_graph_plot_instagram.html"

            plotly.offline.plot(plot_save, filename=filename)

        elif plot == "community":
            if not hasattr(self, "community_plot"):
                print("Plot 'community' not generated yet.")
                return

            plot_save = self.community_plot

            filename = (
                date + "_" + str(self.source[0]) + "_community_plot_instagram.html"
            )

            plotly.offline.plot(plot_save, filename=filename)

        return print("Plot saved.")

    # save csv output
    def saveTables(self):

        date = str(pd.to_datetime(datetime.datetime.now())).split(" ")[0]

        source_name = str(self.source[0])

        if hasattr(self, "node_df"):
            self.node_df.to_csv(date + "_node_df_" + source_name + ".csv", index=False)
            print("Saved nodes")

        if hasattr(self, "edge_df"):
            self.edge_df.to_csv(date + "_edge_df_" + source_name + ".csv", index=False)
            print("Saved edges")

        if hasattr(self, "df_edit"):
            self.df_edit.to_csv(date + "_df_edit_" + source_name + ".csv", index=False)
            print("Saved edited dataframe")

        if hasattr(self, "df"):
            self.df.to_csv(date + "_df_" + source_name + "_.csv", index=False)
            print("Saved unedited dataframe")

        return
