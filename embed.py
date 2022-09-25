# Dimensionality reduction using three differeent methods:
# Multidimensional Scaling (MDS)
# Isomap (ISO)
# Laplacian Eigenmaps (LEM)
# outputs csv file(s) of embedded data and specified label if applicable

import argparse
import sklearn
import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, SpectralEmbedding, MDS
import os
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D

# random seed for sklearn
np.random.seed(1902)

parser = argparse.ArgumentParser(
    description="Dimensionality Reduction w/ MDS,Isomap,LEM")

parser.add_argument(
    "-in",
    "--input_path",
    type=str,
    default="data.csv",
    help="path to the input data",
)

parser.add_argument(
    "-out",
    "--output_path",
    type=str,
    default="./embedded_data/",
    help="path to the output data without filename",
)

parser.add_argument(
    "-d",
    "--d",
    type=int,
    default=2,
    help="dimensionality of the final embedding",
)

parser.add_argument(
    "-y",
    "--label",
    type=str,
    default="y",
    help="name of column with the label",
)

parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="all",
    help="Choose between 'MDS', 'ISO', 'LEM' or 'all'",
)

parser.add_argument(
    "-n",
    "--n",
    type=int,
    default=2,
    help="Number of neighbours for LEM and Isomap",
)

parser.add_argument(
    "-g",
    "--gamma",
    type=int,
    default=np.infty,
    help=
    "parameter in kernel of LEM, default is infinity which leads to binary weigths",
)

parser.add_argument(
    "-met",
    "--metric",
    type=str,
    default="euclidean",
    help="Metric used in Isomap, see Isomap documentation for options",
)

parser.add_argument(
    "-v",
    "--visualize",
    type=bool,
    default=False,
    help="Visualize results and store as png file in subfolder ./plots",
)

args = parser.parse_args()


# function to create folder to store results from this file
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory: " + directory)


def fit_embedding(args):

    # output folder
    createFolder("{}".format(args.output_path))

    if args.visualize:
        createFolder("{}plots/".format(args.output_path))

    # input data
    df = pd.read_csv("{}".format(args.input_path))

    # deal with label if applicable
    try:
        # store label separately
        y = pd.DataFrame(data=df[args.label], columns=["y"])
        # remove label from original data
        df.drop([args.label], axis=1, inplace=True)
        is_label = True
    except:
        is_label = False

    # make column names for learned data
    columns = []
    for i in range(args.d):
        columns.append("x_" + str(i + 1))

    # MDS only
    if args.method == "MDS":

        embeddingMDS = MDS(n_components=args.d)
        MDSdata = embeddingMDS.fit_transform(df)
        MDSdata = pd.DataFrame(data=MDSdata, columns=columns)

        try:
            MDSdata["y"] = y
        except:
            pass

        MDSdata.to_csv("{}MDS_{}d.csv".format(args.output_path, args.d),
                       index=False)

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    MDSdata.iloc[:, 0],
                    MDSdata.iloc[:, 1],
                    c=MDSdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via MDS")
                plt.savefig(
                    "{}plots/MDS_2d.png".format(args.output_path),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    MDSdata.iloc[:, 0],
                    MDSdata.iloc[:, 1],
                    MDSdata.iloc[:, 2],
                    c=MDSdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via MDS")
                plt.savefig(
                    "{}plots/MDS_3d.png".format(args.output_path),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

    elif args.method == "ISO":

        embeddingISO = Isomap(n_neighbors=args.n,
                              n_components=args.d,
                              metric=args.metric)
        ISOdata = embeddingISO.fit_transform(df)
        ISOdata = pd.DataFrame(data=ISOdata, columns=columns)
        try:
            ISOdata["y"] = y
        except:
            pass
        ISOdata.to_csv(
            "{}ISO_{}NN_{}d.csv".format(args.output_path, args.n, args.d),
            index=False,
        )

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    ISOdata.iloc[:, 0],
                    ISOdata.iloc[:, 1],
                    c=ISOdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via Isomap - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/Isomap_{}NN_2d.png".format(
                        args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    ISOdata.iloc[:, 0],
                    ISOdata.iloc[:, 1],
                    ISOdata.iloc[:, 2],
                    c=ISOdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via Isomap - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/Isomap_{}NN_3d.png".format(
                        args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

    elif args.method == "LEM":

        embeddingLEM = SpectralEmbedding(n_neighbors=args.n,
                                         n_components=args.d,
                                         gamma=args.gamma)
        LEMdata = embeddingLEM.fit_transform(df)
        LEMdata = pd.DataFrame(data=LEMdata, columns=columns)
        try:
            LEMdata["y"] = y
        except:
            pass
        LEMdata.to_csv(
            "{}LEM_{}NN_{}d.csv".format(args.output_path, args.n, args.d),
            index=False,
        )

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    LEMdata.iloc[:, 0],
                    LEMdata.iloc[:, 1],
                    c=LEMdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via LEM - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/LEM_{}NN_2d.png".format(args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    LEMdata.iloc[:, 0],
                    LEMdata.iloc[:, 1],
                    LEMdata.iloc[:, 2],
                    c=LEMdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via LEM - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/LEM_{}NN_3d.png".format(args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

    elif args.method == "all":

        embeddingMDS = MDS(n_components=args.d)
        MDSdata = embeddingMDS.fit_transform(df)
        MDSdata = pd.DataFrame(data=MDSdata, columns=columns)
        try:
            MDSdata["y"] = y
        except:
            pass
        MDSdata.to_csv("{}MDS_{}d.csv".format(args.output_path, args.d),
                       index=False)

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    MDSdata.iloc[:, 0],
                    MDSdata.iloc[:, 1],
                    c=MDSdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via MDS")
                plt.savefig(
                    "{}plots/MDS_2d.png".format(args.output_path),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    MDSdata.iloc[:, 0],
                    MDSdata.iloc[:, 1],
                    MDSdata.iloc[:, 2],
                    c=MDSdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via MDS")
                plt.savefig(
                    "{}plots/MDS_3d.png".format(args.output_path),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

        embeddingISO = Isomap(n_neighbors=args.n,
                              n_components=args.d,
                              metric=args.metric)
        ISOdata = embeddingISO.fit_transform(df)
        ISOdata = pd.DataFrame(data=ISOdata, columns=columns)
        try:
            ISOdata["y"] = y
        except:
            pass
        ISOdata.to_csv(
            "{}ISO_{}NN_{}d.csv".format(args.output_path, args.n, args.d),
            index=False,
        )

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    ISOdata.iloc[:, 0],
                    ISOdata.iloc[:, 1],
                    c=ISOdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via Isomap - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/Isomap_{}NN_2d.png".format(
                        args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    ISOdata.iloc[:, 0],
                    ISOdata.iloc[:, 1],
                    ISOdata.iloc[:, 2],
                    c=ISOdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via Isomap - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/Isomap_{}NN_3d.png".format(
                        args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

        embeddingLEM = SpectralEmbedding(n_neighbors=args.n,
                                         n_components=args.d,
                                         gamma=args.gamma)
        LEMdata = embeddingLEM.fit_transform(df)
        LEMdata = pd.DataFrame(data=LEMdata, columns=columns)
        try:
            LEMdata["y"] = y
        except:
            pass
        LEMdata.to_csv(
            "{}LEM_{}NN_{}d.csv".format(args.output_path, args.n, args.d),
            index=False,
        )

        if args.visualize == True and is_label == True:

            if args.d == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    LEMdata.iloc[:, 0],
                    LEMdata.iloc[:, 1],
                    c=LEMdata.iloc[:, 2],
                    cmap=plt.cm.Spectral,
                )
                plt.title("2d Embedding of Data via LEM - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/LEM_{}NN_2d.png".format(args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            elif args.d == 3:

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    LEMdata.iloc[:, 0],
                    LEMdata.iloc[:, 1],
                    LEMdata.iloc[:, 2],
                    c=LEMdata.iloc[:, 3],
                    cmap=plt.cm.Spectral,
                )
                plt.title("3d Embedding of Data via LEM - {}-NN".format(
                    args.n))
                plt.savefig(
                    "{}plots/LEM_{}NN_3d.png".format(args.output_path, args.n),
                    dpi=300,
                    transparent=True,
                )

            else:
                print("Visualization for d=2 or d=3 only")

        elif args.visualize == True and is_label == False:
            print("Visualization failed; requires a label")

    else:
        print("Error: Method not correctly specified")


if __name__ == "__main__":
    fit_embedding(args)
