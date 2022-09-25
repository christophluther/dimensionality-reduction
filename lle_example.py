# example script to fit LLE on a certain data set
# while adhering to our data format
import matplotlib.pyplot as plt
from sklearn.manifold import locally_linear_embedding
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='world')
    parser.add_argument('--input_path', type=str, default='circle_data/rawdata_circle.csv',
                        help='path to the input data')
    parser.add_argument('--output_path', type=str, default='example/lle_circles.csv',
                        help='path to the input data')
    parser.add_argument('--d', type=int, default=2,
                        help='dimensionality of the final embedding')
    return parser.parse_args()


def fit_embedding(args):
    # read data
    data = pd.read_csv(args.input_path)
    y = data['y']
    data.drop(['y'], axis=1, inplace=True)
    X = data.values

    # fit lle with output
    X_r, err = locally_linear_embedding(X=X,
                                        n_neighbors=12,
                                        n_components=args.d)
    print("Reconstruction error: %g" % err)

    # store the output in the required format
    output = pd.DataFrame(X_r)
    output.columns = [f'x_{idx}' for idx in range(X_r.shape[1])]
    output['y'] = y
    output_name = args.output_path.split('.csv')[0]
    output.to_csv(f'{output_name}_{args.d}d.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    fit_embedding(args)