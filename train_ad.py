import argparse
import transformations as ts
import opt_tc as tc
import numpy as np
from data_loader import Data_Loader

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(args, trans):
    dl = Data_Loader()
    x_train, x_test, y_test = dl.get_dataset(args.dataset,
                                             true_label=args.class_ind,
                                             flip_ones_and_zeros=args.flip)
    print("Computing transformed data for train data")
    x_train_trans, labels = transform_data(x_train, trans)
    print("Computing transformed data for test data")
    x_test_trans, _ = transform_data(x_test, trans)
    x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    y_test = np.array(y_test) == args.class_ind
    return x_train_trans, x_test_trans, y_test


def train_anomaly_detector(args):
    transformer = ts.get_transformer(args.type_trans)
    x_train, x_test, y_test = load_trans_data(args, transformer)
    print("Data fully loaded, using:")
    print(f"{x_train.shape} as training data")
    print(f"{x_test.shape} as test data")
    print(f"{y_test.shape} as test labels")
    num_of_real_labels_in_test_set = sum(y_test == 0)
    num_of_fake_labels_in_test_set = sum(y_test == 1)
    print(f"{num_of_real_labels_in_test_set} of normal labels in dataset")
    print(f"{num_of_fake_labels_in_test_set} of fake labels in dataset")
    ratio = 100.0 * num_of_real_labels_in_test_set / (num_of_real_labels_in_test_set + num_of_fake_labels_in_test_set)
    tc_obj = tc.TransClassifier(transformer.n_transforms, args)
    tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=288, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='complicated', type=str)

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--flip', help='flip zeros and ones at test set',
                        action='store_true')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        for i in range(10):
            args.class_ind = i
            print("Dataset: CIFAR10")
            print("True Class:", args.class_ind)
            train_anomaly_detector(args)
    else:
        print(f"Dataset: {args.dataset}")
        print("True Class:", args.class_ind)
        print(args)
        train_anomaly_detector(args)

