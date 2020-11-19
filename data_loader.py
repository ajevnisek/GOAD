import scipy.io
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import os
import cv2


class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains
        self.urls = [
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
        ]

    def norm_kdd_data(self, train_real, val_real, val_fake, cont_indices):
        symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
        mus = train_real[:, cont_indices].mean(0)
        sds = train_real[:, cont_indices].std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            bin_cols = xs[:, symb_indices]
            cont_cols = xs[:, cont_indices]
            cont_cols = np.array([(x - mu) / sd for x in cont_cols])
            return np.concatenate([bin_cols, cont_cols], 1)

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake


    def norm_data(self, train_real, val_real, val_fake):
        mus = train_real.mean(0)
        sds = train_real.std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            return np.array([(x - mu) / sd for x in xs])

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self, dataset_name, c_percent=None, true_label=1,
                    flip_ones_and_zeros=False):
        if dataset_name == 'cifar10':
            return self.load_data_CIFAR10(true_label)
        if dataset_name == 'kdd':
            return self.KDD99_train_valid_data()
        if dataset_name == 'kddrev':
            return self.KDD99Rev_train_valid_data()
        if dataset_name == 'thyroid':
            return self.Thyroid_train_valid_data()
        if dataset_name == 'arrhythmia':
            return self.Arrhythmia_train_valid_data()
        if dataset_name == 'ckdd':
            return self.contaminatedKDD99_train_valid_data(c_percent)
        if dataset_name == 'faces':
            return self.load_data_faces(flip_ones_and_zeros)

    @staticmethod
    def get_face_crops_as_np_array(face_crops_dirs_path,
                                   norm_func):
        face_crops_dirs = os.listdir(face_crops_dirs_path)
        face_crops_dirs.sort(key=lambda x: int(x))
        all_frames = []
        video_dir_split_indices = []
        counter = 0
        for face_crops_dir in face_crops_dirs:
            dir_full_name = os.path.join(face_crops_dirs_path,
                                         face_crops_dir)
            for frame in os.listdir(dir_full_name):
                frame_path = os.path.join(dir_full_name, frame)
                image = cv2.imread(frame_path)
                resized_image = cv2.resize(image, (32, 32))
                all_frames.append(np.asarray(norm_func(resized_image),
                                             dtype='float32'))
                counter += 1
            video_dir_split_indices.append(counter)
        return np.stack(all_frames), video_dir_split_indices


    def load_data_faces(self, flip_ones_and_zeros):
        root = './data/faces/face_crops'
        original_face_crops_path = os.path.join(root, 'original')
        manipulated_faces_crops_path = os.path.join(root, 'manipulated')

        original_face_crops, video_dir_split_indices = \
            self.get_face_crops_as_np_array(
            original_face_crops_path, self.norm)
        manipulated_face_crops, _ = self.get_face_crops_as_np_array(
            manipulated_faces_crops_path, self.norm)

        num_of_original_frames, height, width, channels = \
            original_face_crops.shape
        # find the closest split to 65% of samples belonging to the original
        # video frames.
        num_of_training_samples_index = np.argmin([abs(index - int(
            num_of_original_frames * 0.65)) for index in
                                             video_dir_split_indices])
        num_of_training_samples = video_dir_split_indices[
            num_of_training_samples_index]

        print(f"Using {num_of_training_samples} samples for training.")
        x_train = original_face_crops[:num_of_training_samples]
        original_samples_for_test = original_face_crops[
                                num_of_training_samples:]
        x_test = np.concatenate([original_samples_for_test,
                                manipulated_face_crops])
        num_of_original_samples_for_test = original_samples_for_test.shape[0]
        num_of_manipulated_samples_for_test = manipulated_face_crops.shape[0]
        print(f"num_of_original_samples_for_test = {num_of_original_samples_for_test}")
        print(
            f"num_of_manipulated_samples_for_test = {num_of_manipulated_samples_for_test}")
        if not flip_ones_and_zeros:
            test_labels = np.concatenate([
                np.ones_like(range(num_of_original_samples_for_test)),
                np.zeros_like(range(num_of_manipulated_samples_for_test))])
        else:
            test_labels = np.concatenate([
                np.zeros_like(range(num_of_original_samples_for_test)),
                np.ones_like(range(num_of_manipulated_samples_for_test))])
        # randomly permute test samples:
        indices = np.random.permutation(len(x_test))
        x_test = x_test[indices]
        test_labels = test_labels[indices]
        return x_train, x_test, test_labels


    def load_data_CIFAR10(self, true_label):
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        trainset = dset.CIFAR10(root, train=True, download=True)
        train_data = np.array(trainset.data)
        train_labels = np.array(trainset.targets)

        testset = dset.CIFAR10(root, train=False, download=True)
        test_data = np.array(testset.data)
        test_labels = np.array(testset.targets)

        train_data = train_data[np.where(train_labels == true_label)]
        x_train = self.norm(np.asarray(train_data, dtype='float32'))
        x_test = self.norm(np.asarray(test_data, dtype='float32'))
        return x_train, x_test, test_labels


    def Thyroid_train_valid_data(self):
        data = scipy.io.loadmat("data/thyroid.mat")
        samples = data['X']  # 3772
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]  # 3679 norm
        anom_samples = samples[labels == 1]  # 93 anom

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 1839 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)


    def Arrhythmia_train_valid_data(self):
        data = scipy.io.loadmat("data/arrhythmia.mat")
        samples = data['X']  # 518
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]  # 452 norm
        anom_samples = samples[labels == 1]  # 66 anom

        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 226 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)


    def KDD99_preprocessing(self):
        df_colnames = pd.read_csv(self.urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
        df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
        df = pd.read_csv(self.urls[0], header=None, names=df_colnames['f_names'].values)
        df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
        df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
        samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])

        smp_keys = samples.keys()
        cont_indices = []
        for cont in df_continuous['f_names']:
            cont_indices.append(smp_keys.get_loc(cont))

        labels = np.where(df['status'] == 'normal.', 1, 0)
        import ipdb;ipdb.set_trace()
        return np.array(samples), np.array(labels), cont_indices


    def KDD99_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()
        anom_samples = samples[labels == 1]  # norm: 97278

        norm_samples = samples[labels == 0]  # attack: 396743

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)


    def KDD99Rev_train_valid_data(self):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        norm_samples = samples[labels == 1]  # norm: 97278

        # Randomly draw samples labeled as 'attack'
        # so that the ratio btw norm:attack will be 4:1
        # len(anom) = 24,319
        anom_samples = samples[labels == 0]  # attack: 396743

        rp = np.random.permutation(len(anom_samples))
        rp_cut = rp[:24319]
        anom_samples = anom_samples[rp_cut]  # attack:24319

        n_norm = norm_samples.shape[0]
        ranidx = np.random.permutation(n_norm)
        n_train = n_norm // 2

        x_train = norm_samples[ranidx[:n_train]]
        norm_test = norm_samples[ranidx[n_train:]]

        val_real = norm_test
        val_fake = anom_samples
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)


    def contaminatedKDD99_train_valid_data(self, c_percent):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        ranidx = np.random.permutation(len(samples))
        n_test = len(samples)//2
        x_test = samples[ranidx[:n_test]]
        y_test = labels[ranidx[:n_test]]

        x_train = samples[ranidx[n_test:]]
        y_train = labels[ranidx[n_test:]]

        norm_samples = x_train[y_train == 0]  # attack: 396743
        anom_samples = x_train[y_train == 1]  # norm: 97278
        n_contaminated = int((c_percent/100)*len(anom_samples))

        rpc = np.random.permutation(n_contaminated)
        x_train = np.concatenate([norm_samples, anom_samples[rpc]])

        val_real = x_test[y_test == 0]
        val_fake = x_test[y_test == 1]
        return self.norm_kdd_data(x_train, val_real, val_fake, cont_indices)


