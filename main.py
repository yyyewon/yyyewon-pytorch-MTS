from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import pandas as pd
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets


def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    # x_train과 x_test를 numpy 배열로 변환
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # y_train과 y_test를 numpy 배열로 변환하고, float 타입으로 변경
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save original y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)





def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


# main

# change this directory for your machine
root_dir = '/Users/yewon/PycharmProjects/pytorch4'
archive_name = 'mts_Archive'
dataset_name = 'WalkvsRun_TRAIN'

import sys

# 명령줄 인수가 제대로 전달되었는지 확인
print("sys.argv:", sys.argv)
if len(sys.argv) < 2:
    print("Error: Not enough command line arguments provided.")
    print("Usage: python main.py <command>")
    print("Available commands: run_all, transform_mts_to_ucr_format, visualize_filter, viz_for_survey_paper, viz_cam, generate_results_csv")
    sys.exit(1)

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)
            archive_name = archive_name.lower()
            if archive_name == 'mtsarchive':
                archive_name = 'mts_archive'
            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'
                print(f"root_dir: {root_dir}")
                print(f"classifier_name: {classifier_name}")
                print(f"archive_name: {archive_name}")
                print(f"trr: {trr}")
                print(f"tmp_output_directory: {tmp_output_directory}")

                # archive_name을 소문자로 변환하여 사용
                archive_name = archive_name.lower()

                # 디버깅을 위한 출력 추가
                print(f"archive_name: {archive_name}")
                print(f"dataset_names_for_archive keys: {utils.constants.dataset_names_for_archive.keys()}")

                # for문에서 dataset_name을 찾을 때 정확히 키가 있는지 확인 후 사용
                if archive_name in utils.constants.dataset_names_for_archive:
                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                        print('\t\t\tdataset_name: ', dataset_name)

                        output_directory = tmp_output_directory + dataset_name + '/'

                        create_directory(output_directory)

                        fit_classifier()

                        print('\t\t\t\tDONE')

                        # the creation of this directory means
                        create_directory(output_directory + '/DONE')
                else:
                    print(f"Error: {archive_name} 키가 dataset_names_for_archive에 존재하지 않습니다.")


elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()

elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)

elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)

elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)

elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())

else:
    if len(sys.argv) < 5:
        print("Error: Not enough arguments for running an experiment.")
        print("Usage: python main.py <archive_name> <dataset_name> <classifier_name> <iteration>")
        sys.exit(1)

    # 인수에 따른 실험 실행 코드
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')



