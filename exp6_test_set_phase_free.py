import models
import utils_train
import utils
from config import *
import config
from scipy.signal import medfilt2d

import argparse
import os



def main(args):

    pth_model = args.pth_model
    save_path = args.save_path
    #list_of_files = args.list_of_files

    #audio_path = np.array(pd.read_csv(list_of_files, header=None))[0][0]
    #fname_list = np.array(pd.read_csv(list_of_files, header=None))[1:]

    save_key = 'exp3multif0'
    model_path = os.path.join(pth_model, "{}.pkl".format(save_key))

    model = models.build_model3()
    model.load_weights(model_path)

    model.compile(
        loss=utils_train.bkld, metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )

    print("Model compiled")

    data_splits_path = os.path.join(config.data_save_folder, 'data_splits.json')
    test_set = utils.load_json_data(data_splits_path)

    for fname in test_set['test']:

        #fname = fname[0]
        if not fname.endswith('.wav'): continue

        if 'rev_' in fname:
            # predict using trained model
            predicted_output, _, _ = utils_train.get_single_test_prediction_phase_free(model,
                                                                            npy_file=None,
                                                                            audio_file=os.path.join(
                                                                                config.audio_save_folder, 'reverb',
                                                                                                    fname[4:]))
        else:
            predicted_output, _, _ = utils_train.get_single_test_prediction_phase_free(model,
                                                                                       npy_file=None,
                                                                                       audio_file=os.path.join(
                                                                                           config.audio_save_folder, fname))

        predicted_output = medfilt2d(predicted_output, kernel_size=(1, 11))

        np.save(os.path.join(
            save_path, "{}_prediction.npy".format(fname.split('.')[0])),
            predicted_output.astype(np.float32))

        # get multif0 output from prediction
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:

            est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)

            for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                if any(fqs <= 0):
                    est_freqs[i] = np.array([f for f in fqs if f > 0])

            output_mf0 = os.path.join(save_path, "{}_{}.csv".format(fname.split('.')[0], thresh))
            utils_train.save_multif0_output(est_times, est_freqs, output_mf0)

            print("     Multiple F0 prediction exported for {}".format(fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output using model 3 with the phase set to 0. Experiment 6.")

    parser.add_argument("--model_path",
                        dest='pth_model',
                        type=str,
                        help="Path to the model weights.")

    parser.add_argument("--save_path",
                        dest='save_path',
                        type=str,
                        help="Folder to save predicted outputs.")



    main(parser.parse_args())