import tensorflow as tf
import glob
from log_reg_exc import load_exchange
from explore_tensorflow import train
import argparse
import sys

FLAGS = None

def main(options):

    exchanges = [file_name.split('.csv')[0].split('\\')[1] for file_name in glob.glob(FLAGS.data_dir+'*.csv')]

    for exc in exchanges:
        print ('Building model for {:s}'.format(exc))
        save_files_to = FLAGS.data_dir + 'models/' + exc +'/'
        if tf.gfile.Exists(save_files_to):
            tf.gfile.DeleteRecursively(save_files_to)
        tf.gfile.MakeDirs(save_files_to)

<<<<<<< HEAD
=======
        # dewald 
>>>>>>> refs/remotes/origin/AidanHelmboldTBD-patch-1
        X_train, X_test, y_train, y_test = load_exchange(FLAGS.data_dir + exc, x_sum=False, y_loc=True)

        models = {}
        models[exc] = train(X_train, X_test, y_train, y_test, 
            model_name=exc, 
            n_step=FLAGS.n_step, 
            learning_rate = FLAGS.learning_rate,
            save_file=save_files_to)
    print ('ave final accuracy score: {:f}'.format((1.0 * sum(v for v in models.values()))/ len(models.values())))
        
if __name__ == '__main__':

    options = { 'max_steps' : 1000,
                'n_step'    : 100,
                'learning_rate' : 0.01,
                'data_dir' : 'Exchanges/'
                }

    parser = argparse.ArgumentParser()
    for k, v in options.items():
            parser.add_argument('--'+k, default=v, type=type(v))

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)