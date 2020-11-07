from __future__ import print_function
from __future__ import absolute_import

from src.ku import applications as apps, model_helper as mh

from munch import Munch
import pandas as pd, numpy as np
import shutil
from os import path

from keras.layers import Input
from keras.models import Model

def test_validation_save_best_multiple_training_rounds():
    
    ids = pd.DataFrame(dict(a = np.arange(100), 
                            b = np.flip(np.arange(100))))
    ids = apps.get_train_test_sets(ids)

    X = Input(shape=(1,), dtype='float32')
    y = apps.fc_layers(X, name = 'head',
                       fc_sizes      = [5, 1],
                       dropout_rates = [0, 0],
                       batch_norm    = 0)
    model = Model(inputs=X, outputs=y)

    gen_params = Munch(batch_size   = 4,
                      data_path     = '',
                      input_shape   = (1,),
                      inputs_df     = ['a'],
                      outputs       = ['b'])

    helper = mh.ModelHelper(model, 'test_model', ids, 
                            loss       = 'MSE',
                            metrics    = ['mean_absolute_error'],
                            monitor_metric = 'val_mean_absolute_error',
                            multiproc  = False, workers = 2,
                            logs_root  = 'logs',
                            models_root= 'models',
                            gen_params = gen_params)

    print('Model name:', helper.update_name(test='on'))

    valid_gen = helper.make_generator(ids[ids.set == 'validation'], 
                                      shuffle     =  False)
    valid_gen.batch_size = len(valid_gen.ids)
    valid_gen.on_epoch_end()
    assert valid_gen.ids_index.batch_index.unique().size == 1

    helper.train(lr=1e-1, epochs=50, verbose=False, valid_in_memory=True);

    assert path.exists(helper.params.logs_root + '/' + helper.model_name())

    helper.load_model(); # best
    valid_best1 = helper.validate(verbose=1)

    helper.train(lr=1, epochs=10, verbose=False, valid_in_memory=True);

    # validate final model
    valid_res_fin = helper.validate(verbose=1)

    helper.load_model(); # best
    valid_best2 = helper.validate(verbose=1)

    if valid_res_fin['loss'] > valid_best1['loss']:
        assert valid_best1['loss'] == valid_best2['loss']

    y_pred = helper.predict(valid_gen)
    y_true = ids[ids.set=='validation'].b.values
    _, _, val_mae, _ = apps.rating_metrics(y_true, y_pred, show_plot=False);
    print('valid_best2', valid_best2)
    assert np.abs(val_mae - valid_best2['mean_absolute_error']) < 1e-2
    
    shutil.rmtree('logs')
    shutil.rmtree('models')