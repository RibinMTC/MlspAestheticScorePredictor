import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from ku import image_utils as img, applications as apps, model_helper as mh


class MlspModel:

    def __init__(self, root_ava_mlsp_path):
        self.__init_tensorflow_to_avoid_cudnn_error()
        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        self.preprocessor, self.helper_predictor = self.__load_model(root_ava_mlsp_path)

    # Code for avoiding cudnnn internal state error
    def __init_tensorflow_to_avoid_cudnn_error(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

    def __load_model(self, root_ava_mlsp_path):
        with self.graph.as_default():
            with self.session.as_default():
                dataset = root_ava_mlsp_path + 'metadata/AVA_data_official_test.csv'
                images_path = root_ava_mlsp_path + 'images/'
                ids = pd.read_csv(dataset)

                # load base model
                model_name = 'mlsp_wide_orig'
                input_shape = (None, None, 3)
                model_base = apps.model_inceptionresnet_pooled(input_shape)
                pre = apps.process_input[apps.InceptionResNetV2]

                # MODEL DEF
                from keras.layers import Input, GlobalAveragePooling2D
                from keras.models import Model

                input_feats = Input(shape=(5, 5, 16928), dtype='float32')
                x = apps.inception_block(input_feats, size=1024)
                x = GlobalAveragePooling2D(name='final_GAP')(x)

                pred = apps.fc_layers(x, name='head',
                                      fc_sizes=[2048, 1024, 256, 1],
                                      dropout_rates=[0.25, 0.25, 0.5, 0],
                                      batch_norm=2)

                model = Model(inputs=input_feats,
                              outputs=pred)

                gen_params = dict(batch_size=1,
                                  data_path=images_path,
                                  process_fn=pre,
                                  input_shape=input_shape,
                                  inputs='image_name',
                                  outputs='MOS',
                                  fixed_batches=False)

                helper = mh.ModelHelper(model, model_name, ids,
                                        gen_params=gen_params)

                # load head model
                helper.load_model(model_name=root_ava_mlsp_path + 'models/irnv2_mlsp_wide_orig/model')

                # join base and head models
                helper.model = Model(inputs=model_base.input,
                                     outputs=model(model_base.output))

                return pre, helper

    def predict(self, img_path):
        model_input_img = img.read_image(img_path)
        return self.predict_from_frame(model_input_img)
        # with self.graph.as_default():
        #     with self.session.as_default():
        #         try:
        #         # load, pre-process it, and pass it to the model
        #             I = self.preprocessor(img.read_image(img_path))
        #             I = np.expand_dims(I, 0)
        #             I_score = self.helper_predictor.model.predict(I)
        #
        #             return I_score[0][0]
        #         except Exception as e:
        #             print("!!!!!!!!! The following exception occurred: " + str(e))
        #             return -1

    def predict_from_frame(self, frame):
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    I = self.preprocessor(frame)
                    I = np.expand_dims(I, 0)
                    I_score = self.helper_predictor.model.predict(I)

                    return I_score[0][0]
                except Exception as e:
                    print("!!!!!!!!! The following exception occurred: " + str(e))
