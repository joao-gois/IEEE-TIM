from .base_network import BaseNetwork
from .pb_base import PB_NILM
from .aux_functions import *

import tensorflow as tf
import keras.backend as K


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units, kernel_initializer='he_normal')
        self.V = tf.keras.layers.Dense(1, kernel_initializer='he_normal')

    def call(self, encoder_output, **kwargs):
        score = self.V(K.tanh(self.W(encoder_output)))

        attention_weights = K.softmax(score, axis=1)

        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W'       : self.W,
            'V'       : self.V,
        })
        return config


class AttentionRNN(BaseNetwork):

    def __init__(self, x, y, window_size,
                 model_name, model_dir, dropout_rate=0.5, use_callbacks=False, stop_patience=50,
                 batch_size=128, n_epochs=100, validation_split=0.15):

        super().__init__(x, y, window_size, model_name, model_dir,
                         use_callbacks=use_callbacks, stop_patience=stop_patience,
                         batch_size=batch_size, n_epochs=n_epochs, validation_split=validation_split)

        # Set network specifics
        self.dropout_rate = dropout_rate

        # Normalization variable initialization
        self.max_val_x, self.max_val_y = self.get_max()

        self.normalized_x = None
        self.normalized_y = None

    def get_max(self):
        """
        Get maximum value of X and Y
        """
        max_x = self.x.max()
        max_y = self.y.max()

        return max_x, max_y

    def normalize(self, array, max_val):
        """
        Returns array normalized (divided) by max value
        """
        #return array/max_val
        return array

    def denormalize(self, array, max_val):
        """
        Returns array multiplied by max value
        """
        return array * max_val

    def preprocessing(self):
        """
        Normalizes by maximum value, then builds the sliding windows
        """

        self.normalized_x = self.normalize(self.x, self.max_val_x)
        self.normalized_y = self.normalize(self.y, self.max_val_y)

        preprocessed_x = split_sequence(self.normalized_x, self.window_size)
        preprocessed_x = preprocessed_x.reshape(preprocessed_x.shape[0],
                                                preprocessed_x.shape[1],
                                                1)
        preprocessed_x = tf.convert_to_tensor(preprocessed_x, dtype=tf.float32)
        preprocessed_y = tf.convert_to_tensor(self.normalized_y[self.window_size:], dtype=tf.float32)

        return preprocessed_x, preprocessed_y
    
    
    def network_architecture(self):
        '''Creates the RNN_Attention module described in the paper
        '''
        model = tf.keras.models.Sequential()

        # 1D Conv
        model.add(tf.keras.layers.Conv1D(16, 4, activation="linear",
                                         input_shape=(self.window_size, 1), padding="same", strides=1))

        # Bi-directional LSTMs and attention layer
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, 
                                                                     stateful=False),merge_mode='concat'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, 
                                                                     stateful=False),merge_mode='concat'))
        model.add(AttentionLayer(units=128))
        
        # Fully Connected Layers
        model.add(tf.keras.layers.Dense(128, activation='tanh'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.stop_patience, restore_best_weights=True)
        callbacks = [early_stopping]

        return model, callbacks

    
    def fit(self):

        # Get the data preprocessed
        processed_x, processed_y = self.preprocessing()

        # Get the network object
        self.model, callbacks = self.network_architecture()

        if self.use_callbacks:
            self.model.fit(processed_x, processed_y,
                           epochs=self.n_epochs,
                           batch_size=self.batch_size,
                           validation_split=self.validation_split,
                           callbacks=callbacks,
                           verbose=1)
        else:
            self.model.fit(processed_x, processed_y,
                           epochs=self.n_epochs,
                           batch_size=self.batch_size,
                           validation_split=self.validation_split,
                           verbose=1)
    
    def predict(self, x, preprocess=True):

        if preprocess:
            x = split_sequence(x, self.window_size)
            x = x.reshape(x.shape[0], x.shape[1], 1)
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        y_pred = self.model.predict(x)
        return y_pred
    