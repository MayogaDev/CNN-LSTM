# model.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

"""def get_model(max_length_frames, output_length):
    # Definir la entrada del modelo
    inputs = Input(shape=(max_length_frames, LENGTH_KEYPOINTS), name="input_layer")
    
    # Primera capa LSTM
    lstm_1_units = 64
    lstm_1_return_sequences = True
    lstm_1_activation = 'relu'
    lstm_1_kernel_regularizer = l2(0.001)
    lstm_1 = LSTM(lstm_1_units, 
                  return_sequences=lstm_1_return_sequences, 
                  activation=lstm_1_activation, 
                  kernel_regularizer=lstm_1_kernel_regularizer,
                  name="lstm_layer_1")(inputs)
    
    # Segunda capa LSTM
    lstm_2_units = 128
    lstm_2_return_sequences = True
    lstm_2_activation = 'relu'
    lstm_2_kernel_regularizer = l2(0.001)
    lstm_2 = LSTM(lstm_2_units, 
                  return_sequences=lstm_2_return_sequences, 
                  activation=lstm_2_activation, 
                  kernel_regularizer=lstm_2_kernel_regularizer,
                  name="lstm_layer_2")(lstm_1)
    
    # Tercera capa LSTM
    lstm_3_units = 128
    lstm_3_return_sequences = False
    lstm_3_activation = 'relu'
    lstm_3_kernel_regularizer = l2(0.001)
    lstm_3 = LSTM(lstm_3_units, 
                  return_sequences=lstm_3_return_sequences, 
                  activation=lstm_3_activation, 
                  kernel_regularizer=lstm_3_kernel_regularizer,
                  name="lstm_layer_3")(lstm_2)
    
    # Primera capa densa
    dense_1_units = 64
    dense_1_activation = 'relu'
    dense_1_kernel_regularizer = l2(0.001)
    dense_1 = Dense(dense_1_units, 
                    activation=dense_1_activation, 
                    kernel_regularizer=dense_1_kernel_regularizer,
                    name="dense_layer_1")(lstm_3)
    
    # Segunda capa densa
    dense_2_units = 64
    dense_2_activation = 'relu'
    dense_2_kernel_regularizer = l2(0.001)
    dense_2 = Dense(dense_2_units, 
                    activation=dense_2_activation, 
                    kernel_regularizer=dense_2_kernel_regularizer,
                    name="dense_layer_2")(dense_1)
    
    # Tercera capa densa
    dense_3_units = 32
    dense_3_activation = 'relu'
    dense_3_kernel_regularizer = l2(0.001)
    dense_3 = Dense(dense_3_units, 
                    activation=dense_3_activation, 
                    kernel_regularizer=dense_3_kernel_regularizer,
                    name="dense_layer_3")(dense_2)
    
    # Capa de salida
    output_layer_units = output_length
    output_layer_activation = 'softmax'
    outputs = Dense(output_layer_units, 
                    activation=output_layer_activation,
                    name="output_layer")(dense_3)
    
    # Crear el modelo
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Model")
    
    # Configurar el optimizador
    optimizer_learning_rate = 0.001
    optimizer = Adam(learning_rate=optimizer_learning_rate)
    
    # Configurar la función de pérdida
    loss_function = CategoricalCrossentropy()
    
    # Compilar el modelo
    model.compile(optimizer=optimizer, 
                  loss=loss_function, 
                  metrics=['accuracy'])
    
    return model"""

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    #model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
"""def get_model(max_length_frames, output_length: int):
    model = Sequential()
    
    # Primera capa LSTM
    model.add(LSTM(64, return_sequences=True, input_shape=(max_length_frames, LENGTH_KEYPOINTS),kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))  # Sustitución de ReLU por LeakyReLU
    model.add(Dropout(0.3))
    
    # Segunda capa LSTM
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    
    # Capa densa
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    
    # Capa densa
    model.add(Dense(32, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    
    # Salida
    model.add(Dense(output_length, activation='softmax'))
    
    # Compilación del modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model"""
