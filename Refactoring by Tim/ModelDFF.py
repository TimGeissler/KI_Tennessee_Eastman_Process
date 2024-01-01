import tensorflow as tf

class ModelDFF:
    def __init__(self, learningRate, dynamicLearningRate=False):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(52,  activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(22,  activation=tf.nn.softmax))

        if (dynamicLearningRate):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learningRate,
                decay_steps=100,
                decay_rate=0.96
            )
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
         
