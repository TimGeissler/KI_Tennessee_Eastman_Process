import tensorflow as tf

class ModelDFF:
    def __init__(self, learningRate, dynamicLearningRate=False, decayRate = 0.98, decaySteps=1000, dropout=0.0, numberOfHiddenLayers=2, neuronsPerHiddenLayer=128):
        firstHalfOfLayers = round(numberOfHiddenLayers/2)
        secondHalfOfLayers = numberOfHiddenLayers - firstHalfOfLayers

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(52,  activation=tf.nn.relu))

        for i in range(firstHalfOfLayers):
            self.model.add(tf.keras.layers.Dense(neuronsPerHiddenLayer, activation=tf.nn.relu))

        self.model.add(tf.keras.layers.Dropout(dropout))

        for k in range(secondHalfOfLayers):
            self.model.add(tf.keras.layers.Dense(neuronsPerHiddenLayer, activation=tf.nn.relu))

        self.model.add(tf.keras.layers.Dense(22,  activation=tf.nn.softmax))



        if (dynamicLearningRate):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learningRate,
                decay_steps=decaySteps,
                decay_rate=decayRate
            )
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
         
