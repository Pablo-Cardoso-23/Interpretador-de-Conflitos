import tensorflow as tf

class CriadorModeloEmocoes:
    """
    Construtor da Rede Neural Convolucional (CNN) para classificação de emoções faciais.
    Atualizado com Data Augmentation para combater o Overfitting.
    """
    def __init__(self, formato_entrada=(48, 48, 1), num_classes=7):
        self.formato_entrada = formato_entrada
        self.num_classes = num_classes

    def construir_modelo(self):
        modelo = tf.keras.Sequential()

        # Estas camadas só agem durante o treinamento. Elas modificam as imagens "em tempo real"
        # para que a IA não decore os dados.
        modelo.add(tf.keras.layers.RandomFlip("horizontal", input_shape=self.formato_entrada))
        modelo.add(tf.keras.layers.RandomRotation(0.1)) # Gira a imagem até 10%
        modelo.add(tf.keras.layers.RandomZoom(0.1))     # Aproxima ou afasta até 10%
        
        # 1. Camadas Convolucionais
        modelo.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # BatchNormalization ajuda a rede a treinar mais rápido e a estabilizar
        modelo.add(tf.keras.layers.BatchNormalization())
        modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
        modelo.add(tf.keras.layers.Dropout(0.3)) 

        modelo.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        modelo.add(tf.keras.layers.BatchNormalization())
        modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
        modelo.add(tf.keras.layers.Dropout(0.3))

        modelo.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        modelo.add(tf.keras.layers.BatchNormalization())
        modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
        modelo.add(tf.keras.layers.Dropout(0.4))

        # 2. Achatar os dados
        modelo.add(tf.keras.layers.Flatten())

        # 3. Camadas Densas (Cérebro da Decisão)
        modelo.add(tf.keras.layers.Dense(256, activation='relu'))
        modelo.add(tf.keras.layers.BatchNormalization())
        modelo.add(tf.keras.layers.Dropout(0.5))
        
        # Camada de Saída
        modelo.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        otimizador = tf.keras.optimizers.Adam(learning_rate=0.0005)

        modelo.compile(optimizer=otimizador,
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
        return modelo