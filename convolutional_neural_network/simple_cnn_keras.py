from tensorflow import keras
from tensorflow.keras import layers

# 假设用于分类任务，共有10个类别
num_classes = 10
# 输入图像的形状: (height, width, channels)
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        # Conv2D 的 3 个参数分别为: 卷积核数量、每个卷积核的大小、激活函数
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
)

print(model.summary())