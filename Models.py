import tensorflow as tf
from keras.layers import Lambda, Dropout
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.applications import VGG16, ResNet50V2,DenseNet121
import keras.backend as K

VGG="VGG"
RESNET="RESNET"
DENSNET="DENSNET"


class Models:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def create_model(self,model,opt):
        if model == DENSNET:
            return self.create_DenseNet121_model(opt)
        if model == VGG:
            return self.create_vgg16_model(opt)
        if model == RESNET:
            return self.create_resnet50_model(opt)
    def create_vgg16_model(self, opt):
        inputs = Input(shape=self.input_shape)

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling="max")

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model(inputs)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def create_resnet50_model(self, opt):
        inputs = Input(shape=self.input_shape)

        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling="max")

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model(inputs)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def create_DenseNet121_model(self, opt):
        inputs = Input(shape=self.input_shape)

        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling="max")

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model(inputs)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        return model