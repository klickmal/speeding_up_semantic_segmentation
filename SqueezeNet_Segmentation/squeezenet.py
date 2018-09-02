import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Concatenate, Activation, add
from keras.layers import Convolution2D, MaxPooling2D, Convolution2DTranspose, BatchNormalization
from L2_Normalization import L2Normalization

class squeeze_segNet():

    def __init__(self, n_labels, image_shape, weights_path):
        self.n_labels = n_labels
        self.image_shape = image_shape
        self.pretrain_weights_path = weights_path
        pass

    def fire_module(self, x, filters, name="fire"):
        sq_filters, ex1_filters, ex2_filters = filters
        squeeze = Convolution2D(sq_filters, (1, 1), activation='elu', padding='same', name=name + "_squeeze1x1")(x)
        expand1 = Convolution2D(ex1_filters, (1, 1), activation='elu', padding='same', name=name + "_expand1x1")(squeeze)
        expand2 = Convolution2D(ex2_filters, (3, 3), activation='elu', padding='same', name=name + "_expand3x3")(squeeze)
        x = Concatenate(axis=-1, name=name)([expand1, expand2])
        return x

    def squeeze_net(self, x):
        
        x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="elu", name='conv1')(x)
        x_low1 = x

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="same")(x)

        x = self.fire_module(x, (16, 64, 64), name="fire2")
        x = self.fire_module(x, (16, 64, 64), name="fire3")

        x_low2 = x
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="same")(x)

        x = self.fire_module(x, (32, 128, 128), name="fire4")
        x = self.fire_module(x, (32, 128, 128), name="fire5")

        x_low3 = x
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="same")(x)

        x = self.fire_module(x, (48, 192, 192), name="fire6")
        x = self.fire_module(x, (48, 192, 192), name="fire7")

        x = self.fire_module(x, (64, 256, 256), name="fire8")
        x = self.fire_module(x, (64, 256, 256), name="fire9")

        return x, x_low1, x_low2, x_low3

    def paral_dilat_module(self, x, n_filter=128):
        
        x1 = Convolution2D(64, kernel_size=(3,3), dilation_rate=(1,1), activation='elu', kernel_initializer='he_normal', padding='same', name='dilat1')(x)
        x2 = Convolution2D(64, kernel_size=(3,3), dilation_rate=(3,3), activation='elu', kernel_initializer='he_normal', padding='same', name='dilat2')(x)
        x3 = Convolution2D(64, kernel_size=(3,3), dilation_rate=(5,5), activation='elu', kernel_initializer='he_normal', padding='same', name='dilat3')(x)
        x4 = Convolution2D(64, kernel_size=(3,3), dilation_rate=(7,7), activation='elu', kernel_initializer='he_normal', padding='same', name='dilat4')(x)

        x_sum = add([x1, x2, x3, x4])

        return x_sum
    
    def conv_trans_block(self, x, out_filters, name_trans):

        x = Convolution2DTranspose(64, (1, 1), activation='elu', padding='same', kernel_initializer='he_normal', name=name_trans+'_tran1')(x)
        x = Convolution2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', kernel_initializer='he_normal', name=name_trans+'_tran2')(x)
        x = Convolution2DTranspose(out_filters, (1, 1), activation='elu', padding='same', kernel_initializer='he_normal', name=name_trans+'_tran3')(x)

        return x 
    
    def refine_block(self, x, x_low, refine_name):
        x = Convolution2D(64, kernel_size=(3,3), activation='elu', kernel_initializer='he_normal', padding='same', name=refine_name+'_block1')(x)
        #x = BatchNormalization()(x)
        x = L2Normalization(gamma_init=20, name=refine_name+'l2_norm1')(x)

        x_low = Convolution2D(64, kernel_size=(3,3), activation='elu', kernel_initializer='he_normal', padding='same', name=refine_name+'_block2')(x_low)
        #x_low = BatchNormalization()(x_low)
        x_low = L2Normalization(gamma_init=20, name=refine_name+'l2_norm2')(x_low)

        print(f'shape of x, x_low: {x.shape, x_low.shape}')
        x_sum1 = add([x, x_low])

        return x_sum1

    def conv_transpose(self, x, x_low1, x_low2, x_low3):
        x = self.conv_trans_block(x, 256, name_trans='tran1')
        x = self.refine_block(x, x_low3, refine_name='refine1')
        x = self.conv_trans_block(x, 128, name_trans='tran2')
        x = self.refine_block(x, x_low2, refine_name= 'refine2')
        x = self.conv_trans_block(x, 64, name_trans='tran3')
        x = self.refine_block(x, x_low1, refine_name='refine3')
        x = Convolution2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', kernel_initializer='he_normal', name='lasttran')(x)
        x = Convolution2D(self.n_labels, kernel_size=(1,1), activation='softmax', kernel_initializer='he_normal', padding='same', name='lastconv')(x)

        return x
        
    def init_model(self):
        h, w, d = self.image_shape

        input1 = Input(shape=(h,w,d), name='input')
        output1, x_low1, x_low2, x_low3 = self.squeeze_net(input1)
        #squeeze_net = Model(inputs=input1, outputs=[output1, x_low1, x_low2, x_low3], name='squeez_net')
        #squeeze_net.load_weights(self.pretrain_weights_path, by_name=True)
        #output2, x_low1, x_low2, x_low3 = squeeze_net.output

        output_3 = self.paral_dilat_module(output1)
        result = self.conv_transpose(output_3, x_low1, x_low2, x_low3)

        squeeze_seg = Model(inputs=input1, outputs=result, name='squeeze_seg')
        opt = keras.optimizers.RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
        squeeze_seg.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return squeeze_seg








    