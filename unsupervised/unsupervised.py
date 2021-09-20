import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, TimeDistributed
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Cropping2D, ZeroPadding2D, Reshape
from tensorflow.keras.layers import LSTM, Masking

from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dropout, Flatten
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

from unsupervised_utils import sampling_gaussian, reconstruction_loss, kl_divergence, repeat_vector


def ChannelTSNE(encoder, test_data, dimension=2, batch_size=512):
    """
    t-SNE dimensionality reduction for visualization purposes

    Inputs
        encoder: func or 2D array
            Preferred method would be to pass in the encoder function along 
            with the test data to extract the latent space. Alternatively, 
            the latent space mean or variance can be passed in for direct plotting
            as (samples, features)
        test_data: array
            array for the test data that matches the encoder input dimensionality
        dimension: int
            Dimension desired for the t-SNE output. This is typically 2 for 
            visualization
        batch_size: int
            Number of samples to utilize at a time when predicting the encoder output

    Outputs
        x_latent: array
            Dimension reduced data of the latent space in the shape of (samples, dimension)
    """

    # Retrieve latent space mean for display
    # Variance is useful for understanding, not display
    try:
        z_mu, _ = encoder.predict(test_data, batch_size=batch_size)
        x_latent = TSNE(n_components=dimension, verbose=1).fit_transform(z_mu)
    except ValueError:
        z_mu = encoder
        x_latent = TSNE(n_components=dimension, verbose=1).fit_transform(z_mu)

    return x_latent


def ChannelKernelPCA(encoder, test_data, ker='rbf', gam=None, deg=3, dimension=2, batch_size=512):
    """
    Kernel PCA dimensionality reduction for visualization purposes

    Inputs
        encoder: func or 2D array
            Preferred method would be to pass in the encoder function along 
            with the test data to extract the latent space. Alternatively, 
            the latent space mean or variance can be passed in for direct plotting
            as (samples, features)
        test_data: array
            array for the test data that matches the encoder input dimensionality
        ker: str
            Name of the kernel method of choice for PCA
        gam: float
            Kernel coefficient when using non-linear kernel methods
        deg: int
            Degree of the non-linear kernel methods used
        dimension: int
            Dimension desired for the t-SNE output. This is typically 2 for 
            visualization
        batch_size: int
            Number of samples to utilize at a time when predicting the encoder output

    Outputs
        x_latent: array
            Dimension reduced data of the latent space in the shape of (samples, dimension)
    """

    # Retrieve latent space mean for display
    # Variance is useful for understanding, not display
    try:
        z_mu, _ = encoder.predict(test_data, batch_size=batch_size)
        x_latent = KernelPCA(n_components=dimension, kernel=ker, gamma=gam, degree=deg).fit_transform(z_mu)
    except ValueError:
        z_mu = encoder
        x_latent = KernelPCA(n_components=dimension, kernel=ker, gamma=gam, degree=deg).fit_transform(z_mu)

    return x_latent


def ChannelVAE(r_loss=reconstruction_loss, kl_div=kl_divergence, mask_value=-1,
               original_dim=70, intermediate_dim=256, latent_dim=15, alpha=1, beta=1):
    """
    Full implementation of Linear Variational Autoencoder for time series data. 
    Returns VAE, encoder, and decoder
    Trains on 2D data where the input is (samples, features) and reconstructs the full sequence

    Inputs
        r_loss: func
            Callable reconstruction loss method to allow users to implement alternative methods
            Default is utilizing binary cross entropy
        kl_div: func
            Callable kl divergence method with the default assuming unit gaussian prior for 
            the latent space
        mask_value: int
            Value to be ignored if padding was incorporated in pre-processing
        original_dim: int
            Features in the dataset
        intermediate_dim: int
            Number of units of the hidden layer in the network
        latent_dim: int
            Dimension of the latent space in the network
        alpha: int
            Tunable parameter regarding the weight of the reconstruction loss
        beta: int
            Tunable parameter regarding the weight of the kl divergence

    Outputs
        vae: Model
            Full model including encoder and decoder of the Variational Autoencoder with 
            inputs of (batch, features) and outputs the reconstruction
        encoder: Model
            Encoder portion of the VAE taking in the original input data and outputting 
            the mean and log variance of the latent space 
        decoder: Model
            Decoder portion of the VAE taking in a sample from the latent space distribution
            and outputting a reconstruction with the same dimension as the input data
    """

    x = Input(shape=original_dim)
    x_mask = Masking(mask_value=mask_value)(x)
    h = Dense(intermediate_dim, activation='elu')(x_mask)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling_gaussian, output_shape=(
        latent_dim,))([z_mu, z_log_var])

    encoder = Model(x, [z_mu, z_log_var])

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='elu'),
        Dense(original_dim)
    ])

    x_pred = decoder(z)

    vae = Model(inputs=[x], outputs=x_pred)
    loss = alpha*r_loss(x, x_pred, ax=-1) + beta*kl_div(z_mu, z_log_var)
    vae.add_loss(loss)
    print(vae.summary())

    return vae, encoder, decoder


def ChannelLSTMAE(original_dim=70, timesteps=None, intermediate_dim=15, drop=0.0, mask_value=-1, reg=0.001):
    """
    Full implementation of LSTM Autoencoder for time series data. 
    Returns autoencoder
    Trains on 3D data where the input is (samples, timesteps, features) and reconstructs the full sequence

    Inputs
        original_dim: int
            Features in the dataset
        timesteps: None or int
            Number of samples per sequence, None meaning they are not the same size 
            and an integer indicating each sequence is the same length
        intermediate_dim: int
            Number of units of the output of the hidden layer
        drop: float
            Percentage of dropout to be applied 
        mask_value: int
            Value to be ignored if padding was incorporated in pre-processing
        reg: float
            Degree of L2 regularization applied to the initial LSTM stage

    Outputs
        lstmae: Model
            Full model including encoder and decoder stages with the input 
            of (batch, timesteps, original_dim) and the outputs the reconstruction
    """

    x = Input(shape=(timesteps, original_dim))
    x_mask = Masking(mask_value=mask_value)(x)

    L1 = LSTM(original_dim, activation='elu', return_sequences=True,
              kernel_regularizer=tf.keras.regularizers.l2(l2=reg))(x_mask)
    L2 = LSTM(intermediate_dim, activation='elu', return_sequences=False)(L1)
    L2 = Dropout(drop)(L2)
    L3 = Lambda(repeat_vector)([x, L2])
    L4 = LSTM(intermediate_dim, return_sequences=True)(L3)
    L5 = LSTM(original_dim, activation='elu', return_sequences=True)(L4)
    L5 = Dropout(drop)(L5)
    output = TimeDistributed(Dense(original_dim))(L5)

    lstmae = Model(inputs=x, outputs=output)
    print(lstmae.summary())

    return lstmae


def ChannelLSTMVAE(loss_option=2, r_loss=reconstruction_loss, kl_div=kl_divergence, original_dim=70, timesteps=None,
                   intermediate_dim=50, latent_dim=15, alpha=1, beta=1, drop=0.0, mask_value=-1):
    """
    Full implementation of LSTM Variational Autoencoder for time series data. 
    Returns LSTM-VAE
    Trains on 3D data where the input is (samples, timesteps, features) and reconstructs the full sequence

    Inputs
        loss_option: int
            Parameter to pass to the reconstruction loss method to determine which loss is selected
        r_loss: func
            Callable reconstruction loss method to allow users to implement alternative methods
            Default is utilizing binary cross entropy
        kl_div: func
            Callable kl divergence method with the default assuming unit gaussian prior for 
            the latent space
        original_dim: int
            Features in the dataset
        timesteps: None or int
            Number of samples per sequence, None meaning they are not the same size 
            and an integer indicating each sequence is the same length
        intermediate_dim: int
            Number of units of the output of the hidden layer
        latent_dim: int
            Dimension of the latent space in the network
        alpha: int
            Tunable parameter regarding the weight of the reconstruction loss
        beta: int
            Tunable parameter regarding the weight of the kl divergence
        drop: float
            Percentage of dropout to be applied 
        mask_value: int
            Value to be ignored if padding was incorporated in pre-processing

    Outputs
        lstmvae: Model
            Full model including encoder and decoder stages with the input 
            of (batch, timesteps, original_dim) and the outputs the reconstruction
    """

    x = Input(shape=(timesteps, original_dim))
    x_mask = Masking(mask_value=mask_value)(x)

    h = LSTM(original_dim, activation='tanh', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2())(x_mask)
    h = LSTM(intermediate_dim, activation='tanh', return_sequences=False)(h)
    h = Dropout(drop)(h)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling_gaussian, output_shape=(
        latent_dim,))([z_mu, z_log_var])

    out = Lambda(repeat_vector)([x, z])
    out = LSTM(intermediate_dim, return_sequences=True)(out)
    out = LSTM(original_dim, activation='tanh', return_sequences=True)(out)
    out = Dropout(drop)(out)
    x_pred = TimeDistributed(Dense(original_dim))(out)

    lstmvae = Model(inputs=[x], outputs=x_pred)
    r = alpha*r_loss(x, x_pred, option=loss_option)
    kl = beta*kl_div(z_mu, z_log_var)
    loss = r + kl
    lstmvae.add_loss(tf.keras.backend.mean(loss))
    print(lstmvae.summary())

    return lstmvae


def ConvAE(mask_value=-1, channels=70, samples=1500, filters=8, kernel=3):
    """
    Convolutional implementation of Autoencoder for time series data. 
    Returns AE, encoder, and decoder
    Trains on 4D data where the input is (batch, channels, samples, 1) and reconstructs the full sequence

    Inputs
        mask_value: int
            Value to be ignored if padding was incorporated in pre-processing
        channels: int
            Features in the dataset
        samples: int
            Number of samples per sequence instance
        filters: int
            number of filters for the final reduction layer, there will be multiples of these 
            filters for the previous layers. This concept is mirrored for the expansion layers. 
            The number of filters reduces to shrink the size of the dataset to create a bottleneck 
            in the network to enforce learning
        kernel: int
            the kernel, given this being time series data, is only applied along one of the dimensions. 
            This value is the size of that kernel

    Outputs
        ae: Model
            Full model including encoder and decoder of the Autoencoder with 
            inputs of (batch, channels, samples, 1) and outputs the reconstruction
        encoder: Model
            Encoder portion of the AE taking in the original input data and outputting 
            the mean and log variance of the latent space 
        decoder: Model
            Decoder portion of the AE taking in a sample from the latent space distribution
            and outputting a reconstruction with the same dimension as the input data
    """

    x = Input((channels, samples, 1))
    x_pad = ZeroPadding2D((2, 2))(x)
    x_mask = Masking(mask_value=mask_value)(x_pad)
    h1 = Conv2D(filters*3, (1, kernel), strides=2, padding='same')(x_mask)
    h1 = BatchNormalization()(h1)
    h1 = Activation('tanh')(h1)
    h2 = Conv2D(filters*2, (1, kernel), strides=2, padding='same')(h1)
    h2 = BatchNormalization()(h2)
    h2 = Activation('tanh')(h2)
    h3 = Conv2D(filters, (1, kernel), strides=2, padding='same')(h2)
    h3 = BatchNormalization()(h3)
    h3 = Activation('tanh')(h3)

    encoder = Model(x, h3)

    decoder = Sequential([
        Conv2DTranspose(filters, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(filters*2, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(filters*3, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(1, (1, kernel), padding='same'),
        Cropping2D(((3, 3), (2, 2)))
    ])

    x_pred = decoder(h3)

    ae = Model(inputs=[x], outputs=x_pred)
    print(ae.summary())

    return ae, encoder, decoder


def ConvVAE(loss_option=2, r_loss=reconstruction_loss, kl_div=kl_divergence, mask_value=-1,
            channels=70, samples=1500, filters=8, kernel=3, latent_dim=15, alpha=1, beta=1):
    """
    Convolutional implementation of Variational Autoencoder for time series data. 
    Returns CNN-VAE, encoder, and decoder
    Trains on 4D data where the input is (batch, channels, samples, 1) and reconstructs the full sequence

    Inputs
        loss_option: int
            Parameter to pass to the reconstruction loss method to determine which loss is selected
        r_loss: func
            Callable reconstruction loss method to allow users to implement alternative methods
            Default is utilizing binary cross entropy
        kl_div: func
            Callable kl divergence method with the default assuming unit gaussian prior for 
            the latent space
        mask_value: int
            Value to be ignored if padding was incorporated in pre-processing
        channels: int
            Features in the dataset
        samples: int
            Number of samples per sequence instance
        filters: int
            number of filters for the final reduction layer, there will be multiples of these 
            filters for the previous layers. This concept is mirrored for the expansion layers. 
            The number of filters reduces to shrink the size of the dataset to create a bottleneck 
            in the network to enforce learning
        kernel: int
            the kernel, given this being time series data, is only applied along one of the dimensions. 
            This value is the size of that kernel
        latent_dim: int
            Dimension of the latent space in the network
        alpha: int
            Tunable parameter regarding the weight of the reconstruction loss
        beta: int
            Tunable parameter regarding the weight of the kl divergence

    Outputs
        vae: Model
            Full model including encoder and decoder of the Variational Autoencoder with 
            inputs of (batch, channels, samples, 1) and outputs the reconstruction
        encoder: Model
            Encoder portion of the VAE taking in the original input data and outputting 
            the mean and log variance of the latent space 
        decoder: Model
            Decoder portion of the VAE taking in a sample from the latent space distribution
            and outputting a reconstruction with the same dimension as the input data
    """

    x = Input((channels, samples, 1))
    x_mask = Masking(mask_value=mask_value)(x)
    x_pad = ZeroPadding2D((2, 2))(x_mask)
    h1 = Conv2D(filters*3, (1, kernel), strides=2, padding='same')(x_pad)
    h1 = BatchNormalization()(h1)
    h1 = Activation('tanh')(h1)
    h2 = Conv2D(filters*2, (1, kernel), strides=2, padding='same')(h1)
    h2 = BatchNormalization()(h2)
    h2 = Activation('tanh')(h2)
    h3 = Conv2D(filters, (1, kernel), strides=2, padding='same')(h2)
    h3 = BatchNormalization()(h3)
    h3 = Activation('tanh')(h3)
    h = Flatten()(h3)
    h = Dense(latent_dim)(h)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling_gaussian, output_shape=(
        latent_dim,))([z_mu, z_log_var])

    encoder = Model(x, [z_mu, z_log_var])

    decoder = Sequential([
        Dense(((samples//8)+1)*(channels+2)//8*filters,
              input_dim=latent_dim, activation="elu"),
        Reshape(((channels+2)//8, (samples//8)+1, filters)),
        Conv2DTranspose(filters, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(filters*2, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(filters*3, (1, kernel), strides=2, padding='same'),
        BatchNormalization(),
        Activation('tanh'),
        Conv2DTranspose(1, (1, kernel), padding='same'),
        Cropping2D(((1, 1), (2, 2)))
    ])
    for layer in decoder.layers:
        print(layer.output_shape)

    x_pred = decoder(z)

    vae = Model(inputs=[x], outputs=x_pred)
    loss = alpha*r_loss(x, x_pred, option=loss_option) + \
        beta*kl_div(z_mu, z_log_var)
    vae.add_loss(loss)
    print(vae.summary())

    return vae, encoder, decoder
