from tensorflow.keras.backend import random_normal, shape, int_shape, exp
from tensorflow.keras.backend import binary_crossentropy, square, mean
from tensorflow.keras.backend import ones_like, expand_dims, batch_dot
from tensorflow.keras.backend import sum as ksum


def sampling_gaussian(args):
    """
    Take in the mean and variance from the neural network 
    Lambda layer and sample from the augmented unit normal distribution

    Inputs
        args: 2D Tensor and 2D tensor
            Tensors incorporating the batch size within the first dimension 
            and the size of the latent space of the Variational Autoencoder 
            as the second dimension.

    Outputs
        aug_gauss: 2D Tensor
            If mean and log variance are properly passed, a unit gaussian is 
            augmented with the mean and log variance from the neural network 
    """

    try:
        # Extract normal distribution parameters
        z_mean, z_log_variance = args
        batch = shape(z_mean)[0]
        dimension = int_shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dimension))

        # Augment unit gaussian
        aug_gauss = z_mean + exp(0.5 * z_log_variance) * epsilon
        return aug_gauss

    except ValueError:
        print('Expected two arguments corresponding to mean and log variance')


def reconstruction_loss(y_true, y_pred, ax=(1,2), option=1):
    """
    Selectable reconstruction loss configuration for testing in 
    Variational Autoencoder

    Inputs
        y_true: 3D Tensor
            3D tensor with the first dimension corresponding to 
            the batch size and the other dimensions being the real data
        y_pred: 3D Tensor
            3D tensor with the first dimension being the batch size 
            and the remaining dimensions being the predicted data 
        ax: tuple(int)
            Axes that the reconstruction error will be taken over - 
            provide all axes that are not related to batch size
        option: int
            Decides whether to use the binary crossentropy as a proxy 
            for reconstruction loss or to use mean squared error

    Outputs
        : 2D Tensor
            For each sample in the batch, return the error between the real 
            and predicted data
    """

    if option==1:
        return ksum(binary_crossentropy(y_true, y_pred), axis=ax)
    else:
        return mean(square(y_true - y_pred), axis=ax)


def kl_divergence(z_mean, z_log_variance):
    """
    Compute method for KL divergence with a unit normal latent prior
    Used for determining proximity of the predicted encoder distribution to 
    the real one

    Inputs
        z_mean: 2D Tensor
            Tensor incorporating the batch size within the first dimension 
            and the size of the latent space of the Variational Autoencoder 
            as the second dimension.
        z_log_variance: 2D Tensor
            Tensor incorporating the batch size within the first dimension 
            and the size of the latent space of the Variational Autoencoder 
            as the second dimension.

    Outputs
        kl_loss: 2D Tensor
            For each item in the batch, the following equation is realized
            KL = -0.5(sum(log(sigma_i^2)+1)-sum(sigma_i^2)-sum(mu_i^2))
    """

    kl_int = 1.0 + z_log_variance - square(z_mean) - exp(z_log_variance)
    kl_loss = -0.5 * ksum(kl_int, axis=-1)

    return kl_loss


def repeat_vector(x):
    """
    Cuts into every time slice of input signals for variable length in order to expand the 
    reduced feature vector for reconstruction purposes

    Inputs
        x: tuple
            First item in the tuple is a 2D tensor of shape (batch, timesteps, original_dim)
            Second item in the tuple is a Tensor of shape (batch, latent_dim,) 

    Outputs
        : 2D Tensor
            Dot product of the latent dimension with the input shape dimensionality to match 
            needed output dimensionality
    """

    input_ = ones_like(x[0][:,:,:1]) 
    latent_ = expand_dims(x[1],axis=1)

    return batch_dot(input_, latent_)
