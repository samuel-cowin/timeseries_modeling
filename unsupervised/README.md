# Example on how to use the models with data

## Define the model and set hyperparameters
```python 
cvae, encVAE, decVAE = ConvVAE(loss_option=1, filters=16, kernel=3, \
                                  latent_dim=5, samples=1500, alpha=1000, beta=1)
cvae.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00001, rho=0.95, epsilon=None, decay=0.0))
```

## Train the model
```python
cvae_history = cvae.fit(x_t, x_t, shuffle=True, epochs=100, batch_size=10, verbose=1)
```
