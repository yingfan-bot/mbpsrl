# neural bayesian framework for transition & cost model
import torch
import numpy as np
# import gpflow
# import gpytorch

import os
import time

# import tensorflow as tf
from datetime import datetime

from scipy.stats import invgamma

import pickle

import warnings

warnings.filterwarnings("ignore")


class neural_bays_dx_tf(object):
    def __init__(self, args, model, model_type, output_shape, device=None, train_x=None, train_y=None, sigma_n2=0.1,
                 sigma2=0.1):
        self.model = model
        self.model_type = model_type
        self.args = args
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.output_shape = output_shape
        self.hidden_dim = 2*model.layers[0].get_input_dim()
        # if model_type == "dx":
            # self.hidden_dim = args.hidden_dim_dx
        # if model_type == "cost":
            # self.hidden_dim = args.hidden_dim_cost
        self.beta_s = None
        self.latent_z = None

        # for update formula in BDQN
        self.sigma2 = sigma2  # W prior variance
        self.sigma_n2 = sigma_n2  # noise variacne
        self.eye = np.eye(self.hidden_dim)
        self.mu_w = np.random.normal(loc=0, scale=.01, size=(output_shape, self.hidden_dim))
        # self.cov_w = np.random.normal(loc = 0, scale = 1, size = (output_shape, self.hidden_dim, self.hidden_dim)) + args.var_scale * self.eye
        self.cov_w = np.array([self.sigma2 * np.eye(self.hidden_dim) for _ in range(output_shape)])


    def add_data(self, new_x, new_y):
        if self.train_x is None:
            self.train_x = new_x
            self.train_y = new_y
        else:
            self.train_x = np.vstack((self.train_x, new_x))
            self.train_y = np.vstack((self.train_y, new_y))

    def generate_latent_z(self):
        # Update the latent representation of every datapoint collected so far
        # if self.do_scaling:
        #   self.data_h.scale_contexts()
        # ctx = self.data_h.get_contexts(scaled=self.do_scaling)
        new_z = self.get_representation(self.train_x)

        self.latent_z = new_z

    def train(self, epochs = 5):
        self.model.train(self.train_x,self.train_y,epochs=epochs)

        # self.replace_latent_z()
        self.generate_latent_z()

    def get_representation(self, input):
        """
        Returns the latent feature vector from the neural network.
        This vector is called z in the Google Brain paper.
        """
        # if not isinstance(input, torch.Tensor):
        #     c = torch.tensor(input).float()
        # else:
        #     c = input.float()
        # z = self.model.get_representation(c)
        z = self.model.predict(input, layer = True)
        z = z.squeeze()

        return z

    def check_dim(self):
        print("prior to sampling, check dim as follows: ")
        if self.output_shape == 1:
            print("sampling from cost model")
        else:
            print("sampling from transition model")
        print("dim of mu: ", np.array(self.mu).shape)
        print("a = ", self.a)
        print("dim of a: ", np.array(self.a).shape)
        print("b = ", self.b)
        print("cov dim: ", np.array(self.cov).shape)


    def sample_BDQN(self, parallelize=False):
        # Sample sigma2, and beta conditional on sigma2
        d = self.mu_w[0].shape[0]  # hidden_dim
        beta_s = []
        try:

            for i in range(self.output_shape):
                mus = self.mu_w[i]
                covs = self.cov_w[i][np.newaxis, :, :]
                multivariates = np.random.multivariate_normal(mus, covs[0])
                beta_s.append(multivariates)

        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            # Todo: Fix This
            print('Details: {} | {}.'.format(e.message, e.args))
            multivariates = np.random.multivariate_normal(np.zeros((d)), np.eye(d))
            beta_s.append(multivariates)
        self.beta_s = np.array(beta_s)

    # for simple test
    def predict(self, x):
        # Compute last-layer representation for the current context
        z_context = self.get_representation(x)

        # z_context = z_context[np.newaxis, :]
        # Apply Thompson Sampling
        # vals = (self.beta_s * z_context ).sum(axis=-1)
        vals = (self.beta_s.dot(z_context.T))
        # return np.array(vals)
        if self.model_type == "dx":
            state = x[:vals.shape[0]] if len(x.shape) == 1 else x[:, :vals.shape[0]]
            return vals.T + state+ self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)
        return vals.T + self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[:self.output_shape]+np.random.normal(loc=0, scale=np.sqrt(self.sigma_n2),size = vals.T.shape)



    def update_bays_reg_BDQN(self):
        # self.train_x = torch.tensor(self.train_x).detach().clone().float().to(self.device)
        # self.train_y = torch.tensor(self.train_y).detach().clone().float().to(self.device)
        # self.generate_latent_z()

        for i in range(self.output_shape):
            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.latent_z
            y = self.train_y[:, i] - self.model.layers[len(self.model.layers)-1].biases.eval(session =self.model.sess).squeeze()[i]
            s = np.dot(z.T, z)

            # inv = np.linalg.inv((s/self.sigma_n + 1/self.sigma*self.eye))
            A = s / self.sigma_n2 + 1 / self.sigma2 * self.eye
            B = np.dot(z.T, y) / self.sigma_n2
            # A = torch.as_tensor(s/self.sigma_n + 1/self.sigma*self.eye)
            # B = torch.as_tensor(np.dot(z.T, y))/self.sigma_n
            reg_coeff = 0
            for _ in range(10):
                try:
                    # A = torch.as_tensor(A)+ reg_coeff * torch.DoubleTensor(self.eye)
                    # X, _ = torch.lstsq(B, A_p)
                    # A = A_p.numpy()
                    A = A + reg_coeff * self.eye
                    inv = np.linalg.inv(A)
                except Exception as e:
                    print(e)
                    reg_coeff += 10
                # Store new posterior distributions
                # self.mu_w[i] = np.dot(inv,np.dot(z.T, y))/sigma_n
                else:
                    self.mu_w[i] = inv.dot(B).squeeze()
                    self.cov_w[i] = inv
                    # only a scale
                    break

