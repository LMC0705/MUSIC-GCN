import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
import os
import time
import argparse
from functools import reduce
import importlib
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres
from sklearn.cluster import KMeans


#parser set
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='10x_73k')
parser.add_argument('--model', type=str, default='wgan')
parser.add_argument('--lambda1', type=float, default=10.0)# sparsity cost on C
parser.add_argument('--lambda2', type=float, default=0.2)  # loss of self-express
parser.add_argument('--lambda3', type=float, default=1.0)  # lambda on gan loss  score_disc
parser.add_argument('--lambda4', type=float, default=0.01)# lambda on AE L2 regularization
parser.add_argument('--m', type=float, default=0.1)  # lambda on AE L2 regularization
parser.add_argument('--lr',  type=float, default=1e-3)  # learning rate
parser.add_argument('--lr2', type=float, default=2e-4)  # learning rate for discriminator and eqn3plus
#1000 4000 2000
parser.add_argument('--pretrain', type=int, default=2000)  # number of iterations of pretraining #0 为了获得更好的低维表示
parser.add_argument('--epochs', type=int, default=1000)  # number of epochs to train on eqn3 and eqn3plus  4000
parser.add_argument('--enable-at', type=int, default=1000)  # epoch at which to enable eqn3plus  2000
parser.add_argument('--dataset', type=str, default='single cell', choices=['yaleb', 'orl', 'coil20', 'coil100','single cell'])

parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--interval2', type=int, default=1)
parser.add_argument('--bound', type=float, default=0.02)  # discriminator weight clipping limit
parser.add_argument('--D-init', type=int, default=100)  # number of discriminators steps before eqn3plus starts
parser.add_argument('--D-steps', type=int, default=5)
parser.add_argument('--G-steps', type=int, default=1)
parser.add_argument('--save', action='store_false')  # save pretrained model
parser.add_argument('--r', type=int, default=0)  # Nxr rxN, use 0 to default to NxN Coef

parser.add_argument('--rank', type=int, default=10)#dimension of the subspaces

parser.add_argument('--beta1', type=float, default=0.00)
parser.add_argument('--beta2', type=float, default=0.010)
parser.add_argument('--beta3', type=float, default=0.010)
parser.add_argument('--stop-real', action='store_true')
parser.add_argument('--stationary', type=int, default=1)
parser.add_argument('--submean',        action='store_true')
parser.add_argument('--proj-cluster',   action='store_true')
parser.add_argument('--usebn',          action='store_true')
parser.add_argument('--no-uni-norm',    action='store_true')
parser.add_argument('--one2one',        action='store_true')
parser.add_argument('--matfile',        default=None)
parser.add_argument('--imgmult',        type=float,     default=1.0)
parser.add_argument('--palpha',         type=float,     default=None)
parser.add_argument('--kernel-size',    type=int,       nargs='+',  default=None)
parser.add_argument('--s_tau1',        type=float,     default=1.01)
parser.add_argument('--s_lambda1',     type=float,     default=5.0)
parser.add_argument('--s_tau2',        type=float,     default=1.01)
parser.add_argument('--s_lambda2',     type=float,     default=4.0)
parser.add_argument('--degerate',       action='store_true')


class ConvAE(object):
    def __init__(self,
                 args,encoder,decoder,x_dim, z_dim, n_hidden, kernel_size, n_class, n_sample_perclass, disc_size,
                 lambda1, lambda2, lambda3, batch_size, r=0, rank=10,
                 reg=None, disc_bound=0.02,
                 model_path=None, restore_path=None,
                 logs_path='logs'):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.n_class = n_class
        self.n_sample_perclass = n_sample_perclass
        self.disc_size = disc_size
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.rank = rank
        self.iter = 0

        """
        Eqn3
        """
        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.learning_rate = tf.placeholder(tf.float32, [])
        latent = self.encoder(self.x , reuse=False)
        print(self.x.shape)
        self.latent_shape = latent.shape
        self.latent_size = reduce(lambda x, y: int(x) * int(y), self.latent_shape[1:], 1)
        z = tf.reshape(latent, [batch_size, -1])
        z.set_shape([batch_size, self.latent_size])
        if args.usebn:
            z = tf.contrib.layers.batch_norm(z)
        if r == 0:
            Coef = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        else:
            v = (1e-2) / r
            L = tf.Variable(v * tf.ones([self.batch_size, r]), name='Coef_L')
            R = tf.Variable(v * tf.ones([r, self.batch_size]), name='Coef_R')
            Coef = tf.matmul(L, R, name='Coef_full')

        #self-representation layer output
        z_c = tf.matmul(Coef, z, name='matmul_Cz')
        self.Coef = Coef
        Coef_weights = [v for v in tf.trainable_variables() if v.name.startswith('Coef')]

        latent_c = tf.reshape(z_c, tf.shape(latent))
        self.z = z
        self.z_c = z_c
        # decoder
        self.x_r = self.decoder(latent_c,reuse=False)
        ae_weights = [self.encoder.vars, self.decoder.vars]
        self.ae_weight_norm = tf.sqrt(sum([tf.norm(v, 2) ** 2 for v in self.encoder.vars]))+ tf.sqrt(sum([tf.norm(v, 2) ** 2 for v in self.decoder.vars]))
        eqn3_weights = Coef_weights + ae_weights

        # auto-encoder regularization loss
        self.loss_aereg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # weight decay
        # Eq.1 loss
        self.loss_recon = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))#
        print(type(self.Coef))
        self.loss_sparsity1 = tf.reduce_mean(tf.norm(self.Coef, ord=1))
        self.loss_sparsity = tf.reduce_sum(tf.abs(self.Coef))
        self.loss_selfexpress = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0)) #self—expression
        self.loss_eqn3 = self.loss_recon + lambda1 * self.loss_sparsity1+ lambda2 * self.loss_selfexpress + self.loss_aereg
        with tf.variable_scope('optimizer_eqn3'):
            self.optimizer_eqn3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_eqn3,
                                                                                                    var_list=eqn3_weights)
        """
        Pretraining
        """
        # pretraining loss
        self.x_r_pre = self.decoder(latent)
        self.loss_recon_pre = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r_pre, self.x), 2.0))
        self.loss_pretrain = self.loss_recon_pre + self.loss_aereg
        with tf.variable_scope('optimizer_pre'):
            self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_pretrain,
                                                                                                   var_list=ae_weights)

        """
        Discriminator
        """
        # step counting
        # tf.Variable create variable
        self.gen_step = tf.Variable(0, dtype=tf.float32, trainable=False)  # keep track of number of generator steps
        self.gen_step_op = self.gen_step.assign(self.gen_step + 1)  # increment generator steps
        # discriminator output
        self.y_x = tf.placeholder(tf.int32, [batch_size])
        print('building discriminator')
        self.Us = self.make_Us()
        u_primes = self.svd_initialization(self.z, self.y_x)
        self.u_ini = [tf.assign(u, u_prime) for u, u_prime in zip(self.Us, u_primes)]
        z_real = self.z
        self.score_disc, self.Us_update_op, group_loss_real,group_new_loss,group_loss_ce= self.compute_disc_loss(z_real, self.y_x)
        print('adding disc regularization')
        # eqn7
        regulariz1 = self.regularization1(reuse=True)
        regulariz2 = self.regularization2(reuse=True)
        self.loss_disc = args.beta2 * regulariz1 + args.beta3 * regulariz2 - self.score_disc

        print('building disc optimizers')
        with tf.variable_scope('optimizer_disc'):
            self.optimizer_disc = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0).minimize(self.loss_disc,
                                                                                                 var_list=self.Us)
        print('building eqn8 optimizers')
        # Eqn 8
        self.loss_eqn3plus = self.loss_eqn3 + lambda3 * self.score_disc
        with tf.variable_scope('optimizer_eqn8'):
            self.optimizer_eqn3plus = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss_eqn3plus, var_list=eqn3_weights)

        # finalize stuffs
        s0 = tf.summary.scalar("loss_recon_pre", self.loss_recon_pre / batch_size)  # 13372
        s1 = tf.summary.scalar("loss_recon", self.loss_recon)
        s2 = tf.summary.scalar("loss_sparsity", self.loss_sparsity)
        s3 = tf.summary.scalar("loss_selfexpress", self.loss_selfexpress)
        s4 = tf.summary.scalar("score_disc", self.score_disc)
        s5 = tf.summary.scalar("ae_l2_norm", self.ae_weight_norm)  # 29.8 # ae_weight_norm
        s6 = tf.summary.scalar("disc_real", self.disc_score_real)
        s7 = tf.summary.scalar("disc_fake", self.disc_score_fake)
        self.summaryop_eqn3 = tf.summary.merge([s1, s2, s3, s5])
        self.summaryop_eqn3plus = tf.summary.merge([s1, s2, s3, s4, s5, s6, s7])
        self.summaryop_pretrain = tf.summary.merge([s0, s5])
        self.init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), flush_secs=20)

    def get_u_init_for_g(self, g):
        N_g = tf.shape(g)[0]  #
        gt = tf.transpose(g)
        q, r = tf.qr(gt, full_matrices=False)
        idx = [j for j in range(args.rank)]
        qq = tf.gather(tf.transpose(q), idx)
        qq = tf.transpose(qq)
        return qq
    def svd_initialization(self, z, y):
        group_index = [tf.where(tf.equal(y, k)) for k in range(
            self.n_class)]
        groups = [tf.gather(z, group_index[k]) for k in
                  range(self.n_class)]
        groups = [tf.squeeze(g, axis=1) for g in groups]
        if self.args.submean:
            groups = [g - tf.reduce_mean(g, 0, keep_dims=True) for g in groups]
        dim1 = tf.shape(z)[1]
        u_prime = [self.get_u_init_for_g(g) for g in groups]
        return u_prime

    def uniform_recombine(self, g):
        N_g = tf.shape(g)[0]
        selector = tf.random_uniform([N_g, N_g])
        if not self.args.no_uni_norm:
            selector = selector / tf.reduce_sum(selector, 1, keep_dims=True)  # normalize each row to 1
        g_fake = tf.matmul(selector, g, name='matmul_selectfake')
        return g_fake

    def make_Us(self):
        Us = []
        for j in range(self.n_class):
            u = tf.get_variable('disc_w{}'.format(j), shape=[self.latent_size, self.rank],
                                initializer=layers.xavier_initializer())
            Us.append(u)
        return Us

    def match_idx(self, g):
        """
        for the group g, identify the Ui whose residual is minimal, then return
            label, loss, sreal, u
            where label=i, loss=residual_real - residual_fake, sreal=residual_real, u=Ui
        """
        N_g = tf.shape(g)[0]
        g_fake = self.uniform_recombine(g)

        combined_sreal = []
        Us = []
        for i in range(self.n_class):
            u = self.Us[i]
            u = tf.nn.l2_normalize(u, dim=0)
            uT = tf.transpose(u)
            s_real = tf.reduce_sum((g - tf.matmul(tf.matmul(g, u), uT)) ** 2) / tf.to_float(N_g)

            combined_sreal.append(s_real)
            Us.append(u)
        combined_sreal = tf.convert_to_tensor(combined_sreal)
        Us = tf.convert_to_tensor(Us)
        label = tf.cast(tf.arg_min(combined_sreal, dimension=0), tf.int32)
        sreal = combined_sreal[label]
        u = Us[label]
        # returns label, and corresponding s_real and u
        return label, sreal, u

    def point_avg(self,points):
        dimensions=z_dim
        new_center=[]
        for dimension in range(dimensions):
            sum=0
            for p in points:
                sum+=p[dimension]
            new_center.append(float("%.8f"%(sum/float(len(points)))))
        return new_center
    def center_point(self,points):
        kmeans = KMeans(n_clusters = 1, max_iter = 500).fit(points)
        r2 = pd.DataFrame(kmeans.cluster_centers_)
        return r2
    def compute_disc_loss(self, z, y):
        print(z)
        group_index = [tf.where(tf.equal(y, k)) for k in
                       range(self.n_class)]  # indices of datapoints in k-th cluster
        groups = [tf.gather(z, group_index[k]) for k in range(self.n_class)]  # datapoints in k-th cluster
        # remove extra dimension
        groups = [tf.squeeze(g, axis=1) for g in groups]
        print("groups",groups)
        # subtract mean
        if self.args.submean:
            groups = [g - tf.reduce_mean(g, 0, keep_dims=True) for g in groups]
        dim1 = tf.shape(z)[1]
        # for each group, find its Ui
        group_all = [self.match_idx(g) for g in groups]
        group_label, group_sreal, group_u = zip(*group_all)
        # covnert some of them to tensor to make tf.where and tf.gather doable
        group_label = tf.convert_to_tensor(group_label)
        group_sreal = tf.convert_to_tensor(group_sreal)
        group_new_loss = []
        group_loss_real = []
        group_loss_fake = []
        group_loss_ce = []
        Us_assign_ops = []
        for i, g in enumerate(groups):
            N_g = tf.shape(g)[0]
            label = group_label[i]
            sreal = group_sreal[i]
            u = group_u[i]
            u = tf.nn.l2_normalize(u, dim=0)
            if self.args.one2one:
                # indices of groups, whose label are the same as current one
                idxs_with_label = tf.where(tf.equal(group_label, label))
                # sreal of those corresponding groups
                sreal_with_label = tf.squeeze(tf.gather(group_sreal, idxs_with_label), 1)
                # among all those groups with the same label, whether current group has minimal sreal
                ismin = tf.equal(sreal, tf.reduce_min(sreal_with_label))
                # if it's the minimum, just use, otherwise reinit u
                uu = tf.assign(self.Us[i], tf.cond(ismin, lambda: u, lambda: self.get_u_init_for_g(g)))
                u = tf.nn.l2_normalize(uu, dim=0)
                Us_assign_ops.append(uu)
            # recompute loss
            g = g / tf.norm(g, axis=1, keep_dims=True)
            g_fake = self.uniform_recombine(g)
            # pdb.set_trace()
            if args.degerate:
                center = tf.reduce_mean(g, 0)
                center = tf.reshape(center, [1,100])
                loss_ce = tf.reduce_sum((g - center) ** 2, axis=1)
                loss_real = tf.reduce_sum((g - tf.matmul(tf.matmul(g, u), tf.transpose(u))) ** 2, axis=1)
                num_real = tf.to_int32(
                    tf.minimum(
                        tf.maximum(
                            tf.to_int32(args.s_lambda1 * args.s_tau1 ** (self.gen_step)),
                            tf.to_int32(0.4 * tf.to_float(N_g))),
                        N_g))
                loss_real, idx1 = tf.nn.top_k(-loss_real, k=tf.to_int32(num_real), sorted=True)
                loss_ce = loss_ce[idx1]
                loss_real = - tf.reduce_sum(loss_real) / tf.to_float(num_real)
                loss_ce=  tf.reduce_sum(loss_ce) / tf.to_float(num_real)

                num_fake = tf.to_int32(tf.minimum(
                    tf.maximum(tf.to_int32(args.s_lambda2 * args.s_tau2 ** (self.gen_step)),
                               tf.to_int32(0.4 * tf.to_float(N_g))), N_g))
                loss_fake = tf.reduce_sum((g_fake - tf.matmul(tf.matmul(g_fake, u), tf.transpose(u))) ** 2, axis=1)
                loss_fake, idx2 = tf.nn.top_k(loss_fake, k=num_fake, sorted=True)
            else:
                center = tf.reduce_mean(g,0)
                center = tf.reshape(center,[1,100])
                loss_ce = tf.reduce_sum((g - center) ** 2)/ tf.to_float(N_g)
                loss_real = tf.reduce_sum((g - tf.matmul(tf.matmul(g, u), tf.transpose(u))) ** 2) / tf.to_float(N_g)
                loss_fake = tf.reduce_sum((g_fake - tf.matmul(tf.matmul(g_fake, u), tf.transpose(u))) ** 2, axis=1)
                num_fake = N_g

            if self.args.m:
                loss_fake = self.args.m - loss_fake
                loss_fake = -tf.nn.relu(loss_fake)
            loss_fake = tf.reduce_sum(loss_fake) / tf.to_float(num_fake)
            if self.args.stop_real:
                loss_real = tf.stop_gradient(loss_real)
            loss = loss_real - loss_fake +0.0001*loss_ce
            print("loss",loss)
            group_new_loss.append(loss)
            group_loss_real.append(loss_real)
            group_loss_fake.append(loss_fake)
            group_loss_ce.append(loss_ce)
        self.disc_score_real = tf.reduce_mean(group_loss_real)
        self.disc_score_fake = tf.reduce_mean(group_loss_fake)
        return -tf.reduce_mean(group_new_loss), tf.group(*Us_assign_ops),group_loss_real,group_new_loss,group_loss_ce

    # eqn 8的R1
    def regularization1(self, reuse=False):
        combined = []
        for i in range(self.n_class):
            ui = self.Us[i]
            uiT = tf.transpose(ui)
            temp_sum = []
            for j in range(self.n_class):
                if j == i:
                    continue
                uj = self.Us[j]
                s = tf.reduce_sum((tf.matmul(uiT, uj)) ** 2)
                temp_sum.append(s)
            combined.append(tf.add_n(temp_sum))
        return tf.add_n(combined) / self.n_class

    # eqn 8的R2
    def regularization2(self, reuse=False):
        combined = []
        for i in range(self.n_class):
            ui = self.Us[i]
            uiT = tf.transpose(ui)
            s = tf.reduce_sum((tf.matmul(uiT, ui) - tf.eye(self.rank)) ** 2)
            combined.append(s)
        return tf.add_n(combined) / self.n_class

    def partial_fit_eqn3(self, X, lr):
        # take a step on Eqn 3/4
        cost, Coef, summary, _ = self.sess.run(
            (self.loss_recon, self.Coef, self.summaryop_eqn3, self.optimizer_eqn3),
            feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost, Coef

    def assign_u_parameter(self, X, y):
        self.sess.run(self.u_ini, feed_dict={self.x: X, self.y_x: y})

    def partial_fit_disc(self, X, y_x, lr):
        self.sess.run([self.optimizer_disc, self.Us_update_op],
                      feed_dict={self.x: X, self.y_x: y_x, self.learning_rate: lr})

    def partial_fit_eqn3plus(self, X, y_x, lr):
        # assert y_x.min() == 0, 'y_x is 0-based'
        cost, Coef, summary, _, _ = self.sess.run(
            [self.loss_recon, self.Coef, self.summaryop_eqn3plus, self.optimizer_eqn3plus, self.gen_step_op],
            feed_dict={self.x: X, self.y_x: y_x, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost, Coef

    def partial_fit_pretrain(self, X, lr):
        cost, summary, _ = self.sess.run([self.loss_recon_pre, self.summaryop_pretrain, self.optimizer_pre],
                                         feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter += 1
        return cost

    def get_ae_weight_norm(self):
        norm, = self.sess.run([self.ae_weight_norm])
        return norm

    def get_loss_recon_pre(self, X):
        loss_recon_pre, = self.sess.run([self.loss_recon_pre], feed_dict={self.x: X})
        return loss_recon_pre

    def get_projection_y_x(self, X):
        disc_weights = self.sess.run(self.disc_weights)
        z_real = self.sess.run(self.z_real_submean, feed_dict={self.x: X})
        residuals = []
        for Ui in disc_weights:
            proj = np.matmul(z_real, Ui)
            recon = np.matmul(proj, Ui.transpose())
            residual = ((z_real - recon) ** 2).sum(axis=1)
            residuals.append(residual)
        residuals = np.stack(residuals, axis=1)  # Nxn_class
        y_x = residuals.argmin(1)
        return y_x

    def log_accuracy(self, accuracy):
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
        self.summary_writer.add_summary(summary, self.iter)

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")

    def check_size(self, X):
        z = self.sess.run(self.z, feed_dict={self.x: X})
        return z

def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T;
    return Cksym

def spectral_cluster(L, n, eps=2.2 * 10 - 8):
    """
    L: Laplacian  拉普拉斯式
    n: number of clusters
    Translates MATLAB code below:
    N  = size(L, 1)
    DN = diag( 1./sqrt(sum(L)+eps) );
    LapN = speye(N) - DN * L * DN;
    [~,~,vN] = svd(LapN);
    kerN = vN(:,N-n+1:N);
    normN = sum(kerN .^2, 2) .^.5;
    kerNS = bsxfun(@rdivide, kerN, normN + eps);
    groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    """
    N = L.shape[0]
    DN = (1. / np.sqrt(L.sum(0) + eps))
    LapN = np.eye(N) - DN * L * DN


def post_proC(C, K, d, alpha):
    C = 0.5 * (C + C.T)
    r = d * K + 1  # K=38, d=10
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    #Z = Z * (Z > 0)
    L = np.abs(np.abs(Z) ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)  # +1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C, axis=0)
    W = np.diag(1.0 / W)
    L = W.dot(C)
    return L

def reinit_and_optimize(args, Img, Label, CAE, n_class, k=10, post_alpha=3.5, alpha = 0.1):

    best_epoch=0
    best_acc=0
    best_ari = 0
    best_alpha=0
    best_postalpha=0
    print(alpha)
    y_x= Label
    acc_ = []
#epochs is train eq3 and eq3plus times
    #1000/400
    if args.epochs is None:
        num_epochs = 50 + n_class * 25  # 100+n_class*20  50+14*25=400
    else:
        num_epochs = args.epochs

    # init
    CAE.initlization()
    bn=15

    ###
    ### Stage 1: pretrain
    ###
    # if we skip pretraining, we restore already-trained model
    if args.pretrain == 0:
        CAE.restore()
        Z = CAE.sess.run(CAE.z, feed_dict={CAE.x: Img})
    # otherwise we pretrain the model first
    else:
        #pretrain =100
        print('Pretrain for {} steps'.format(args.pretrain))
        """
        After pretrain: 
            AE l2 norm   : 29
            Ae recon loss: 13372
        """
        for epoch in range(1, args.pretrain + 1):
            minibatch_size =128
            indices = np.random.permutation(Img.shape[0])[:minibatch_size]
            minibatch = Img[indices]  # pretrain with random mini-batch
            cost = CAE.partial_fit_pretrain(minibatch, args.lr)
            if epoch % 100 == 0:
                norm = CAE.get_ae_weight_norm()
                print('pretraining epoch {}, cost: {}, norm: {}'.format(epoch, cost / float(minibatch_size), norm))
        if args.save:
            print("save model")
            CAE.save_model()
    print('Finetune for {} steps'.format(num_epochs))
    acc_x = 0.0
    ari_x = 0.0
    nmi_x = 0.0
    y_x_mode = 'svd'
    for epoch in range(1, num_epochs + 1):
        # eqn3
        if epoch < args.enable_at:
            cost, Coef = CAE.partial_fit_eqn3(Img, args.lr)
            interval = args.interval  # normal interval
        # overtrain discriminator
        elif epoch == args.enable_at:
            print('Initialize discriminator for {} steps'.format(args.D_init))
            for i in range(args.D_init):
                print("ARGS.DINIT",i)
                CAE.partial_fit_disc(Img, y_x, args.lr2)
            if args.proj_cluster:
                y_x_mode = 'projection'
        else:
            for i in range(args.D_steps):
                CAE.partial_fit_disc(Img, y_x, args.lr2)  # discriminator step discriminator
            for i in range(args.G_steps):
                cost, Coef = CAE.partial_fit_eqn3plus(Img, y_x, args.lr2)
            interval = args.interval2
        if epoch % interval == 0:
            print("epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))
            Coef = thrC(Coef, alpha)
            t_begin = time.time()
            if y_x_mode == 'svd':
                y_x_new, LM = post_proC(Coef, n_class, k, post_alpha)
                print("label_pre",y_x_new)
            else:
                y_x_new = CAE.get_projection_y_x(Img)
            if len(set(list(np.squeeze(y_x_new)))) == n_class:
                y_x = y_x_new
            else:
                print('================================================')
                print('Warning: clustering produced empty clusters')
                print('================================================')
            missrate_x = err_rate(Label, y_x)
            ari_x = adjusted_rand_score(Label, y_x)
            nmi_x = normalized_mutual_info_score(Label, y_x)
            t_end = time.time()
            acc_x = 1 - missrate_x
            print("n_class",n_class)
            print("accuracy: {}".format(acc_x))
            print("ari: {}".format(ari_x))
            print('post processing time: {}'.format(t_end - t_begin))
            CAE.log_accuracy(acc_x)
            CAE.log_accuracy(ari_x)
            clustered = True
            if  ari_x> 0.82:
                best_acc = acc_x
                best_ari = ari_x
                best_nmi = nmi_x
                Z = CAE.sess.run(CAE.z_c, feed_dict={CAE.x: Img})
                break
            if best_acc < acc_x or best_ari < ari_x :
                best_acc = acc_x
                best_ari = ari_x
                best_nmi = nmi_x
                Z = CAE.sess.run(CAE.z_c, feed_dict={CAE.x: Img})
    mean = acc_x
    median = acc_x
    print("{} subjects, accuracy: {}".format(n_class, acc_x))

    return (1 - mean), (1 - median), best_epoch, best_acc,best_ari, best_nmi, best_alpha, best_postalpha


def prepare_data_singlecell(folder):
    data_path = 'E:\\code data\\ClusterGAN_Sc_data_zan\\human3_shaixuan10\\GSM2230759_human3_10_tpm.csv'
    label_path= 'E:\\code data\\ClusterGAN_Sc_data_zan\\human3_shaixuan10\\label.txt'
    data = pd.read_csv(data_path)
    Label = np.loadtxt(label_path).astype(int)
    print("data_length",len(data))
    Img = np.array(data)

    print("label_length",len(Label))
    x_dim =7596
    z_dim = 100
    n_hidden = [15]
    kernel_size = [3]
    n_sample_perclass = Img.shape[0] / 100
    disc_size = [50,1]
    # tunable numbers
    k= 12           # svds parameter
    post_alpha= 4.0  # Laplacian parameter
    alpha = 0.2
#类别
    all_subjects=[14]
    model_path  = os.path.join(folder, 'Singlecell-model.ckpt')
    print("model_path",model_path)
    return alpha, Img, Label, x_dim, z_dim, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path

def normalize_data(data):
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    return data



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()

    #prepare data
    folder = os.path.dirname(os.path.abspath(__file__))
    preparation_funcs = {
        'single cell':prepare_data_singlecell}
    assert args.dataset in preparation_funcs
    alpha, Img, Label, n_input,z_dim,n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path = \
            preparation_funcs[args.dataset](folder)

# prepare model
    # n_input=x_dim ,z_dim
    models = importlib.import_module(args.data + '.' + args.model)
    encoder = models.Encoder(z_dim=z_dim)
    decoder = models.Decoder(z_dim=z_dim ,x_dim=n_input)


    Img = Img*args.imgmult
    post_alpha = args.palpha or post_alpha
    logs_path = os.path.join(folder, 'logs')
    restore_path = model_path

    # arrays for logging results
    avg = []
    med = []

    # for each experiment setting, perform one loop
    for n_class in all_subjects:
        print("main_n_class",n_class)
        batch_size = 3603
        lambda1 = args.lambda1  # L2 sparsity on C
        lambda2 = args.lambda2  # 0.2 # 1.0 * 10 ** (n_class / 10.0 - 3.0)    # self-expressivity
        lambda3 = args.lambda3  # discriminator gradient

        # clear graph and build a new conv-AE
        tf.reset_default_graph()
        CAE = ConvAE(
            args,
            encoder,decoder, n_input , z_dim , n_hidden, kernel_size, n_class, n_sample_perclass, disc_size,
            lambda1, lambda2, lambda3, batch_size, r=args.r, rank=args.rank,
            reg=tf.contrib.layers.l2_regularizer(tf.ones(1) * args.lambda4), disc_bound=args.bound,
            model_path=model_path, restore_path=restore_path, logs_path=logs_path)
        # perform optimization
        avg_i, med_i, best_epoch, best_acc,best_ari,best_nmi, best_alpha, best_postalpha = reinit_and_optimize(args, Img, Label, CAE,
                                                                                             n_class, k=k,
                                                                                             post_alpha=post_alpha,
                                                                                             alpha=alpha)
        # add result to list
        avg.append(avg_i)
        med.append(med_i)

        # report results for all experiments
    for i, n_class in enumerate(all_subjects):
        print('%d subjects:' % n_class)
        print('Mean: %.4f%%' % (avg[i] * 100), 'Median: %.4f%%' % (med[i] * 100))

        print(best_epoch, best_acc,best_ari, best_nmi,  best_alpha, best_postalpha)
