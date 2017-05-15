import numpy as np
import tensorflow as tf
import random
from seqGAN.dataloader import Gen_Data_loader, Dis_dataloader
from seqGAN.generator import Generator
from seqGAN.discriminator import Discriminator
from seqGAN.rollout import ROLLOUT
import pickle

#
#  Generator  Hyper-parameters
# ----------------------------------------------------------------------------
EMB_DIM = 32            # embedding dimension
HIDDEN_DIM = 32         # hidden state dimension of lstm cell
SEQ_LENGTH = 20         # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120     # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
ROLLOUT_NUM = 16
G_STEPS = 1
# ----------------------------------------------------------------------------

#
#  Discriminator  Hyper-parameters
# ----------------------------------------------------------------------------
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
D_STEPS = 5
# ----------------------------------------------------------------------------

#
#  Basic Training Parameters
# ----------------------------------------------------------------------------
TOTAL_BATCH = 800
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
# ----------------------------------------------------------------------------


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    # Use the <trainable_model> to generate the positive examples.
    # And then write the samples into <output_file>.
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # <target_loss> means the oracle negative log-likelihood tested with the oracle model <target_lstm>
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, feed_dict={target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def groundtruth_loss(sess, target_lstm, data_loader):
    # <target_loss> means the oracle negative log-likelihood tested with the oracle model <target_lstm>
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, feed_dict={target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    #
    # Declare data loader
    # ----------------------------------------------------------------------------
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 5000
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    # ----------------------------------------------------------------------------


    #
    # Declare Generator & Discriminator
    # ----------------------------------------------------------------------------
    # declare: generator
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    # declare: discriminator
    discriminator = Discriminator(sequence_length=20, num_classes=2,
                                  vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    # ----------------------------------------------------------------------------


    #
    # Set the session <sess>
    # ----------------------------------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # ----------------------------------------------------------------------------

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate samples by using <target_lstm> and write the samples to file <positive_file>
    #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')


    #
    # Pre-train <generator> by using <gen_data_loader>,
    # and then compute the <test_loss> of <target_lstm> and <likelihood_data_loader>
    # ----------------------------------------------------------------------------
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            # generate samples by using <generator> and write the samples to file <eval_file>
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)

            # load samples from file <eval_file>
            likelihood_data_loader.create_batches(eval_file)

            # compute <test_loss> of <target_lstm>, with input <likelihood_data_loader>
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
    # ----------------------------------------------------------------------------


    #
    # Pre-train <discriminator> by using <generator>
    # ----------------------------------------------------------------------------
    print('Start pre-training discriminator...')
    # Generate data and train 3 epoch on the generated data, which will be done for 50 times
    for _ in range(50):
        # generate samples by using <generator> and write the samples to file <negative_file>
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

        # load samples from file <negative_file>
        dis_data_loader.load_train_data(positive_file, negative_file)

        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob}
                _ = sess.run(discriminator.train_op, feed_dict=feed)
    # ----------------------------------------------------------------------------

    rollout = ROLLOUT(generator, 0.8)

    #
    # Start seqGAN, train <discriminator> and <generator>
    # ----------------------------------------------------------------------------
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):

        # ----- Train the generator for one step -----------------
        for it in range(G_STEPS):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, ROLLOUT_NUM, discriminator, SEQ_LENGTH)
            feed = {generator.x: samples,
                    generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)
        # --------------------------------------------------------

        # Update roll-out parameters
        rollout.update_params()

        # ----- Train the discriminator -------------------------
        for _ in range(D_STEPS):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dis_dropout_keep_prob}
                    _ = sess.run(discriminator.train_op, feed_dict=feed)
        # --------------------------------------------------------
    # ----------------------------------------------------------------------------

    log.close()


if __name__ == '__main__':
    main()
