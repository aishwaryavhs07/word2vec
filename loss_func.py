import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    
    product=tf.matmul(inputs,tf.transpose(true_w)) #product of Transpose(u_o) and v_c
    expa=tf.math.exp(product)
    A=tf.log(tf.linalg.diag_part(expa)+1e-10)
    B=tf.log(tf.reduce_sum(expa,1)) #summing column-wise
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    
    #obtaining values of dimensions
    batch_size=inputs.shape[0].value
    embedding_size=inputs.shape[1].value
    k=len(sample)
    
    #calculating Pr
    pr=tf.convert_to_tensor(unigram_prob,dtype=tf.float32)
    #calculating Pr(D=1,wo|wc)
    pr_woc=tf.nn.embedding_lookup(pr,labels) #as per labels
    pr_woc=tf.reshape(pr_woc,[batch_size])
    #calculating uo predicting words
    u_o=tf.nn.embedding_lookup(weights,labels) #as per labels
    u_o=tf.reshape(u_o,[-1,embedding_size]) #flatten to 1D
    #calculating Pr(D=1,wx|wc)
    pr_wxc=np.ndarray(shape=(k),dtype=np.float32)
    #initialization
    for index in range(k):
        pr_wxc[index]=unigram_prob[sample[index]] #words with negative samples
        
    #calculating ux
    u_x=tf.nn.embedding_lookup(weights,sample)
    s_o=tf.diag_part(tf.matmul(inputs,tf.transpose(u_o)))
    bias_o=tf.nn.embedding_lookup(biases,labels)
    tf.reshape(bias_o,[batch_size])
    s_x=tf.matmul(inputs,tf.transpose(u_x))
    bias_x=tf.nn.embedding_lookup(biases,sample)
    
    final_so= tf.sigmoid(s_o+bias_o-tf.log(k*pr_woc+1e-10))
    final_sx= tf.sigmoid(s_x+bias_x-tf.log(k*pr_wxc+1e-10))
    return -(tf.log(final_so+1e-10)+tf.reduce_sum(tf.log(1-final_sx+1e-10),1))
   

    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
