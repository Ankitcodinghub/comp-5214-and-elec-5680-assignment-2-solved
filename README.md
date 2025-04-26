# comp-5214-and-elec-5680-assignment-2-solved
**TO GET THIS SOLUTION VISIT:** [COMP 5214 and ELEC 5680 Assignment 2 Solved](https://www.ankitcodinghub.com/product/comp-5214-and-elec-5680-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;105713&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP 5214 and ELEC 5680 Assignment 2  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1 Word Vectors [50 points]

In this assignment, you will need Python 3.5 or above. You also need to install the genism package. You can work directly on Jupyter Notebook file “exploring word vectors.ipynb” and submit that. However, if you are not familiar with Jupyter Notebook, please work on the “exploring word vectors.py” file, and submit additional pdf file for the written answers. You can refer to our Lecture 15 for the word2vec background.

Note that this assignment is taken from Stanford CS224N, and you can refer to the materials from Stanford for background knowledge. Please don’t simply take solutions from online, if there is any.

Here is a good tutorial about NLP: https://pytorch.org/tutorials/ beginner/chatbot_tutorial.html

2 Graph Neural Networks [50 points]

Graph Neural Networks (GNNs) are a class of neural network architectures used for deep learning on graph-structured data. Broadly, GNNs aim to generate high-quality embeddings of nodes by iteratively aggregating feature information from local graph neighborhoods using neural networks; embeddings can then be used for recommendations, classification, link prediction or other downstream tasks. Two important types of GNNs are GCNs (graph convolutional networks) and GraphSAGE (graph sampling and aggregation).

Let G = (V,E) denote a graph with node feature vectors Xu for u ∈ V . To generate the embedding for a node u, we use the neighborhood of the node as the computation graph. At every layer l, for each pair of nodes u ∈ V and its neighbor v ∈ V , we compute a message function via neural networks, and apply a convolutional operation that aggregates the messages from the node’s local graph neighborhood (Figure 1), and updates the node’s representation at next layer. By repeating this process through K GNN layers, we capture feature and structural information from the local K-hop neighborhood. For each of the message computation, aggregation and update functions, the learnable parameters are shared across all nodes in the same layer.

We initialize the feature vector of each node Xu based on the data we have available. If we already have outside information about the nodes, we can embed that as a feature vector. Otherwise we can use constant feature (vector of 1) or the degree of the node as the feature vector.

These are the key steps in each layer of a GNN:

• Message computation: We use a neural network to learn a message function between nodes. For each pair of nodes u and its neighbor v, the neural network message function can be expressed as ). In GCN and GraphSAGE, this can simply be σ(Whv + b), where W and b are the weights are bias of a neural network linear layer. Here refers to the hidden representation of node u at layer k, and eu,v denotes available information about the edge (u,v), like the edge weight or other features. For GCN and GraphSAGE, the neighbors of u are simply defined as nodes that are connected to u. However, many other variants of GNNs have different definitions of neighborhood.

• Aggregation: At each layer, we apply a function to aggregate information from all of the neighbors of each node. The aggregation function is usually permutation invariant, to reflect the fact that nodes’ neighbors have no canonical ordering. In a GCN, the aggregation is done by a weighted sum, where the weight for aggregating from v to u corresponds to the (u,v) entry of the normalized adjacency matrix D− / AD−1/2.

• Update: We update the representation of a node based on the aggregated representation of the neighborhood. For example, in GCNs, a multi-layer perceptron (MLP) is used; GraphSAGE combines a skip layer with the MLP.

• Pooling: The representation of an entire graph can be obtained by adding a pooling layer at the end. The simplest pooling methods are just taking the mean, max or sum of all of the individual node representations1. This is usually done for the purposes of graph classification2.

We can formulate the Message computation, Aggregation and Update steps for a GCN as a layerwise propagation rule given by:

hk+1 = σ(D−1/2AD−1/2hkWk), (1)

where hk represents the matrix of activations in the k-th layer, D−1/2AD−1/2 is the normalized adjacency of graph G, Wk is a layer-specific learnable matrix, and is a non-linearity function. Dropout and other forms of regularization can also be used.

We provide the pseudo-code for GraphSAGE embedding generation below.

Figure 1: GNN architecture.

In this question, we investigate the effect of the number of message passing layers on the expressive power of Graph Convolutional Networks. In neural networks, expressiveness refers to the set of functions (usually the loss function for classification or regression tasks) a neural network is able to compute, which depends on the structural properties of a neural network architecture.

2.1 Effect of Depth on Expressiveness [18 points]

1. Consider the following 2 graphs, where all nodes have 1-dimensional feature. We use a simplified version of GNN, with no nonlinearity and linear layers, and sum aggregation. We run the GNN to compute node embeddings for the 2 red nodes respectively. Note that the 2 red nodes have different 4-hop neighborhood structure. How many layers of message passing are needed so that these 2 nodes can be distinguished (i.e., have different GNN embeddings)?

2. Consider training a GNN on a node classification task, where nodes areclassified to be positive if they are part of an induced cyclic subgraph of length 10, and negative otherwise. Give an example of a graph and the node which should be classified as True. Show that no GNN with fewer than 5 layers can perfectly perform this classification task.

2.2 Relation to Random Walk [16 Points]

1. Let’s explore similarity between message passing and random walks. Let be the embedding of node i in layer l. Suppose that we are using a mean aggregator for message passing, and omit the learned linear transformation and non-linearity: . If we start at a node u, and take a uniform random walk for 1 step, the expectation of the embedding of the node we end on is exactly the embedding of u in the next layer. What is the transition matrix of the random walk? Describe the transition matrix using the adjacency matrix A, and degree matrix D, a diagonal matrix where Di,i is the degree of node i.

2. Suppose that we add skip connection in aggregation:

What is the corresponding transition matrix?

2.3 Learning BFS with GNN [16 Points]

1. Next we investigate the expressive power of GNN for learning simple graphalgorithms. Consider breadth-first search (BFS), where at every step, nodes that are connected to visited nodes become visited. Suppose that we use GNN to learn to execute the BFS algorithm. Suppose that the embeddings are 1-dimensional. Initially, all nodes have input feature 0, except a source node which has input feature 1. At every step, nodes reached by BFS has embedding 1, and nodes not reached by BFS has embedding 0. Write the update rule at every step of BFS execution.

2. Describe a message function and an aggregation function for the GNNsuch that it learns the task perfectly.

What to submit

Question 2.1:

• Reasoning of why a proposed number of message passing step is enough to distinguish the neighborhoods.

• An example that should be classified to True.

• Reasoning of why GNNs with fewer than 5 layers do not have enough expressive power.

Question 2.2:

• Correct transition matrix, in terms of D and A.

• Correct transition matrix for the case with skip layer.

Question 2.3:

• Update rule: the function that determines whether a node is reachable from source at step t, given its previous reachability and its neighbors’ previous reachability.

• aggregation method (anything containing sum+clamp; max; min) all acceptable.
