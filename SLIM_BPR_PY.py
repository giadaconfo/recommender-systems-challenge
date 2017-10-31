import sys
import time

import numpy as np
import scipy.sparse as sps
from scipy.special import expit
from tqdm import tqdm

def sigmoidFunction(x):
  return 1 / (1 + np.exp(-x))


class SLIM_BPR_Python():

    def __init__(self, URM_train, sparse_weights = False):
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

        self.normalize = False
        self.sparse_weights = sparse_weights

        self.URM_mask = self.URM_train.copy()
        self.URM_mask.data = self.URM_mask.data == 1
        self.URM_mask.eliminate_zeros()


        if self.sparse_weights:
            self.S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        else:
            self.S = np.zeros((self.n_items, self.n_items)).astype('float32')


    def updateSimilarityMatrix(self):

        if self.topK != False:
            self.sparse_weights = True
            self.W_sparse = self.similarityMatrixTopK(self.S.T, k=self.topK)
        else:
            if self.sparse_weights == True:
                self.W_sparse = self.S.T
            else:
                self.W = self.S.T



    def updateWeightsLoop(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        x_ui = self.S[i]
        x_uj = self.S[j]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        sigmoid = expit(-x_uij)

        delta_i = sigmoid-self.lambda_i*self.S[i]
        delta_j = -sigmoid-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        for sampleIndex in range(self.batch_size):

            user_id = u[sampleIndex]

            for item_id in self.userSeenItems[user_id]:
                # Do not update items i
                if item_id != i[sampleIndex]:
                    self.S[i] += self.learning_rate * delta_i

                # Do not update j
                if item_id != j[sampleIndex]:
                    self.S[j] += self.learning_rate * delta_j


    def updateWeightsBatch(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        if self.batch_size==1:
            seenItems = self.userSeenItems[u[0]]

            x_ui = self.S[i, seenItems]
            x_uj = self.S[j, seenItems]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            #x_uij = x_uij[0,seenItems]
            x_uij = np.sum(x_uij)

            # log(sigm(+x_uij))
            gradient = 1 / (1 + np.exp(x_uij))

            # sigm(-x_uij)
            #exp = np.exp(x_uij)
            #gradient = exp/np.power(exp+1, 2)

        else:

            x_ui = self.S[i]
            x_uj = self.S[j]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            x_uij = self.URM_mask[u,:].dot(x_uij.T).diagonal()

            gradient = np.sum(1 / (1 + np.exp(x_uij))) / self.batch_size


        if self.batch_size==1:

            userSeenItems = self.userSeenItems[u[0]]

            self.S[i, userSeenItems] += self.learning_rate * gradient
            self.S[i, i] = 0

            self.S[j, userSeenItems] -= self.learning_rate * gradient
            self.S[j, j] = 0



        else:
            itemsToUpdate = np.array(self.URM_mask[u, :].sum(axis=0) > 0).ravel()

            # Do not update items i, set all user-posItem to false
            # itemsToUpdate[i] = False

            self.S[i] += self.learning_rate * gradient * itemsToUpdate
            self.S[i, i] = 0

            # Now update i, setting all user-posItem to true
            # Do not update j

            # itemsToUpdate[i] = True
            # itemsToUpdate[j] = False

            self.S[j] -= self.learning_rate * gradient * itemsToUpdate
            self.S[j, j] = 0

    def sampleUser(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while (True):
            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if (numSeenItems > 0 and numSeenItems < self.n_items):
                return user_id


    def sampleItemPair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while (True):

            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id


    def sampleBatch(self):
        user_id_list = np.random.choice(self.eligibleUsers, size=(self.batch_size))
        pos_item_id_list = [None]*self.batch_size
        neg_item_id_list = [None]*self.batch_size

        for sample_index in range(self.batch_size):
            user_id = user_id_list[sample_index]

            pos_item_id_list[sample_index] = np.random.choice(self.userSeenItems[user_id])

            negItemSelected = False

            # It's faster to just try again then to build a mapping of the non-seen items
            # for every user
            while (not negItemSelected):
                neg_item_id = np.random.randint(0, self.n_items)

                if (neg_item_id not in self.userSeenItems[user_id]):
                    negItemSelected = True
                    neg_item_id_list[sample_index] = neg_item_id

        return user_id_list, pos_item_id_list, neg_item_id_list

    def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100, verbose = False, inplace=True):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param forceSparseOutput:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if forceSparseOutput:
                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time()-start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):

                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx+1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                idx_sorted = np.argsort(column_data)  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[top_k_idx])
                rows_indices.extend(column_row_index[top_k_idx])


            cols_indptr.append(len(data))

            # During testing CSR is faster
            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

    def fit(self, epochs=30, minRatingsPerUser=1,
            batch_size = 1000, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):

        """
        Fits the model performing a round of testing at the end of each epoch
        :param epochs:
        :param batch_size:
        :param logFile:
        :return:
        """


        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK


        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate


        start_time_train = time.time()

        for currentEpoch in tqdm(range(epochs)):

            start_time_epoch = time.time()

            if currentEpoch > 0:
                if self.batch_size>0:
                    self.epochIteration()
                else:
                    print("No batch not available")
            else:
                self.updateSimilarityMatrix()

            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))


    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz*1)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        totalNumberOfBatch = int(numPositiveIteractions/self.batch_size)+1

        # Uniform user sampling without replacement
        for numCurrentBatch in range(totalNumberOfBatch):

            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()

            self.updateWeightsBatch(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )

            """
            self.updateWeightsLoop(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )
            """

            if(time.time() - start_time_batch >= 30 or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()



        self.S[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0

        self.updateSimilarityMatrix()
