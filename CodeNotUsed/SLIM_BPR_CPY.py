from SLIM_BPR_PY import SLIM_BPR_Python
import subprocess
import os, sys
import numpy as np


class SLIM_BPR_Cython(SLIM_BPR_Python):


    def __init__(self, URM_train, recompile_cython = False, sparse_weights = False, sgd_mode='adagrad'):


        super(SLIM_BPR_Cython, self).__init__(URM_train, sparse_weights = sparse_weights)


        self.sgd_mode = sgd_mode


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def fit(self, epochs=30, minRatingsPerUser=1,
            batch_size = 1000, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, sgd_mode='adagrad'):

        self.eligibleUsers = []

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data == 1
        URM_train_positive.eliminate_zeros()

        for user_id in range(self.n_users):

            start_pos = URM_train_positive.indptr[user_id]
            end_pos = URM_train_positive.indptr[user_id+1]

            if len(URM_train_positive.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode


        # Import compiled module
        from SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch


        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                 self.sparse_weights,
                                                 self.eligibleUsers,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode = sgd_mode)


        # Cal super.fit to start training
        super(SLIM_BPR_Cython, self).fit(epochs=epochs,
                                         minRatingsPerUser=minRatingsPerUser,
                                         batch_size=batch_size,
                                         lambda_i = lambda_i,
                                         lambda_j = lambda_j,
                                         learning_rate = learning_rate,
                                         topK = topK)




    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/SLIM_BPR/Cython"
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        #python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "SLIM_BPR_Cython_Epoch.pyx"])


    def epochIteration(self):

        self.S = self.cythonEpoch.epochIteration_Cython()

        if self.sparse_weights:
            self.W_sparse = self.S
        else:
            self.W = self.S
