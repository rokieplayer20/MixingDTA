The relationships between DT nodes in the train dataset can be pre-saved in .npy format in this directory.
The matrix is stored in a Boolean format. Even if it is not pre-generated, the 'find_connection' function will search for the relationships individually.
Since it searches for each relationship individually, the preprocessing can take a considerable amount of time. To avoid this, it is recommended to pre-generate and save the Boolean matrix using NumPy. Since the size of the '.npy' files is very large, they cannot be directly shared on GitHub.
