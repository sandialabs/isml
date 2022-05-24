import netCDF4 as nc
import numpy as np
import os

class Dataset(object):
    def __init__(self, dimensions, features):
        self._dimensions = dimensions
        
        self._features = []
        self._paths = []
        for key, val in features.items():
            self._features.append(key)
            self._paths.append(val)

        ncdata = nc.Dataset(self._paths[0], "r")

        self._timesteps = [str(i) for i in range(len(ncdata['time']))]
        self._processes = ["0000" for s in ncdata['time'][:]]

        self._dim0 = ncdata[self._dimensions[0]][:]
        self._dim1 = ncdata[self._dimensions[1]][:]
        self._coords = np.zeros([len(self._dim0), len(self._dim1), 2])
        Dim0, Dim1 = np.meshgrid(self._dim0, self._dim1, indexing='ij')
        self._coords[:,:,0] = Dim0
        self._coords[:,:,1] = Dim1

        ncdata.close()

    def __len__(self):
        return len(self._timesteps)

    def __getitem__(self, index):
        return self._timesteps[index], self._processes[index], self._paths, None

    @property
    def features(self):
        return self._features

    @property
    def dimensions(self):
        return [len(self._dim0), len(self._dim1)]

class DataLoader(object):
    def __init__(self, dataset):
        self._dataset = dataset
        self._feature_list = self._dataset.features
        self._feature_dim = self._dataset.dimensions
        self._feature_dim.append(len(self._feature_list))

    def __iter__(self):
        for index in range(len(self._dataset)):
            timestep, process, netcdf_paths, anomaly_path = self._dataset[index]

            features = np.zeros(self._feature_dim)

            for feature_id in range(len(self._feature_list)):
                ncdata = nc.Dataset(netcdf_paths[feature_id])
                features[:,:,feature_id] = ncdata[self._feature_list[feature_id]][index,:,:]
                ncdata.close()

            anomalies = None

            yield timestep, process, features, anomalies

class DataWriter(object):
    def __init__(self, path, name):
        self._path = path
        self._name = name

    def write(self, output_features, output_types, data):

        # MLC NOTE: NetCDF4 will throw an exception if we try to create a variable that already exists,
        # therefore we can use the `ncks` to remove any output features that are already present.
        for feature in output_features:
            line = "ncks -C -O -x -v " + str(feature) + " " + self._path + " " + self._path
            os.system(line)

        # MLC NOTE: Currently the DataWriter appends the output fields to the input file
        ncdata = nc.Dataset(self._path, "r+")
        for output_index in range(len(output_features)):
            out = ncdata.createVariable(output_features[output_index], output_types[output_index], ('time','y','x',)) 
            for timestep in range(out.shape[0]):
                out[timestep,:,:] = data[output_index][timestep]
        ncdata.close()

def require_valid_mask(mask):
    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be a numpy array.")
    if mask.ndim < 2:
        raise ValueError("Mask must have ndim >= 2.")
    if mask.dtype != np.int32:
        raise ValueError("Mask must have numpy.int32 dtype.")
    if not np.array_equal(mask, mask.astype(bool)):
        raise ValueError("Mask should only have entries equal to 0 or 1.")
    if np.sum(mask) == 0:
        raise ValueError("Mask should have >1 entries equal to 1.")

def select_active_cells(source, coordinates, mask):

    require_valid_mask(mask)
    
    active_features = source[mask == 1, :]
    active_coordinates = coordinates[mask == 1, :]
    active_indices = np.argwhere(mask == 1)

    return active_features, active_coordinates, active_indices

def partition(active_coords, active_indices, num_partitions):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_partitions, init='random', n_init=1, verbose=1).fit(active_coords)
    partition_labels = kmeans.labels_
    partition_indices = []
    for i in range(num_partitions):
        partition_indices.append(active_indices[partition_labels == i,:])
    return partition_indices, partition_labels

def get_partition_data(partition_labels, active_features, num_partitions):
    partition_features = []
    for i in range(num_partitions):
        partition_features.append(active_features[partition_labels == i,:])
    return partition_features

def inflate(data, indices, dims):
    inflated_data = np.zeros([dims[0], dims[1], 1])
    for i in range(len(indices)):
        inflated_data[indices[i][:,0], indices[i][:,1], 0] = data[i]
    return inflated_data