import netCDF4 as nc
import numpy as np
import os

from isml.data import partition as partition_np

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

def compute_signatures(dimensions, features_of_interest, partition_size, preprocess_fn, signature_fn, postprocess_fn, timesteps_of_interest):

    dataset = Dataset(dimensions, features_of_interest)
    loader = DataLoader(dataset)

    if timesteps_of_interest is not None:
        all_timesteps = np.arange(timesteps_of_interest[0], timesteps_of_interest[0]+timesteps_of_interest[1])

    all_signatures = []
    for timestep, process, features, anomalies in loader:
        if (timesteps_of_interest is None) or (int(timestep) in all_timesteps):
            if np.mod(int(timestep), 10) == 0:
                progress_str = "Processing timestep %d." % (int(timestep))
                print(progress_str)
            
            if len(dimensions) == 3:
                features = features[:,:,None,:]

            # Split the features array into partitions.
            counts = (features.shape[0] // partition_size[0], features.shape[1] // partition_size[1], features.shape[2] // partition_size[2])
            partition_indices, partitions = partition_np(features, counts=counts)

            # Scale features to the range [0, 1].
            partitions = preprocess_fn(partitions)

            # Convert partitions to signatures.
            timestep_signatures = signature_fn(partitions)

            # Postprocess signatures.
            timestep_signatures = postprocess_fn(timestep_signatures)
            
            # Keep track of all signatures across time, in case we need them for temporal measures.
            all_signatures.append(timestep_signatures)

    class Results: pass
    results = Results()
    results.all_signatures = all_signatures
    results.partition_indices = partition_indices
    results.timesteps_of_interest = timesteps_of_interest
    results.partition_size = partition_size

    return results

def write_signatures(feature_path, signature_path, data):
    all_signatures    = data.all_signatures
    partition_size    = data.partition_size
    timestep0         = data.timesteps_of_interest[0]
    num_timesteps     = data.timesteps_of_interest[1]
    
    num_signatures = all_signatures[0].shape[-1]
    num_partitions = all_signatures[0].shape[0]

    os.system("nccopy -V time,time_bnds,lat,lon %s %s" % (feature_path, signature_path))

    with nc.Dataset(signature_path, "r+") as ncdata:
        # add feature_path to signature file header
        ncdata.setncattr("feature_path", feature_path)
        ncdata.setncattr("partition_size_x", partition_size[0])
        ncdata.setncattr("partition_size_y", partition_size[1])
        ncdata.setncattr("sampled_timestep0", timestep0)
        ncdata.setncattr("num_sampled_timesteps", num_timesteps)
        # add npbnd dimension (length 6)
        ncdata.createDimension('npbnd', 6)
        # add pid dimension (length n_partitions)
        ncdata.createDimension('pid', num_partitions)
        # add sid dimension (length n_signatures)
        ncdata.createDimension('sid', num_signatures)
        # add signatures variables: signatures(time, pid, sid) = [s1, ..., sN]
        out = ncdata.createVariable('signatures', np.float64, ('time', 'pid', 'sid'))
        for i in np.arange(timestep0, timestep0+num_timesteps):
            out[i,:,:] = all_signatures[i-timestep0]

def load_signatures(signature_path):

    class Results: pass
    results = Results()
    
    with nc.Dataset(signature_path, "r") as ncdata:
        timestep0 = ncdata.getncattr("sampled_timestep0")
        num_timesteps = ncdata.getncattr("num_sampled_timesteps")
        timesteps_of_interest = [timestep0, num_timesteps]

        partition_size_x = ncdata.getncattr("partition_size_x")
        partition_size_y = ncdata.getncattr("partition_size_y")
        partition_size = [partition_size_x, partition_size_y]

        nlat = ncdata.dimensions['lat'].size
        nlon = ncdata.dimensions['lon'].size

        counts = (nlat // partition_size_x, nlon // partition_size_y, 1)
        partition_indices, partitions = partition_np(np.empty((nlat, nlon, 1)), counts=counts)

        all_signatures = []
        for i in np.arange(timestep0, timestep0+num_timesteps):
            all_signatures.append(ncdata['signatures'][i,:,:])

        results.all_signatures = all_signatures
        results.partition_indices = partition_indices
        results.timesteps_of_interest = timesteps_of_interest
        results.partition_size = partition_size

    return results