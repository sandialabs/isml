import functools
import glob
import itertools
import logging
import os
import shutil

import arrow
import graphcat
import imagecat
import imagecat.color.basic
import imagecat.color.brewer
import numpy
import toyplot.pdf


log = logging.getLogger(__name__)


def clean_data(path):
    for directory in ["anomalies", "features"]:
        shutil.rmtree(os.path.join(path, directory), ignore_errors=True)
    for file in []:
        if os.path.exists(os.path.join(path, file)):
            os.remove(os.path.join(path, file))


def clean_experiment(path):
    for directory in ["decisions", "measures", "movie"]:
        shutil.rmtree(os.path.join(path, directory), ignore_errors=True)
    for file in ["anomaly-recall.pdf", "decision-summary.pdf", "movie.mp4"]:
        if os.path.exists(os.path.join(path, file)):
            os.remove(os.path.join(path, file))


class Dataset(object):
    def __init__(self, path, features):
        # Get the complete set of paths for all requested features.
        features_path = os.path.join(path, "features")
        if not os.path.exists(features_path):
            raise ValueError(f"Nonexistent path: {features_path}")

        feature_paths = []
        for feature in features:
            feature_paths.append(sorted(glob.glob(f"{features_path}/{feature}_*.npy")))

        # Create a <number of timesteps> by <number of features> matrix.
        feature_paths = numpy.array(feature_paths).T

        # Confirm that timestep and process indices are consistent across all features.
        str_indexes = numpy.char.partition(feature_paths, "_")[:,:,2]
        str_timesteps = numpy.char.partition(str_indexes, "_")[:,:,0]
        str_processes = numpy.char.partition(numpy.char.partition(str_indexes, "_")[:,:,2], ".")[:,:,0]

        len_timesteps = numpy.char.str_len(str_timesteps)
        if not numpy.all(len_timesteps == len_timesteps[0,0]):
            raise ValueError("Timesteps must all use the same number of digits.")
        len_processes = numpy.char.str_len(str_processes)
        if not numpy.all(len_processes == len_processes[0,0]):
            raise ValueError("Processes must all use the same number of digits.")

        if not numpy.all(str_timesteps.T == str_timesteps[:,0]):
            raise ValueError("Timesteps aren't numbered consistently.")
        if not numpy.all(str_processes.T == str_processes[:,0]):
            raise ValueError("Processes aren't numbered consistently.")

        # Identify (optional) per-timestep ground truth anomaly paths.
        anomalies_path = os.path.join(path, "anomalies")
        anomaly_paths = []
        for timestep, process in zip(str_timesteps[:,0], str_processes[:,0]):
            anomaly_paths.append(os.path.join(anomalies_path, f"anomaly_{timestep}_{process}.npy"))

        self._features = features
        self._timesteps = str_timesteps[:,0]
        self._processes = str_processes[:,0]
        self._feature_paths = feature_paths
        self._anomaly_paths = anomaly_paths

    def __len__(self):
        return len(self._feature_paths)

    def __getitem__(self, index):
        return self._timesteps[index], self._processes[index], self._feature_paths[index], self._anomaly_paths[index]

    @property
    def features(self):
        return self._features


#class Subset(object):
#    def __init__(self, dataset, indices):
#        self._dataset = dataset
#        self._indices = indices
#
#    def __len__(self):
#        return len(self._indices)
#
#    def __getitem__(self, index):
#        return self._dataset[self._indices[index]]
#
#    @property
#    def features(self):
#        return self._dataset.features


class DataLoader(object):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        for index in range(len(self._dataset)):
            timestep, process, feature_paths, anomaly_path = self._dataset[index]

            features = [numpy.load(path) for path in feature_paths]
            features = numpy.stack(features, axis=-1)

            anomalies = numpy.load(anomaly_path) if os.path.exists(anomaly_path) else None

            yield timestep, process, features, anomalies


def partition_indices(source, counts):
    dimensions = []
    for size, count in zip(source.shape, counts):
        dimension = [slice(indices[0], indices[-1]+1) for indices in numpy.array_split(numpy.arange(size), count)]
        dimensions.append(dimension)
    return list(itertools.product(*dimensions))


def partition(source, counts):
    indices = partition_indices(source, counts)
    partitions = [source[index] for index in indices]
    return indices, partitions


def inflate(partitions, indices):
    shape = numpy.max([[dim.stop for dim in index] for index in indices], axis=0)
    inflated = numpy.empty(shape=shape, dtype=partitions[0].dtype)
    for index, partition in zip(indices, partitions):
        inflated[index] = partition
    return inflated


class DataWriter(object):
    def __init__(self, path, name):
        self._path = path
        self._name = name

    def write(self, timestep, process, data):
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        path = os.path.join(self._path, f"{self._name}_{timestep}_{process}.npy")
        log.info(f"Writing {path}")
        numpy.save(path, data)


class ImageWriter(object):
    def __init__(self, path, name, colormap):
        # Setup an image processing workflow.
        graph = graphcat.DynamicGraph()
        imagecat.add_task(graph, "/data", graphcat.constant(None))
        imagecat.add_task(graph, "/colormap", imagecat.operator.color.colormap, mapping=colormap)
        imagecat.add_task(graph, "/save", imagecat.operator.save, path=None)
        imagecat.add_links(graph, "/data", ("/colormap", "image"))
        imagecat.add_links(graph, "/colormap", ("/save", "image"))

        self._path = path
        self._name = name
        self._graph = graph

    def write(self, timestep, data):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        path = os.path.join(self._path, f"{self._name}_{timestep}.png")

        # Run the workflow
        log.info(f"Writing {path}")
        data = data[::-1] # All of our grids are upside-down, for historical reasons.
        self._graph.set_task("/data", graphcat.constant(imagecat.data.Image({"L": imagecat.data.Layer(data=data)})))
        self._graph.set_task("/save/path", graphcat.constant(path))
        self._graph.update("/save")


class MovieWriter(object):
    def __init__(self, *, path, name, res, rows=1, cols=1, scale=1):
        graph = graphcat.DynamicGraph()

        cellres = (res[0] * scale, res[1] * scale)
        fullres = (cellres[0] * cols, cellres[1] * rows)

        imagecat.add_task(graph, "/background", imagecat.operator.color.fill, res=fullres, layer="C", values=[0.5, 0.5, 0.5], role=imagecat.data.Role.RGB)
        imagecat.add_task(graph, "/shadowfill", imagecat.operator.color.fill, res=fullres, layer="C", values=[0, 0, 0], role=imagecat.data.Role.RGB)
        imagecat.add_task(graph, "/textfill", imagecat.operator.color.fill, res=fullres, layer="C", values=[1, 1, 1], role=imagecat.data.Role.RGB)
        imagecat.add_task(graph, "/textmask", imagecat.operator.color.fill, res=fullres, layer="A", values=[0], role=imagecat.data.Role.NONE)
        imagecat.add_task(graph, "/save", imagecat.operator.save, path=None)

        self._cells = 0
        self._cols = cols
        self._finished = False
        self._graph = graph
        self._name = name
        self._path = path
        self._res = res
        self._rows = rows
        self._scale = scale


    @property
    def graph(self):
        return self._graph


    def add_cell(self, *, row, col, label, colormap, labelsize="24px", labelposition=("0.5w", "0.9h")):
        graph = self._graph
        index = self._cells
        self._cells += 1

        res = self._res
        cellres = (res[0] * self._scale, res[1] * self._scale)

        background = "/background" if index == 0 else f"/comp{index-1}"
        data = imagecat.add_task(graph, f"/data{index}", graphcat.constant(None))
        colormap = imagecat.add_task(graph, f"/colormap{index}", imagecat.operator.color.colormap, mapping=colormap)
        rename = imagecat.add_task(graph, f"/rename{index}", imagecat.operator.rename, changes={"L": "C"})
        mask = imagecat.add_task(graph, f"/mask{index}", graphcat.constant(None))
        comp = imagecat.add_task(graph, f"/comp{index}", imagecat.operator.transform.composite, pivot=("0w", "1h"), position=((col / self._cols, "w"), (1 - row / self._rows, "h")), scale=(self._scale, self._scale), order=0)

        imagecat.add_links(graph, background, (comp, "background"))
        imagecat.add_links(graph, data, (colormap, "image"))
        imagecat.add_links(graph, colormap, (rename, "image"))
        imagecat.add_links(graph, rename, (comp, "foreground"))
        imagecat.add_links(graph, mask, (comp, "mask"))

        text = imagecat.add_task(graph, f"/text{index}", imagecat.operator.render.text, res=cellres, string=label, fontsize=labelsize, anchor="ma", position=labelposition)
        textcomp = imagecat.add_task(graph, f"/textcomp{index}", imagecat.operator.transform.composite, pivot=("0w", "1h"), position=((col / self._cols, "w"), (1 - row / self._rows, "h")), order=0, bglayer="A", fglayer="A")

        textbackground = "/textmask" if index == 0 else f"/textcomp{index-1}"
        imagecat.add_links(graph, textbackground, (textcomp, "background"))
        imagecat.add_links(graph, text, (textcomp, "foreground"))



    def write(self, timestep, *inputs):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        path = os.path.join(self._path, f"{self._name}_{timestep}.png")

        # Run the workflow.
        log.info(f"Writing {path}")
        graph = self._graph

        if not self._finished:
            imagecat.add_task(graph, f"/shadowoffset", imagecat.operator.transform.offset, offset=["2px", "-2px"])
            imagecat.add_task(graph, f"/shadowblur", imagecat.operator.blur.gaussian, radius=["1px", "1px"])
            imagecat.add_task(graph, f"/shadowcomp", imagecat.operator.transform.composite, order=0)

            imagecat.add_links(graph, f"/textcomp{self._cells-1}", ("/shadowoffset", "image"))
            imagecat.add_links(graph, f"/shadowoffset", ("/shadowblur", "image"))
            imagecat.add_links(graph, f"/comp{self._cells-1}", ("/shadowcomp", "background"))
            imagecat.add_links(graph, "/shadowfill", ("/shadowcomp", "foreground"))
            imagecat.add_links(graph, "/shadowblur", ("/shadowcomp", "mask"))

            imagecat.add_task(graph, "/textcomp", imagecat.operator.transform.composite, order=0)
            imagecat.add_links(graph, f"/shadowcomp", ("/textcomp", "background"))
            imagecat.add_links(graph, "/textfill", ("/textcomp", "foreground"))
            imagecat.add_links(graph, f"/textcomp{self._cells-1}", ("/textcomp", "mask"))

            imagecat.add_links(graph, f"/textcomp", ("/save", "image"))
            self._finished = True

        monitor = graphcat.PerformanceMonitor(graph)
        for index, input in enumerate(inputs):
            if isinstance(input, tuple):
                data = input[0][::-1]
                mask = input[1][::-1]
                graph.set_task(f"/data{index}", graphcat.constant(imagecat.data.Image({"L": imagecat.data.Layer(data=data)})))
                graph.set_task(f"/mask{index}", graphcat.constant(imagecat.data.Image({"A": imagecat.data.Layer(data=mask)})))
            else:
                data = input[::-1]
                graph.set_task(f"/data{index}", graphcat.constant(imagecat.data.Image({"L": imagecat.data.Layer(data=data)})))
        graph.set_task("/save/path", graphcat.constant(path))
        graph.update("/save")

        tasks = sorted(monitor.tasks.items(), key=lambda x: x[1], reverse=True)
#        for task in tasks:
#            log.info(task)


def blackbody_colormap():
    palette = imagecat.color.basic.palette("Blackbody")
    colormap = functools.partial(imagecat.color.linear_map, palette=palette)
    return colormap


def bluered_colormap():
    palette = imagecat.color.brewer.palette("BlueRed")
    colormap = functools.partial(imagecat.color.linear_map, palette=palette)
    return colormap


def measure_colormap():
    palette = imagecat.color.brewer.palette("BlueRed")
    colormap = functools.partial(imagecat.color.linear_map, palette=palette)
    return colormap


def decision_colormap():
    palette = imagecat.color.brewer.palette("Greys")
    colormap = functools.partial(imagecat.color.linear_map, palette=palette, min=0, max=1)
    return colormap


def feature_writer(path, feature):
    path = os.path.join(path, "features")
    return DataWriter(path, name=feature)


def feature_image_writer(path, feature):
    path = os.path.join(path, "features")
    return ImageWriter(path, name=feature, colormap=bluered_colormap())


def measure_writer(path):
    path = os.path.join(path, "measures")
    return DataWriter(path, name="measure")


def measure_image_writer(path):
    path = os.path.join(path, "measures")
    return ImageWriter(path, name="measure", colormap=measure_colormap())


def decision_writer(path):
    path = os.path.join(path, "decisions")
    return DataWriter(path, name="decision")

def signature_writer(path, outfilename):
    path = os.path.join(path, "signatures")
    return DataWriter(path, name=outfilename)

def decision_image_writer(path):
    path = os.path.join(path, "decisions")
    return ImageWriter(path, name="decision", colormap=decision_colormap())


def movie_writer(path, *, res, rows, cols, scale=1):
    path = os.path.join(path, "movie")
    return MovieWriter(path=path, name="movie", res=res, rows=rows, cols=cols, scale=scale)


def plot_decision_summary(path, all_decisions):
    if not os.path.exists(path):
        os.makedirs(path)
            
    decisions = numpy.array(all_decisions)
    interesting = numpy.sum(decisions, axis=1) / decisions.shape[1]
    total_interesting = numpy.sum(decisions) / decisions.size

    label = f"{path} interesting partitions: {total_interesting * 100:.1f}%"

    canvas = toyplot.Canvas(width=800, height=400)
    axes = canvas.cartesian(label=label, xlabel="Timestep", ylabel="Interesting partitions (%)", ymax=100)
    axes.fill(interesting * 100.0)
    toyplot.pdf.render(canvas, os.path.join(path, "decision-summary.pdf"))


def plot_anomaly_recall(path, all_anomalies, all_decisions):
    for anomalies, decisions in zip(all_anomalies, all_decisions):
        if anomalies.shape != decisions.shape:
            raise ValueError("Anomalies and decisions must have the same shape - did you forget to inflate one or the other?")

    counts = []
    true_positives = []
    for anomalies, decisions in zip(all_anomalies, all_decisions):
        counts.append(numpy.sum(anomalies))
        true_positives.append(numpy.sum(numpy.logical_and(anomalies, decisions)))
    counts = numpy.array(counts)
    true_positives = numpy.array(true_positives)
    false_negatives = counts - true_positives

    total = numpy.sum(counts)
    recall = numpy.sum(true_positives) / (numpy.sum(true_positives) + numpy.sum(false_negatives))

    selection = numpy.flatnonzero(counts)
    selection = numpy.arange(len(counts))

    legend = []
    label = path
    canvas = toyplot.Canvas(width=800, height=400)
    axes = canvas.cartesian(label=label, xlabel="Timestep", ylabel="Anomaly Count")
    legend.append(("Ground Truth", axes.plot(selection, counts[selection], size=4, marker="o")))
    legend.append(("Predicted", axes.plot(selection, true_positives[selection], style={"stroke-width": 0.5}, size=8, marker="o", mstyle={"fill":"none"})))
    legend.append(("Anomalies: {}".format(total), ""))
    legend.append(("<b>Recall: {:.1%}</b>".format(recall), ""))
    legend = canvas.legend(legend, corner=("top-right", 80, 200, 70))
    toyplot.pdf.render(canvas, os.path.join(path, "anomaly-recall.pdf"))


def gradient_sort(partition, gradient):
    if not isinstance(partition, numpy.ndarray):
        raise ValueError("partition must be a numpy array.")
    if partition.ndim != 4:
        raise ValueError("partition must have four dimensions (three spatial dimensions plus features).")
    if not isinstance(gradient, numpy.ndarray):
        raise ValueError("gradient must be a numpy array.")
    if gradient.shape != (3,):
        raise ValueError("gradient must be a vector with three elements.")


    # Enforce a unit vector for the gradient.
    gradient = gradient.astype(numpy.float64)
    gradient /= numpy.linalg.norm(gradient) + 1e-16

    # Project the partition coordinates onto the gradient vector.
    i, j, k = numpy.mgrid[0:partition.shape[0], 0:partition.shape[1], 0:partition.shape[2]]
    projection = numpy.dot(numpy.dstack((i, j, k)), gradient)

    # Return a sort order based on the projection.
    return numpy.argsort(projection, axis=None)

