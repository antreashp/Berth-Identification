##SYSTEM##
import os,sys
import ot
import pickle
from glob  import glob
import requests
import ast
from scipy.linalg import sqrtm, det
import rtree
##MATH##
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from math import radians, sin, cos, sqrt, atan2, pi
from scipy.linalg import sqrtm
##GEOSTUFF##
import geohash2
# import rasterio
import pyproj
import geopandas as gpd
# import pygeohash as pgh
import xml.etree.ElementTree as ET
# import torch
##PLOTTING##
import matplotlib.pyplot as plt
import branca
import selenium 
import folium
import geojsoncontour
from matplotlib.colors import LogNorm
from folium.plugins import HeatMap
from shapely.geometry import Point, Polygon, box
# from mpl_toolkits.basemap import Basemap
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from typing import List, Tuple
##MACHINE LEARNING##
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,DBSCAN
from scipy.stats import multivariate_normal, chi2
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
# from pycave
# from pycave.bayes import GaussianMixture
from scipy.optimize import minimize
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler
##DATABASE##
# from AISStreamCMMI.dbGetter import DBGetter


from itertools import cycle


################################################
################# LOADING DATA #################
ports_coords= {
    "Antwerp": [
        
            [
                50.9590516988698,
                1.930473
            ],
            [
                52.05364406022604,
                5.025225
            ]
        
    ],
    "Limassol": [
        
            [
                34.405838,
                32.85238475
            ],
            [
                34.802594,
                33.53114225
            ]
        
    ],
    "Los Angeles": [
        
            [
                33.603517,
                -118.388599
            ],
            [
                33.821797000000004,
                -117.990769
            ]
        
    ],
    "Livorno": [
        
            [
                43.30184342152549,
                9.81141621825099
            ],
            [
                43.79233489237254,
                10.583892414540053
            ]
        
    ],
    "Singapore": [
        
            [
                0.639762,
                102.24655125
            ],
            [
                2.628318,
                105.63587174999999
            ]
        
    ],
    "Southampton": [
        
            [
                50.47465275,
                -1.79641025
            ],
            [
                51.06477825,
                -0.4421427499999999
            ]
        
    ],
    "Aukland": [
        
            [
                -36.91479161832509,
                174.56027969373062
            ],
            [
                -36.748677885324085,
                174.9815367127736
            ]
        
    ],
    "Gdansk": [
        
            [
                54.27450125,
                18.378808
            ],
            [
                54.63983975,
                19.232662
            ]
        
    ],
    "Busan": [
        
            [
                34.90344125,
                128.61500548070657
            ],
            [
                35.261567750000005,
                129.43637250385868
            ]
        
    ],
    "Algeciras": [
        
            [
                35.6394875,
                -5.93308625
            ],
            [
                36.4429685,
                -4.30846475
            ]
        
    ],
    
    "cape town": [
        
            [
                -34.1454345,
                17.95830025
            ],
            [
                -33.3645135,
                18.63808675
            ]
        
    ],
    "Ambarli": [
        
            [
                40.58673125,
                28.409385
            ],
            [
                41.19855575,
                29.247789
            ]
        
    ],
}
def prepare_port_data(port,ship_types,POI,spyre = False,beefy=False,entopy=False,all=False):
    if spyre:
        ais_data = load_and_filter_ais()
        ais_data = ais_data.rename(columns={"longitude": "Longitude", "latitude": "Latitude","mmsi": "MMSI","speed": "Sog","course": "Cog","heading": "TrueHeading","ship_and_cargo_type": "ShipType",})
    
    else:
        
        if beefy:
            foo = '_beeefy'
        else:
            foo = ''
        if entopy:
            foo_e = '_entropy'
        else:
            foo_e = ''
        if all:
            filename = 'all_data'
            # print(os.path.join(f'data/{filename}',port+'.csv'))
            if os.path.exists(os.path.join(f'data/{filename}',port+'.csv')):
                # ais_data = pickle.load(open(os.path.join(f'data/{filename}',port+'.pkl'), 'rb'))
                ais_data = pd.read_csv(os.path.join(f'data/{filename}',port+'.csv'))
                # print(ais_data.columns)
                ais_data = ais_data.rename(columns={"longitude": "Longitude", "latitude": "Latitude","MetaData.MMSI_String": "MMSI","Sog": "speed","course": "Cog","heading": "TrueHeading","ship_and_cargo_type": "ShipType",'TimeUtc':'timestamp'})
    
            else:
                print('aaaaaaaaaaaaarggrgrgrgrgg')
                return None
            ais_data = filter_ship_type(ais_data,ship_types)
            if POI is not None:
                # print(ti)
                ais_data['timestamp'] =  pd.to_datetime(ais_data['timestamp']).dt.tz_convert(None)

                ais_data = filter_POI(ais_data,POI)
            # ais_data = ais_data[~ais_data.isin([np.nan, np.inf, -np.inf]).any(1)]

            ais_data['timestamp'] = pd.to_datetime(ais_data['timestamp'])
            return ais_data

        else:
            filename = ''
        

        # print(os.path.join(f'data/week_data_every_port{foo}{foo_e}',port))
        if os.path.exists(os.path.join(f'data/week_data_every_port{foo}{foo_e}',port)):
            ais_data = pickle.load(open(os.path.join(f'data/week_data_every_port{foo}{foo_e}',port), 'rb'))
        else:
            print('aaaaaaaaaaaaarggrgrgrgrgg')
    ais_data = filter_ship_type(ais_data,ship_types)
    
    if POI is not None:
        ais_data = filter_POI(ais_data,POI)
    ais_data = ais_data[~ais_data.isin([np.nan, np.inf, -np.inf]).any()]

    ais_data['timestamp'] = pd.to_datetime(ais_data['timestamp'])
    return ais_data

def load_and_filter_ais():
    if not os.path.exists('data/merged_csv'):
        csv_paths = sorted(list(glob('data/spyre_ais_data/*')))
        merged_csv = merge_spreedsheets(csv_paths)
        pickle.dump(merged_csv,  open('data/processed_spyre/merged_csv', 'wb'))
    else:
        merged_csv = pickle.load( open('data/processed_spyre/merged_csv', 'rb'))
        print(merged_csv.shape)
    if not os.path.exists('data/processed_spyre/position_data'):
        print('positional data is not cashed')
        # Position messages
        position_msgs = [1, 2, 3, 4, 18, 27]
        position_data = merged_csv[merged_csv['msg_type'].isin(position_msgs)]
        # print(position_data.columns)
        # exit()
        position_data = position_data[['timestamp', 'mmsi', 'status', 'speed', 'longitude', 'latitude','course','heading']]
        
        position_data = position_data.dropna()#.columns[position_data.notna().any()]
        position_data['timestamp'] = pd.to_datetime(position_data['timestamp']).dt.tz_convert(None)
        pickle.dump(position_data,  open('data/processed_spyre/position_data', 'wb'))
    else:
        position_data = pickle.load( open('data/processed_spyre/position_data', 'rb'))
    if not os.path.exists('data/processed_spyre/static_data'):
        print('static data is not cashed')
        # Static messages
        static_msgs = [5, 24]
        static_data = merged_csv[merged_csv['msg_type'].isin(static_msgs)]
        static_data = static_data[['timestamp', 'mmsi', 'ship_and_cargo_type', 'imo']]
        static_data = static_data.dropna()#.columns[static_data.notna().any()]
        static_data['timestamp'] = pd.to_datetime(static_data['timestamp']).dt.tz_convert(None)
        pickle.dump(static_data,  open('data/processed_spyre/static_data', 'wb'))
    else:
        static_data = pickle.load( open('data/processed_spyre/static_data', 'rb'))
    if not os.path.exists('data/processed_spyre/other_msgs'):
        # Other messages
        other_msgs = [6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 21]
        other_data = merged_csv[merged_csv['msg_type'].isin(other_msgs)]

        # dump information to that file
        pickle.dump(other_data,  open('data/processed_spyre/other_data', 'wb'))
    else:
        # load information to that file
        other_data = pickle.load( open('data/processed_spyre/other_data', 'rb'))

    filtered_static_data = static_data
    filtered_position_data = position_data
    mmsi_to_ship_type = \
    filtered_static_data.drop_duplicates(subset=['mmsi'])[['mmsi', 'ship_and_cargo_type']].set_index('mmsi').to_dict()[
        'ship_and_cargo_type']
    
    

    filtered_position_data['ship_and_cargo_type'] = filtered_position_data['mmsi'].map(mmsi_to_ship_type)
    filtered_position_data['offsetted_timestamp'] = filtered_position_data['timestamp']
    
    return filtered_position_data

def load_db_data(AOI, POI, ship_types, max_speed):
    getter = DBGetter()

    get = getter.getPositionReport(col_subset=['MMSI','Latitude', 'Longitude', 'Time_UTC' ,'Cog', 'Sog','TrueHeading'],bounding_box= AOI,time_subset=POI,sog_range=(0,max_speed))
    position_data = pd.DataFrame(get.fetchall(), columns=['MMSI','Latitude', 'Longitude', 'timestamp' ,'Cog', 'Sog','TrueHeading'])
    

    static_data = pd.DataFrame(getter.getStaticDataSubSet(col_subset=['MMSI','ShipType'],table='ShipStaticData'),columns=['MMSI','ShipType'])
    mmsi_to_ship_type = static_data.drop_duplicates(subset=['MMSI']).set_index('MMSI').to_dict()['ShipType']
    position_data['ShipType'] = position_data['MMSI'].map(mmsi_to_ship_type)

    if ship_types is not None:    
        position_data = position_data[(position_data['ShipType'] >= ship_types[0]) & (position_data['ShipType'] <= ship_types[-1])  ]
    return position_data

##############################################
################# METRTICS #################

def calculate_hausdorff(rects: List[Polygon], rects_b: List[Polygon], threshold: float) -> Tuple[float, float, float]:
    """
    Calculate the Hausdorff distance, precision, and recall between two lists of polygons.

    :param rects: List of ground truth polygons.
    :param rects_b: List of predicted polygons.
    :param threshold: Threshold for Hausdorff distance to consider a match.
    :return: Tuple of average Hausdorff distance, precision, and recall.
    """
    def calculate_hausdorff_distance(poly1: Polygon, poly2: Polygon) -> float:
        return poly1.hausdorff_distance(poly2)

    # Calculate Hausdorff distance for each pair and find matches
    total_distance = 0
    matches = 0
    for pred_poly in rects_b:
        closest_distance = float('inf')
        for gt_poly in rects:
            distance = calculate_hausdorff_distance(pred_poly, gt_poly)
            if distance < closest_distance:
                closest_distance = distance
        total_distance += closest_distance
        if closest_distance < threshold:
            matches += 1

    # Calculate average distance, precision, and recall
    average_distance = total_distance / len(rects_b) if rects_b else 0
    precision = matches / len(rects_b) if rects_b else 0
    recall = matches / len(rects) if rects else 0

    return average_distance, precision, recall

# def custom_evaluation(predicted_polygons, true_polygons, iou_threshold=0.5):

def calculate_aggregate_iou(label, predictions):
    # Calculate the union of all predictions that intersect the label
    intersecting_predictions = [pred for pred in predictions if pred.intersects(label)]
    if not intersecting_predictions:
        return 0.0
    # Create a new polygon that is the union of all intersecting predictions
    aggregated_prediction = unary_union(intersecting_predictions)
    return calculate_iou(aggregated_prediction, label)
def non_maximum_suppression(predictions, iou_threshold=0.5):
    scores = [Polygon(pred).area for pred in predictions]
    indices = sorted(range(len(predictions)), key=lambda i: scores[i], reverse=True)

    kept_indices = []
    while indices:
        current = indices.pop(0)
        kept_indices.append(current)
        predictions[current] = None  # Mark as processed
        indices = [
            i for i in indices
            if predictions[i] is None or calculate_iou(Polygon(predictions[current]), Polygon(predictions[i])) < iou_threshold
        ]
    return [predictions[i] for i in kept_indices if predictions[i] is not None]

def literature_evaluation(labels, predictions, iou_threshold):
    matched_indices = set()
    prediction_scores = []
    binary_labels = []

    # Calculate IoU for each prediction against each label
    for pred in predictions:
        pred_poly = Polygon(pred)
        max_iou = 0
        for label in labels:
            label_poly = Polygon(label)
            iou = calculate_iou(pred_poly, label_poly)
            max_iou = max(max_iou, iou)
        prediction_scores.append(max_iou)
        binary_labels.append(1 if max_iou >= iou_threshold else 0)
        if max_iou >= iou_threshold:
            matched_indices.add(predictions.index(pred))

    true_positives = sum(binary_labels)
    false_positives = len(predictions) - len(matched_indices)
    false_negatives = len(labels) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calculate AUPRC
    precisions, recalls, thresholds = precision_recall_curve(binary_labels, prediction_scores)
    auprc = auc(recalls, precisions)

    # Debugging: Print precision and recall at each threshold
    # for p, r, t in zip(precisions, recalls, thresholds):
    #     print(f"Threshold: {t}, Precision: {p}, Recall: {r}")

    # print(recalls, precisions, auprc)
    return precision, recall, f1_score, auprc
def custom_evaluation(labels, predictions, iou_threshold):
    true_positives = 0
    false_positives = len(predictions)  # Start with all predictions as false positives
    false_negatives = len(labels)  # Start with all labels as false negatives
    precision_recall_points = []
    
    # Calculate true positives and false negatives based on label overlap
    for label in labels:
        agg_iou = calculate_aggregate_iou(label, predictions)
        if agg_iou >= iou_threshold:
            true_positives += 1
            false_negatives -= 1
        precision_recall_points.append((agg_iou, true_positives, false_negatives))
    # print(precision_recall_points)
    # exit()
    # Calculate false positives
    for prediction in predictions:
        if any(label.intersects(prediction) for label in labels):
            false_positives -= 1  # Remove from false positives if there's an intersection
    
    # Calculate precision and recall for each point
    precision_recall_curve = []
    for agg_iou, tp, fn in precision_recall_points:
        if tp + false_positives > 0:
            precision = tp / (tp + false_positives)
        else:
            precision = 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        precision_recall_curve.append((precision, recall))

    # Calculate AUPRC

    precision_recall_curve.sort(key=lambda x: x[1])  # Sort by recall
    precision_values, recall_values = zip(*precision_recall_curve)
    auprc = auc(recall_values, precision_values)
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    # print(precision, recall, f1_score, auprc)
    return precision, recall, f1_score,auprc
def custom_evaluation_withconf(labels, predictions, confidences, iou_threshold):
    # Pair each prediction with its confidence score
    predictions_with_confidence = list(zip(predictions, confidences))

    # Sort predictions by confidence in descending order
    sorted_predictions = sorted(predictions_with_confidence, key=lambda x: x[1], reverse=True)

    true_positives = 0
    false_positives = 0
    false_negatives = len(labels)
    precision_recall_points = []

    # Iterate over sorted predictions
    for prediction, confidence in sorted_predictions:
        matched = False
        for label in labels:
            if calculate_iou(prediction, label) >= iou_threshold:
                matched = True
                true_positives += 1
                false_negatives -= 1
                break  # Assuming one prediction can only match with one label
        if not matched:
            false_positives += 1

        # Calculate precision and recall at this threshold
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 1.0
        precision_recall_points.append((precision, recall))

    # Extract precision and recall values
    precision_values, recall_values = zip(*precision_recall_points)

    # Calculate AUPRC
    auprc = auc(recall_values, precision_values)

    # Calculate the final precision, recall, and F
    final_precision = precision_values[-1]
    final_recall = recall_values[-1]
    f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall) if final_precision + final_recall > 0 else 0
    return final_precision, final_recall, f1_score, auprc
def calculate_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union > 0 else 0

# def calculate_iou(poly1, poly2):
#     intersection = poly1.intersection(poly2).area
#     union = poly1.union(poly2).area
#     return intersection / union if union > 0 else 0
# def calculate_iou(poly1, poly2):
#     intersection_area = poly1.intersection(poly2).area
#     union_area = poly1.union(poly2).area
#     return intersection_area / union_area if union_area != 0 else 0

def calculate_map_precision_recall_debug(pred_polygons, gt_polygons, iou_threshold=0.5):
    """
    Calculate the Mean Average Precision (mAP), precision, and recall for two lists of polygons.
    This function is designed for debugging and sanity checks.

    :param pred_polygons: List of predicted polygons (list of lists of tuples).
    :param gt_polygons: List of ground truth polygons (list of lists of tuples).
    :param iou_threshold: IoU threshold to consider for a valid match.
    :return: Mean Average Precision (mAP), precision, and recall.
    """
    # Convert each set of coordinates to a Shapely Polygon
    shapely_pred_polygons =  pred_polygons
    shapely_gt_polygons = gt_polygons

    # Initialize variables for True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = 0
    FP = 0
    FN = 0

    # Create a list to track which ground truth polygons have been matched
    matched_gt = [False] * len(shapely_gt_polygons)

    for pred_poly in shapely_pred_polygons:
        match_found = False

        for gt_index, gt_poly in enumerate(shapely_gt_polygons):
            iou = calculate_iou(pred_poly, gt_poly)

            if iou >= iou_threshold and not matched_gt[gt_index]:
                TP += 1
                matched_gt[gt_index] = True
                match_found = True
                break

        if not match_found:
            FP += 1

    # False negatives are ground truth polygons that were not matched
    FN = len(shapely_gt_polygons) - sum(matched_gt)

    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate AP (Area under the precision-recall curve)
    # In this case, AP is equivalent to precision since recall changes from 0 to 1 in one step
    ap = precision

    return ap, precision, recall
def intersection_over_union(rects_a, rects_b):
    """Calculate the intersection-over-union for every pair of rectangles
    in the two arrays.

    Arguments:
    rects_a: array_like, shape=(M, 5)
    rects_b: array_like, shape=(N, 5)
        Rotated rectangles, represented as (centre x, centre y, width,
        height, rotation in quarter-turns).

    Returns:
    iou: array, shape=(M, N)
        Array whose element i, j is the intersection-over-union
        measure for rects_a[i] and rects_b[j].

    """
    m = len(rects_a)
    n = len(rects_b)
    if m > n:
        # More memory-efficient to compute it the other way round and
        # transpose.
        return intersection_over_union(rects_b, rects_a).T

    # Convert rects_a to shapely Polygon objects.
    # polys_a = [rect_polygon(*r) for r in rects_a]
    polys_a = rects_a
    # Build a spatial index for rects_a.
    index_a = rtree.index.Index()
    for i, a in enumerate(polys_a):
        index_a.insert(i, a.bounds)

    # Find candidate intersections using the spatial index.
    iou = np.zeros((m, n))
    for j, rect_b in enumerate(rects_b):
        b = Polygon(rect_b)
        for i in index_a.intersection(b.bounds):
            a = polys_a[i]
            intersection_area = a.intersection(b).area
            if intersection_area:
                iou[i, j] = intersection_area / a.union(b).area

    return iou
    
def min_area_bounding_rect(hull_points):
    """
    Find the minimum area bounding rectangle for a set of points.

    Parameters:
    hull_points (ndarray): Numpy array of points forming the convex hull.

    Returns:
    ndarray: Coordinates of the four corners of the minimum area rectangle.
    """

    hull = ConvexHull(hull_points)

 


def min_area_bounding_rect(hull_points):
    hull = ConvexHull(hull_points)
    points = hull_points[hull.vertices]

    # Calculate all the edge angles
    edges = np.diff(points, axis=0, append=points[0:1])
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    # Keep unique angles and remove redundancy due to the periodicity of the angles
    angles = np.unique(angles % (np.pi / 2))

    # Find the rotation matrices for each angle
    rotations = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

    # Initialize variables to store the minimum area rectangle
    min_area = np.inf
    min_rect = np.zeros((4, 2))

    # Calculate the bounding box for each rotation
    for rotation in rotations:
        # Rotate the points to align with the edge
        r = np.array([[rotation[0], -rotation[1]], [rotation[1], rotation[0]]])
        rot_points = np.dot(points, r)

        # Find the min/max x and y of the rotated points
        min_x, max_x = np.min(rot_points[:, 0]), np.max(rot_points[:, 0])
        min_y, max_y = np.min(rot_points[:, 1]), np.max(rot_points[:, 1])

        # Calculate the area of the bounding box
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            # Store the rectangle coordinates
            min_rect = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])

            # Rotate the rectangle coordinates back
            min_rect = np.dot(min_rect, r.T)

    return min_rect

def compute_unscaled_rotated_rectangles_withconf(data_all, data_trans, gmm, scaler, threshold=0.7):
    # Step 1: Filter Data by Probability Threshold using scaled data
    probabilities = gmm.predict_proba(data_trans)
    max_probs = np.max(probabilities, axis=1)
    filtered_indices = max_probs > threshold

    # Applying the filter to both the original and scaled data
    filtered_data_trans = data_trans[filtered_indices]
    filtered_data_all = data_all[filtered_indices]
    filtered_probs = max_probs[filtered_indices]  # Filtered probabilities

    # Step 2: Compute Cluster Assignments using filtered scaled data
    cluster_assignments = gmm.predict(filtered_data_trans)
    unique_clusters = np.unique(cluster_assignments)
    
    filtered_data_trans = scaler.inverse_transform(filtered_data_trans)

    # Step 3: Compute and Unscale Rotated Rectangles
    rotated_rectangles = {}
    average_confidences = {}  # To store average confidences

    for cluster in unique_clusters:
        # Selecting cluster points from filtered scaled data
        cluster_points_scaled = filtered_data_trans[cluster_assignments == cluster]
        cluster_probs = filtered_probs[cluster_assignments == cluster]  # Corresponding probabilities

        if len(cluster_points_scaled) >= 3 and len(set(cluster_points_scaled[:,0])) > 1:
            hull = ConvexHull(cluster_points_scaled)
            hull_points_scaled = cluster_points_scaled[hull.vertices]
            
            # Calculate the rotated rectangle on scaled data
            rect_scaled = min_area_bounding_rect(hull_points_scaled)
            rotated_rectangles[cluster] = rect_scaled

            # Calculate and store the average confidence for this cluster
            average_confidence = np.mean(cluster_probs)
            average_confidences[cluster] = average_confidence

    return rotated_rectangles, average_confidences
def compute_unscaled_rotated_rectangles(data_all, data_trans, gmm, scaler, threshold=0.7):
    # Step 1: Filter Data by Probability Threshold using scaled data
    probabilities = gmm.predict_proba(data_trans)
    max_probs = np.max(probabilities, axis=1)
    filtered_indices = max_probs > threshold

    # Applying the filter to both the original and scaled data
    filtered_data_trans = data_trans[filtered_indices]
    filtered_data_all = data_all[filtered_indices]

    # Step 2: Compute Cluster Assignments using filtered scaled data
    cluster_assignments = gmm.predict(filtered_data_trans)
    unique_clusters = np.unique(cluster_assignments)
    
    filtered_data_trans = scaler.inverse_transform(filtered_data_trans)
    # print(filtered_data_trans)
    # Step 3: Compute and Unscale Rotated Rectangles
    rotated_rectangles = {}

    for cluster in unique_clusters:
        # Selecting cluster points from filtered scaled data
        cluster_points_scaled = filtered_data_trans[cluster_assignments == cluster]
        # print(cluster_points_scaled)
        # exit()
        # Check if there are enough points to form a convex hull
        # print(len(set(cluster_points_scaled[:,0])))
        if len(cluster_points_scaled) >= 3 and len(set(cluster_points_scaled[:,0]))>1:
            # print(cluster_points_scaled)
            # for i in range(len(cluster_points_scaled)):

            #     print(len(cluster_points_scaled[i]))
            hull = ConvexHull(cluster_points_scaled)
            hull_points_scaled = cluster_points_scaled[hull.vertices]
            
            # Calculate the rotated rectangle on scaled data
            rect_scaled = min_area_bounding_rect(hull_points_scaled)
            # print(rect_scaled)
            # Unscale the rectangle points to original coordinates
            # rect_unscaled = scaler.inverse_transform(rect_scaled)
            rotated_rectangles[cluster] = rect_scaled

    # return rotated_rectangles
    return rotated_rectangles
##############################################
################# CLUSTERING #################
def apply_super_cluster(df_grouped,eps,min_points, metric=None)   :
    centroids = df_grouped.drop_duplicates(subset=['MMSI', 'Cluster'])[['MMSI', 'Cluster', 'Cluster_Latitude', 'Cluster_Longitude']]
    # Exclude the noise clusters (Cluster = -1)
    centroids = centroids[centroids['Cluster'] != -1]
    print(centroids)
    print('meh')
    # Apply DBSCAN on centroids
    db = DBSCAN(eps=eps, min_samples=min_points, metric=metric).fit(np.radians(centroids[['Cluster_Latitude', 'Cluster_Longitude']]))
    centroids['Super_Cluster'] = db.labels_
    # If you want to merge these results back to your main DataFrame:
    df_grouped = df_grouped.merge(centroids[['MMSI', 'Cluster', 'Super_Cluster']], on=['MMSI'], how='left')
    # If you don't want NaN for the noise points from the initial DBSCAN, you can set them to -1 or another value:
    df_grouped['Super_Cluster'].fillna(-1, inplace=True)
    return df_grouped['Super_Cluster'].astype(int),centroids  # Convert to int for better formatting

def apply_dbscan(group,eps=0.0005,min_samples=4, metric=None):
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(np.radians(group[['Latitude', 'Longitude']]))
    group['Cluster'] = db.labels_

    # Calculate centroids for each cluster and map them to the group
    centroids = group.groupby('Cluster').agg(
        Cluster_Latitude=('Latitude', 'mean'),
        Cluster_Longitude=('Longitude', 'mean')
    )

    group = group.join(centroids, on='Cluster')
    return group


############################################
################# PLOTTING #################
def calculate_rectangle_coords(lat, lon, dim_a, dim_b, dim_c, dim_d, heading):
    # Adjust heading by 90 degrees and wrap around if necessary
    heading = (heading + 90) % 360

    # Convert dimensions from meters to degrees
    # One degree of latitude is approximately 111 kilometers
    lat_degree = 111000  
    # Adjust the conversion for longitude based on the latitude
    lon_degree = cos(radians(lat)) * lat_degree

    # Convert ship dimensions to degrees
    dim_a_deg = dim_a / lat_degree
    dim_b_deg = dim_b / lat_degree
    dim_c_deg = dim_c / lon_degree
    dim_d_deg = dim_d / lon_degree

    # Calculate initial corner positions relative to the midpoint
    corners = [
        (lat + dim_a_deg, lon + dim_c_deg),  # Front left
        (lat + dim_a_deg, lon - dim_d_deg),  # Front right
        (lat - dim_b_deg, lon - dim_d_deg),  # Back right
        (lat - dim_b_deg, lon + dim_c_deg)   # Back left
    ]

    # Rotate corners around the midpoint
    rotated_corners = [rotate_point(lat, lon, radians(heading), (lat, lon)) for lat, lon in corners]

    # Return the rotated corners
    return rotated_corners

def rotate_point(x, y, angle, origin):
    # Rotates a point around another point
    ox, oy = origin
    px, py = x - ox, y - oy
    qx = ox + cos(angle) * px - sin(angle) * py
    qy = oy + sin(angle) * px + cos(angle) * py
    return qx, qy
def lat_lon_to_cartesian(lat, lon):
    # Placeholder for conversion from latitude and longitude to Cartesian coordinates
    return (lat, lon)

def cartesian_to_lat_lon(x, y):
    # Placeholder for conversion from Cartesian coordinates back to latitude and longitude
    return (x, y)


def make_rects_for_berths(df):
    # Initialize the Folium map, centered at the first ship's location
    m = folium.Map(location=[df['Latitude'].iloc[0], df['Longitude'].iloc[0]], zoom_start=12)

    for _, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        heading = row['TrueHeading']
        dim_a, dim_b, dim_c, dim_d = row['DimensionA'], row['DimensionB'], row['DimensionC'], row['DimensionD']

        if heading == 511:
            folium.CircleMarker(location=[lat, lon], radius=5, color='green').add_to(m)
        elif all(dim == 0 for dim in [dim_a, dim_b, dim_c, dim_d]):
            folium.CircleMarker(location=[lat, lon], radius=5, color='red').add_to(m)
        else:
            rect_coords = calculate_rectangle_coords(lat, lon, dim_a, dim_b, dim_c, dim_d, heading)
            folium.Polygon(locations=rect_coords, color='blue').add_to(m)

    return m



EARTH_RADIUS = 6378137

# Helper functions to convert meters to degrees
def meters_to_lat(meters):
    return meters / EARTH_RADIUS * (180 / pi)

def meters_to_lon(meters, latitude):
    return meters / (EARTH_RADIUS * cos(radians(latitude))) * (180 / pi)

# Function to generate rectangle points based on ship location, dimensions, and heading
def generate_rectangle(lon, lat, dimA, dimB, dimC, dimD, heading):
    # Convert dimensions from meters to degrees
    length = meters_to_lat(dimA + dimB)
    width = meters_to_lon(dimC + dimD, lat)
    
    # Calculate the angle from the heading
    angle = radians(heading)
    
    # Find the offset for the length and width
    offset_length = (length / 2)
    offset_width = (width / 2)
    
    # Calculate the rectangle's corner points based on the heading
    dx = offset_length * sin(angle)
    dy = offset_length * cos(angle)
    points = []
    points.append((lat - dy - offset_width * cos(angle - pi/2), lon - dx + offset_width * sin(angle - pi/2)))
    points.append((lat + dy - offset_width * cos(angle - pi/2), lon + dx + offset_width * sin(angle - pi/2)))
    points.append((lat + dy + offset_width * cos(angle - pi/2), lon + dx - offset_width * sin(angle - pi/2)))
    points.append((lat - dy + offset_width * cos(angle - pi/2), lon - dx - offset_width * sin(angle - pi/2)))
    
    return points

# Function to merge rectangles into a single polygon using Shapely
def merge_polygons(rectangles):
    polygons = [Polygon(rect) for rect in rectangles]
    # Merge the polygons into one
    merged_polygon = unary_union(polygons)
    return merged_polygon

# Function to create a Folium map from AIS data
def plot_ships_on_berths(ais_data):
    # Initialize the Folium map
    folium_map = folium.Map(location=[ais_data['Latitude'].mean(), ais_data['Longitude'].mean()], zoom_start=12)
    
    # Create rectangles for ships that are not green or red dots
    rectangles = [
        generate_rectangle(row['Longitude'], row['Latitude'], row['DimensionA'], row['DimensionB'], row['DimensionC'], row['DimensionD'], row['TrueHeading'])
        for _, row in ais_data.iterrows() if row['TrueHeading'] != 511 and (row['DimensionA'] != 0 or row['DimensionB'] != 0)
    ]
    
    # Merge the rectangles
    merged_polygon = merge_polygons(rectangles)
    
    # Plot the merged polygon
    if isinstance(merged_polygon, Polygon):
        folium.Polygon(locations=[list(merged_polygon.exterior.coords)], color='blue', fill=True).add_to(folium_map)
    elif isinstance(merged_polygon, MultiPolygon):
        for poly in merged_polygon.geoms:
            folium.Polygon(locations=[list(poly.exterior.coords)], color='blue', fill=True).add_to(folium_map)
    
    # Plot green and red dots
    for _, row in ais_data.iterrows():
        if row['TrueHeading'] == 511:
            folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=3, color='green', fill=True).add_to(folium_map)
        elif row['DimensionA'] == 0 and row['DimensionB'] == 0 and row['DimensionC'] == 0 and row['DimensionD'] == 0:
            folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=3, color='red', fill=True).add_to(folium_map)
    
    return folium_map


def draw_ellipse(center, major_radius, minor_radius, rotation_deg=0, map_obj=None):
    """
    Draws an ellipse on a folium map.
    
    :param center: tuple, center of the ellipse (latitude, longitude).
    :param major_radius: float, major radius in meters.
    :param minor_radius: float, minor radius in meters.
    :param rotation_deg: float, rotation of the ellipse in degrees.
    :param map_obj: folium.Map, map object.
    
    :return: folium.Map object with the ellipse drawn.
    """

    # Create a new folium map if not provided
    if map_obj is None:
        m = folium.Map(location=center, zoom_start=15)
    else:
        m = map_obj
    
    # Number of points for the ellipse
    N = 100
    
    # Angles for the ellipse
    theta = np.linspace(0, 2*np.pi, N)
    
    # Ellipse points in cartesian coordinates
    x = major_radius * np.cos(theta)
    y = minor_radius * np.sin(theta)

    # Rotate ellipse if needed
    if rotation_deg != 0:
        rotation_rad = np.deg2rad(rotation_deg)
        x_rot = x * np.cos(rotation_rad) - y * np.sin(rotation_rad)
        y_rot = x * np.sin(rotation_rad) + y * np.cos(rotation_rad)
        x, y = x_rot, y_rot




    lat = center[0] + meters_to_delta_lat(y)
    lon = center[1] + meters_to_delta_lon(x, center[0])
        
    ellipse_coords = list(zip(lat, lon))
    
    # Draw the ellipse on the map
    folium.Polygon(locations=ellipse_coords, fill=True).add_to(m)
    
    return m

def is_inside_port_coordinates(coord, port_coordinates):
    # print(coord,port_coordinates)
    coord = Polygon(coord)
    port_coordinates = Polygon(port_coordinates)

    if coord.within(port_coordinates):
        return True
    return False
    
    # exit()
    # return True


def find_label_boxes(port='Limassol'):
    port_coords = pickle.load(open(os.path.join(f'data/port_birth_coords','port_coords'), 'rb'))
    port_coordinates = port_coords[port]
    #  = ports_coords[port]
    sefNames = glob('./sefs/*.json')
    # with open('./sefNames.json') as json_file:
    #     sefNames = json.load(json_file)
    types=[]
    coords = []
    for sef in sefNames:
        # print(sef)
        with open(sef) as json_file:
            data = json.load(json_file)
        if 'featureCollection' in data['data']:
            for dt in data['data']['featureCollection']['features']:
                # print(dt['geometry']['coordinates'])
                # print(dt)
                coord = [[y,x] for x,y in dt['geometry']['coordinates'][0]]
                try:
                    if dt['properties']['type'] in ['B-Dry','B-Wet']:
                        if is_inside_port_coordinates(coord, port_coordinates): 
                            # print(coord)
                            # print(Polygon(coord))
                            # exit()
                            coords.append(Polygon(coord))
                except Exception as e:
                    print(str(e))
                    continue
    return coords
def gmm_to_gaussian(gmm):
    """Transform a GMM into a single Gaussian distribution."""
    # Overall mean is the weighted sum of component means
    overall_mean = np.dot(gmm.weights_, gmm.means_)

    # For the covariance, consider both the component covariances and the spread of means
    weighted_covariances = sum(gmm.weights_[i] * gmm.covariances_[i] for i in range(gmm.n_components))
    spread_of_means = sum(gmm.weights_[i] * np.outer(gmm.means_[i] - overall_mean, gmm.means_[i] - overall_mean)
                          for i in range(gmm.n_components))
    overall_covariance = weighted_covariances + spread_of_means
    
    return overall_mean, overall_covariance

def calculate_emd_gmm(gmm1, gmm2, n_samples=1000):
    print('EMD')
    # Sample points from each GMM
    samples_gmm1 = gmm1.sample(n_samples)[0]
    samples_gmm2 = gmm2.sample(n_samples)[0]
    return calculate_emd_vai(samples_gmm1, samples_gmm2)
    # Calculate pairwise distance matrix between samples
    distance_matrix = ot.dist(samples_gmm1, samples_gmm2)
    
    # Uniform distribution over samples
    a, b = np.ones((n_samples,)) / n_samples, np.ones((n_samples,)) / n_samples
    
    # Calculate Earth Mover's Distance
    emd_value = ot.emd2(a, b, distance_matrix)
    
    return emd_value

def estimate_bhattacharyya_distance_gmm(gmm1, gmm2, n_samples=10000):
    # Generate samples from both GMMs
    samples1 = gmm1.sample(n_samples)[0]
    samples2 = gmm2.sample(n_samples)[0]
    dist = calculate_bhattacharyya_distance_vai(samples1,samples2)
    return dist
    # Calculate probability densities (in linear space) for the samples under both models
    prob1_on_1 = np.exp(gmm1.score_samples(samples1))
    prob2_on_1 = np.exp(gmm2.score_samples(samples1))
    prob1_on_2 = np.exp(gmm1.score_samples(samples2))
    prob2_on_2 = np.exp(gmm2.score_samples(samples2))
    
    # Calculate the Bhattacharyya coefficient (BC) using geometric mean of probabilities
    BC1 = np.mean(np.sqrt(prob1_on_1 * prob2_on_1))
    BC2 = np.mean(np.sqrt(prob1_on_2 * prob2_on_2))
    BC = np.sqrt(BC1 * BC2)  # Combine BCs for both sets of samples
    
    # Ensure BC is within valid range [0,1] to avoid negative log
    # BC = np.clip(BC, 0, 1)
    
    # Calculate the Bhattacharyya distance (BD) from the Bhattacharyya coefficient
    bhattacharyya_distance = -np.log(BC)
    
    return bhattacharyya_distance
def plotBoundries(m):
    sefNames = glob('./data/sefs/*.json')
    # with open('./sefNames.json') as json_file:
    #     sefNames = json.load(json_file)
    types=[]
    for sef in sefNames:
        # print(sef)
        with open(sef) as json_file:
            data = json.load(json_file)
        if 'featureCollection' in data['data']:
            for dt in data['data']['featureCollection']['features']:
                # print(dt['geometry']['coordinates'])
                coords=[[y,x] for x,y in dt['geometry']['coordinates'][0]]
                color='white'
                popUpStr=""
                if 'type' in dt['properties']:
                    types.append(dt['properties']['type'])
                    if dt['properties']['type']=='B-Wet':
                        color='blue'
                    elif dt['properties']['type']=='B-Dry':
                        color='red'
                    elif dt['properties']['type']=='T-Wet':
                        color='green'
                    elif dt['properties']['type']=='T-Dry':
                        color='orange'
                    else:
                        color='black'  
                for key,val in dt['properties'].items():
                    popUpStr+=f"{key}: {val}<br>"
                folium.Polygon(coords, color=color, weight=2,popup=popUpStr,fill=True, fill_opacity=0.5).add_to(m)
    return m
def contourf_to_geojson(contourf):
    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.5
    )
    return geojson

def plot_singles_contours_with_labels(gmm,data_,scaler,stds=1,samples=1000,layers=20,m=None):
    labels = gmm.predict(data_)
    # gmm.means_ = scaler.inverse_transform(gmm.means_)
    # gmm.covariances_ = [scaler.scale_[:, None] * c * scaler.scale_ for c in gmm.covariances_]
    means = scaler.inverse_transform(gmm.means_) #mm.means_
    covariances = [scaler.scale_[:, None] * c * scaler.scale_ for c in gmm.covariances_]#gmm.covariances_
    weights = gmm.weights_
    plt.figure()
    # print(set(labels))
    data = scaler.inverse_transform(data_)
    # print(data)
    # print(means)
    # print(covariances)
    data = pd.DataFrame(data, columns = ['Latitude','Longitude'])
    y = np.arange(data['Latitude'].min()-0.05, data['Latitude'].max()+0.05,   0.0001)
    x = np.arange(data['Longitude'].min()-0.05, data['Longitude'].max()+0.05, 0.0001)
    
    Y,X = np.meshgrid(y, x)
    XX = np.array([Y.ravel(), X.ravel()]).T
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink', 'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink','red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink','red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink','red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink','red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink',]
    # print(data)
    m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=10)

    

    ############# plot points #############
    data = data.to_numpy()
    for (lat,lon),label in zip(data,labels):
        color = colors[int(label) % len(colors)] if label != -1 else 'black'  # Using gray for noise (-1)
    
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            popup=f"cluster: {label}<br>Prob: {0}",#<br>STD: {std}",
            color=color,
            fill=True,
            fill_color=color,
        ).add_to(m)    
    ############# plot components #############
    for i, color in zip(range(gmm.n_components), colors[:gmm.n_components]):
        # Evaluate the density for this component only
        # print(covariances)
        Z_component = multivariate_normal(means[i], covariances[i]).pdf(XX).reshape(X.shape) * weights[i]
        z_min = Z_component.min()
        z_max = Z_component.max()
        z_levels = np.linspace(z_min + (z_max - z_min) * 0.01, z_max - (z_max - z_min) * 0.01, layers)


        # Create a contour plot for this component
        contourf = plt.contourf(X, Y, Z_component, levels=z_levels, colors=color, extend='neither')
        
        # Convert to geojson
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            min_angle_deg=3.0,
            ndigits=5,
            stroke_width=1,
            fill_opacity=0.1
        )
        with open('geojson.txt','w') as f:
            f.write(geojson)
        # Add the geojson to Folium
        folium.GeoJson(
            geojson,
            style_function=lambda x: {
                'color':     x['properties']['stroke'],
                'weight':    x['properties']['stroke-width'],
                'fillColor': x['properties']['fill'],
                'opacity':   0.6,
            }
        ).add_to(m)
    plt.close()
    return m

def pick_vis(image_array, ais_df,tci_path,fig,ax,transform,crs):
    # c = np.random.randint(1,5,size=15)
    # def onpick3(event):
    #     ind = event.ind
        # print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))
    
    names = ais_df['ShipType'].tolist()
    ais_df['pixel_y'], ais_df['pixel_x'] = zip(*ais_df.apply(lambda row: coord_to_pixel(row['Latitude'], row['Longitude'],transform,crs), axis=1))
    
    print(ais_df)
    sc = plt.scatter(ais_df['pixel_x'], ais_df['pixel_y'], marker='o', c=ais_df['time_diff'], s=20, edgecolor='white', linewidth=0.5, label="AIS Data",picker=True)
    # plt.scatter(lon,lat,c='red')
    annotation = ax.annotate(
    text='',
    xy=(0, 0),
    xytext=(15, 15), # distance from x, y
    textcoords='offset points',
    bbox={'boxstyle': 'round', 'fc': 'w'},
    arrowprops={'arrowstyle': '->'}
    )
    colors = np.random.randint(1, 5, size=len(ais_df['pixel_y']))
    norm = plt.Normalize(1, 4)
    cmap = plt.cm.PiYG
    annotation.set_visible(False)
    def motion_hover(event):
        annotation_visbility = annotation.get_visible()
        if event.inaxes == ax:
            is_contained, annotation_index = sc.contains(event)
            if is_contained:
                data_point_location = sc.get_offsets()[annotation_index['ind'][0]]
                annotation.xy = data_point_location
                print(annotation_index)
                text_label = ''#'mmsi:{},\n{},\n{},\n{},\nSpeed:{}kn,\nHeading:{} degrees'.format(ais_df['mmsi'].tolist()[annotation_index['ind'][0]],ais_df['timestamp'].tolist()[annotation_index['ind'][0]],ship_type_mapping[names[annotation_index['ind'][0]]],'None','None', 'None')
                annotation.set_text(text_label)

                annotation.get_bbox_patch().set_facecolor(cmap(norm(colors[annotation_index['ind'][0]])))
                annotation.set_alpha(0.4)

                annotation.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annotation_visbility:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', motion_hover)

def plot_empty_folium(name,labels=False,x=0,y=0):
    m = folium.Map(location=[x,y], zoom_start=10)
    if labels:
        plotBoundries(m)
    m.save(name)
    return m

def plot_gmm_heatmap_within_std_on_folium(gmm, map_obj=None, num_samples_per_cluster=100):
    if map_obj is None:
        # Assuming 2D data, use the first two means as the center for the map initialization
        map_obj = folium.Map(location=gmm.means_[0], zoom_start=15)
    
    all_samples = []
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        samples = generate_gaussian_samples_within_std(mean, cov, num_samples_per_cluster)
        all_samples.extend(samples)
    
    # Add the samples to the heatmap
    HeatMap(all_samples, radius=5, blur=1).add_to(map_obj)
    
    return map_obj

def plot_gmm_on_folium(gmm, map_obj=None, means=None, covariances=None):
    
    if means is None:
        means = gmm.means_
    if covariances is None:
        covariances = gmm.covariances_
    # For each Gaussian in the GMM
    for mean, covar in zip(means, covariances):
        # Compute the width, height, and angle of the ellipse representing the covariance
        v, w = np.linalg.eigh(covar)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi
        # print(covar)

        # The factor 2.0 is a scaling factor for the ellipse to be plotted
        ellipse_radius_x = v[0]
        ellipse_radius_y = v[1]
        if ellipse_radius_x < ellipse_radius_y:
            ellipse_radius_x, ellipse_radius_y = ellipse_radius_y, ellipse_radius_x
            angle = angle + 90
        map_obj = draw_ellipse((mean[0], mean[1]), max(ellipse_radius_x, ellipse_radius_y) , min(ellipse_radius_x, ellipse_radius_y), rotation_deg=angle, map_obj=map_obj)
    
    return map_obj

def plot_folium_clusters(df,level='vessel_cluster',port='limassol',plot_port_bounds =False,filename = ''):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',  'beige', 
                'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink',  
                'lightgreen', 'gray',  ]
    
    if level == 'vessel_cluster':
        lat_lon_foo = ''
        cluster_foo = ''
        name = f'clusters_{port}_vessel_level_entropy.html'
    
    elif level == 'super_cluster':
        lat_lon_foo = 'Cluster_'
        cluster_foo = 'Super_'
        name = f'clusters_{port}_super_level_entropy.html'

    elif level == 'single_ship':
        lat_lon_foo = ''
        cluster_foo = ''
        name = f'clusters_{port}_single_ship_entropy.html'

    m = folium.Map(location=[df[lat_lon_foo+'Latitude'].mean(), df[lat_lon_foo+'Longitude'].mean()], zoom_start=10)
    for _, row in df.iterrows():

        if level == 'vessel_cluster':
            color = 'black'
            cluster = row['Cluster']
        else:
            cluster = row['Super_Cluster']
        
        # cluster= cluster+1
        # print(cluster)
        color = 'red' if cluster != -1 else 'black'  # Using gray for noise (-1)
        
        folium.CircleMarker(
            location=(row[lat_lon_foo+'Latitude'], row[lat_lon_foo+'Longitude']),
            radius=5,
            popup=f"MMSI: {row['MMSI']}<br>Cluster: {cluster}",#<br>STD: {std}",
            color=color,
            fill=True,
            fill_color=color,
        ).add_to(m)    
    if plot_port_bounds:
        
        plotBoundries(m)

    if filename == '':
        m.save(name)
    else:
        m.save(filename)
    return m  

def plot_rectsangle_clusters(X,rectangles):
    # Plotting rectangles
    map_center = [np.mean(X[:, 0]), np.mean(X[:, 1])]
    m = folium.Map(location=map_center, zoom_start=12)
# Plotting oriented rectangles
    for rect_coords in rectangles:
        folium.Polygon(locations=list(rect_coords), color='#3186cc', fill=True, fill_opacity=0.2).add_to(m)

    return m
def get_cluster_bounds(data, labels, cluster_id):
    """
    Compute the minimum area rectangle that encloses the points of a given cluster.
    """
    cluster_points = data[labels == cluster_id]
    hull = ConvexHull(cluster_points)
    hull_points = cluster_points[hull.vertices]

    min_area_rect = None
    min_area = np.inf

    for i in range(len(hull_points)):
        edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
        edge_angle = np.arctan2(edge[1], edge[0])

        rotated_points = np.dot(cluster_points - hull_points[i], [[np.cos(-edge_angle), -np.sin(-edge_angle)], [np.sin(-edge_angle), np.cos(-edge_angle)]])
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])

        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            corner_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            reverse_rotated = np.dot(corner_points, [[np.cos(edge_angle), -np.sin(edge_angle)], [np.sin(edge_angle), np.cos(edge_angle)]]) + hull_points[i]
            min_area_rect = reverse_rotated

    return min_area_rect
def plot_contours_with_labels(gmm,data,stds=1,samples=1000,layers=20,m=None):

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',  'pink', 'lightgreen', 'gray']
    m = folium.Map(location=[data[:,0].mean(), data[:,1].mean()], zoom_start=10)
    labels = gmm.predict(data)
    for (lat,lon),label in zip(data,labels):
        color = colors[int(label) % len(colors)] if label != -1 else 'black'  # Using gray for noise (-1)
    
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            popup=f"cluster: {label}<br>Prob: {0}",#<br>STD: {std}",
            color=color,
            fill=True,
            fill_color=color,
        ).add_to(m)    
    m = plot_contours(gmm,data,stds=1,samples=samples,layers=layers,m=m)
    return m


#########################################
################# UTILS #################

def is_on_sea(lat, lon, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    response = requests.get(url).json()

    # Check the response for features indicating a sea or ocean
    for result in response.get('results', []):
        if 'natural_feature' in result.get('types', []):
            return True
    return False

def filter_sea_points(df, api_key):
    sea_points = []
    for _, row in df.iterrows():
        if is_on_sea(row['Latitude'], row['Longitude'], api_key):
            sea_points.append(row)

    return pd.DataFrame(sea_points)

def meters_to_delta_lat(meters):
    """Convert meters to latitude degrees."""
    return meters / 111320
def gmm_trial(n_clusters_start,n_clusters_end,data_even,data_odd,d_even,N_even,d_odd,N_odd):
    MDLs = []
    clusters_range = range(n_clusters_start, n_clusters_end)
    

    for n_clusters in clusters_range:
        # Fit the GMM
        gmm_even = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.0010, n_init=2,max_iter=200,verbose=0,random_state=1)
        gmm_even.fit(data_even)
        gmm_odd = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.0010, n_init=2,max_iter=200,verbose=0,random_state=1)
        gmm_odd.fit(data_odd)

        # gmm_even = CustomGMM(n_components=n_clusters,fixed_variance=0.6,max_iter=1000,tol=0.00001).fit(data_even)
        # gmm_odd = CustomGMM(n_components=n_clusters,fixed_variance=0.6,max_iter=1000,tol=0.00001).fit(data_odd)
        # print(gmm_even)

        # Compute the log likelihood
        log_likelihood_even = np.sum(gmm_even.score_samples(data_even))
        log_likelihood_odd = np.sum(gmm_odd.score_samples(data_odd))
        
        # Compute the number of parameters
        p_even = n_clusters * (d_even + (((d_even + 1) * d_even) / 2))
        p_odd = n_clusters * (d_odd + (((d_odd + 1) * d_odd) / 2))
        
        # Compute the MDL
        MDL_even = -log_likelihood_even + 0.5 * p_even * np.log(N_even)
        MDL_even = MDL_even / N_even
        MDL_odd = -log_likelihood_odd + 0.5 * p_odd * np.log(N_odd)
        MDL_odd = MDL_odd / N_odd
        MDL = (MDL_even + MDL_odd) / 2
        MDLs.append(MDL)



    optimal_clusters_mdl = clusters_range[np.argmin(MDLs)]

    gmm_even = GaussianMixture(n_components=optimal_clusters_mdl, covariance_type='full', tol=0.00010, n_init=3,max_iter=100,verbose=0,random_state=1)
    gmm_even.fit(data_even)
    gmm_odd = GaussianMixture(n_components=optimal_clusters_mdl, covariance_type='full', tol=0.00010, n_init=3,max_iter=100,verbose=0,random_state=1)
    gmm_odd.fit(data_odd)

    kl_div = kl_symm(gmm_even, gmm_odd, n_samples=1e3)
    return kl_div,MDLs,optimal_clusters_mdl
def preprocess_data( port,POI,ship_types,filter_poly=True,filtered_speed=0.1,scale_data=True,interpolate_data=True,filter_heading=10):
    ais_data = prepare_port_data(port,ship_types,POI=POI,spyre = False,beefy=False,entopy=False,all=True)
    port_coords = pickle.load(open(os.path.join(f'data/port_birth_coords','port_coords'), 'rb'))
    port_coord = port_coords[port]
    if filter_poly:
        ais_data = filter_polygon(ais_data,port_coord)
    if filter_speed is not None:
        ais_data = filter_speed(ais_data, filtered_speed)
    filtered_df = ais_data[(ais_data['DimensionA'] != 0) & (ais_data['DimensionB'] != 0) & 
                        (ais_data['DimensionC'] != 0) & (ais_data['DimensionD'] != 0)]

    # Further filter out records where TrueHeading is 511
    ais_data = filtered_df[filtered_df['TrueHeading'] != 511]

    if interpolate_data:
        ais_data = interpolate(ais_data)
    if filter_heading is not None:
        ais_data = filter_heading_changes(ais_data, threshold=filter_heading)
    if scale_data:   
        scaler = StandardScaler()
        scaler = scaler.fit(ais_data[['Latitude', 'Longitude']])
    mmsi_counts = ais_data['MMSI'].value_counts()
    sorted_mmsis = mmsi_counts.sort_values(ascending=False).index.tolist()

    # 3. Split the sorted list of MMSIs using list slicing
    even_mmsis = sorted_mmsis[::2]
    odd_mmsis = sorted_mmsis[1::2]
    # 4. Split the original dataframe
    even_df = ais_data[ais_data['MMSI'].isin(even_mmsis)]
    odd_df = ais_data[ais_data['MMSI'].isin(odd_mmsis)]
    return even_df,odd_df,scaler

def preprocess_trial(even_df,odd_df,scaler,eps,min_points,gen_points=10,sample_frac=1.0):

    even_interpolated_clustered = even_df.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
    data_even = even_interpolated_clustered[even_interpolated_clustered['Cluster']!=-1]

    odd_interpolated_clustered = odd_df.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
    data_odd = odd_interpolated_clustered[odd_interpolated_clustered['Cluster']!=-1]

    data_even = generate_points_in_ship_area(data_even, gen_points)
    data_odd =  generate_points_in_ship_area(data_odd,  gen_points)
    data_even = data_even[['Latitude', 'Longitude']].sample(frac=sample_frac, replace=False, random_state=1)
    data_odd = data_odd[['Latitude', 'Longitude']].sample(frac=sample_frac, replace=False, random_state=1)
    data_even = scaler.transform(data_even)
    data_odd = scaler.transform(data_odd)
    
    d_even = data_even.shape[1]  # number of dimensions
    N_even = data_even.shape[0]  # number of data points
    d_odd = data_odd.shape[1]  # number of dimensions
    N_odd = data_odd.shape[0]  # number of data points
    return data_even,data_odd,d_even,N_even,d_odd,N_odd

def meters_to_delta_lon(meters, lat):
    """Convert meters to longitude degrees at a specified latitude."""
    m_per_deg_long = 111412.84 * np.cos(np.radians(lat)) - 93.5 * np.cos(3 * np.radians(lat)) + 0.118 * np.cos(5 * np.radians(lat))
    return meters / m_per_deg_long

def filter_polygon(df, polygon):
    """
    Subset a dataframe by a given polygon based on its latitude and longitude columns.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe with 'Latitude' and 'Longitude' columns.
    - polygon (list of tuple): List of (longitude, latitude) pairs defining the polygon.
    
    Returns:
    - pd.DataFrame: Subsetted dataframe with points that fall inside the polygon.
    """

    # Convert dataframe to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Latitude, df.Longitude))

    # Create a GeoSeries of the polygon
    poly_gdf = gpd.GeoSeries([Polygon(polygon)])

    # Check which points are within the polygon
    mask = gdf.within(poly_gdf.unary_union)

    # Return the subsetted dataframe
    return df[mask].reset_index(drop=True)
def filter_heading_changes(df, threshold):
    # Function to filter rows within each group
    def filter_group(group):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        # Calculate heading changes
        heading_changes = group['TrueHeading'].diff().abs()
        # Filter based on threshold
        return group[heading_changes <= threshold]

    # Group by MMSI and apply the filter function
    return df.groupby('MMSI').apply(filter_group).reset_index(drop=True)
def gmm_pdf(x, weights, means, covariances):

    K = len(weights)
    pdf = 0.0
    
    for k in range(K):
        # Calculate the PDF for each Gaussian component
        component_pdf = weights[k] * multivariate_normal.pdf(x, mean=means[k], cov=covariances[k])
        pdf += component_pdf

    return pdf

def kl_symm(gmm_p, gmm_q, n_samples=1e7):
    kl_1 = gmm_kl(gmm_p, gmm_q, n_samples=n_samples)
    kl_2 = gmm_kl(gmm_q, gmm_p, n_samples=n_samples)
    return max(kl_1, kl_2)    

def gmm_kl(gmm_p, gmm_q, n_samples=1e7):
    X = gmm_p.sample(n_samples)[0]
    # print(gmm_p.weights_, gmm_p.means_, gmm_p.covariances_)
    p_X = (gmm_pdf(X, gmm_p.weights_, gmm_p.means_, gmm_p.covariances_))
    q_X = (gmm_pdf(X, gmm_q.weights_, gmm_q.means_, gmm_q.covariances_))
    return np.mean(np.log(p_X/(q_X+1e-10)))

def generate_gaussian_samples_within_std(mean, cov, num_samples=1000):
    """Generate sample points from a Gaussian distribution within one standard deviation."""
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    
    # Filter samples outside one standard deviation
    rv = multivariate_normal(mean, cov)
    threshold = rv.pdf(mean)  # PDF value at the mean is the maximum value
    mask = rv.pdf(samples) > threshold * 0.1  # Approximately one standard deviation for 2D Gaussian
    return samples[mask]

def file_to_dict(filename):
    # Open and read the file
    with open(filename, 'r') as file:
        file_content = file.read()

    # Ensure proper formatting for dictionary conversion
    # Add curly braces and quote the keys
    dict_str = "{" + file_content + "}"
    dict_str = dict_str.replace(':', '":').replace(',', ',"').replace('{', '{"')

    # Convert string representation of dictionary to actual dictionary
    # Use ast.literal_eval for safe evaluation
    try:
        dictionary = ast.literal_eval(dict_str)
    except ValueError as e:
        print(f"Error converting string to dictionary: {e}")
        return None
    
    return dictionary

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate distance using Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def filter_speed(ais,speed):
    return ais[(ais['speed'] <= speed) ]

def filter_ship_status(ais,status):
    return ais[(ais['NavigationalStatus'].isin( status)) ]

def coord_to_pixel(lon, lat, transform,crs):
    # with rasterio.open(path) as dataset:
        
    source_crs = pyproj.CRS("EPSG:4326")  # WGS84
    target_crs = pyproj.CRS(crs)
    proj_x, proj_y = pyproj.transform(source_crs, target_crs, lon, lat)
    
    # Convert the projected coordinates to pixel coordinates
    x, y = ~transform * (proj_x, proj_y)
    
    return round(y), round(x)

def ais_within_time_window(image_time, ais_timestamps, hours=10.0):
    """Return AIS timestamps within the defined window before the given image_time."""
    end_time = image_time + pd.Timedelta(hours=0) 
    start_time = image_time - pd.Timedelta(hours=hours) 
    return ais_timestamps[(ais_timestamps >= start_time) & (ais_timestamps <= end_time)]

def filter_ais_by_spatial_extent(ais_df, min_lat, max_lat, min_lon, max_lon):
    """Filter AIS data by spatial extent."""
    # print(ais_df['latitude'].head(2),ais_df['longitude'].head(2) , min_lat, max_lat, min_lon, max_lon)
    # exit()
    return ais_df[
        (ais_df['Latitude'] <= min_lat) &
        (ais_df['Latitude'] >= max_lat) &
        (ais_df['Longitude'] >= min_lon) &
        (ais_df['Longitude'] <= max_lon)
    ]

def sensing_time(path):
    some_tree = ET.parse(path)
    # print(str(some_tree))
    # exit()
    time = ''
    for child in some_tree.getroot():
        # print(child.tag)
        # print(child.attrib)
        if 'General_Info' in child.tag:
            # print(child.attrib)
            # print('meh')
            for att in child.findall('SENSING_TIME'):
                # print('here')
                # print(att.text)
                time = att.text.split('.')[0]
    date = time.split('T')[0]
    year = date.split('-')[0]
    month = date.split('-')[1]
    day = date.split('-')[2]
    times = time.split('T')[1]
    hour = times.split(':')[0]
    minute = times.split(':')[1]
    second = times.split(':')[2].split('.')[0]
    # soup = BeautifulSoup(open(path))
    # sensing_time = soup.findAll('SENSING_TIME')
    
    
    target_time = pd.to_datetime(f'{year}{month}{day} {hour}{minute}{second}', format='%Y%m%d %H:%M:%S')
    return target_time

def pixel_to_coord(x, y, transform,crs):
    # with rasterio.open(path) as dataset:
    #     transform = dataset.transform

    """Convert pixel coordinates to geographical coordinates."""
    lon, lat = transform * (x, y)
    source_crs = pyproj.CRS(crs)
    target_crs = pyproj.CRS("EPSG:4326")  # WGS84
    
    # Perform the transformation
    lon, lat = pyproj.transform(source_crs, target_crs, lon, lat)
    return lon, lat

def merge_spreedsheets(file_paths):
    # Load and concatenate the spreadsheets
    all_data = pd.concat([pd.read_csv(file, low_memory=False) for file in file_paths])
    # Save the merged data to a new spreadsheet
    return all_data

def load_tci(path):
    temp_tcir = rasterio.open(path)
    tci = np.array(temp_tcir.read())
    tci = np.transpose(tci,(1,2,0))
    
    # current_tci_timestamp = correct_path2datetime(path) 
    # exit()
    return tci,temp_tcir

def filter_ship_type(ais,ship_type):
        return ais[(ais['ShipType'] >= ship_type[0]) & (ais['ShipType'] <= ship_type[-1])  ]

def filter_POI(ais,POI):
    """Return AIS timestamps within the defined window before the given image_time."""
    start_time = POI[0] 
    end_time = POI[1]
    return ais[(ais['timestamp'] >= start_time) & (ais['timestamp'] <= end_time)]

def filter_AOI(ais_df, AOI):
    """Filter AIS data by spatial extent."""
    # print(ais_df['latitude'].head(2),ais_df['longitude'].head(2) , min_lat, max_lat, min_lon, max_lon)
    # exit()
    min_lat, max_lat, min_lon, max_lon = AOI
    # print(min_lat, max_lat, min_lon, max_lon)
    return ais_df[
        (ais_df['Latitude'] >= min_lat) &
        (ais_df['Latitude'] <= max_lat) &
        (ais_df['Longitude'] >= min_lon) &
        (ais_df['Longitude'] <= max_lon)
    ]

def geohash_center(ghash):
    # Decode geohash
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(ghash)
    
    return lat, lon

def encode_geohash(lat, lon, precision=7):
    return geohash2.encode(lat, lon, precision=precision)

def latlon_to_utm(lat, lon):
    proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_utm = pyproj.Proj(proj='utm', zone=33, datum='WGS84')
    utm_x, utm_y = pyproj.transform(proj_latlon, proj_utm, lon, lat)
    return utm_x, utm_y

def get_std(df_grouped):


    df_grouped['Rad'] = np.radians(df_grouped['TrueHeading'])

    # Compute mean cosine and sine
    df_grouped['Cos'] = np.cos(df_grouped['Rad'])
    df_grouped['Sin'] = np.sin(df_grouped['Rad'])
    mean_cos = df_grouped.groupby(['MMSI', 'Cluster'])['Cos'].transform('mean')
    mean_sin = df_grouped.groupby(['MMSI', 'Cluster'])['Sin'].transform('mean')

    # Calculate mean resultant length
    R = np.sqrt(mean_cos**2 + mean_sin**2)

    # Derive circular standard deviation
    return np.sqrt(-2 * np.log(R))

def interpolate(ais_data):
    for col in ['Latitude', 'Longitude', 'Cog', 'speed']:
        ais_data[col] = pd.to_numeric(ais_data[col], errors='coerce')
        # Resampling and interpolating function
        def interpolate_group(group):
            group = group.set_index('timestamp').resample('1H').first()
            
            # Interpolate the desired columns
            group[['Latitude', 'Longitude', 'Cog', 'speed']] = group[['Latitude', 'Longitude', 'Cog', 'speed']].interpolate()
            
            # Drop rows where 'MMSI' is NaN which indicates there were no original data for that timestamp
            group = group.dropna(subset=['MMSI'])

            return group.reset_index()
        df_grouped = pd.concat(
    [interpolate_group(group) for _, group in ais_data.groupby('MMSI')],
    ignore_index=True
)
    return df_grouped

def filter_variation_ratio(df, threshold, include_511=False):
    # Filter out the TrueHeading with 511 if include_511 is False
    if not include_511:
        df = df[df['TrueHeading'] != 511]
    
    # Calculate the mean and std dev, ignoring the 511 values
    mean = df['TrueHeading'].replace(511, np.nan).mean()
    std_dev = df['TrueHeading'].replace(511, np.nan).std()

    # Compute the coefficient of variation, ignoring NaN values in the mean
    variation_ratio = std_dev / mean

    # If include_511 is True, replace 511 back to the std and mean before filtering
    if include_511:
        df['Variation_Ratio'] = variation_ratio
        df.loc[df['TrueHeading'] == 511, 'Variation_Ratio'] = 0
    else:
        df['Variation_Ratio'] = variation_ratio

    # Filter the DataFrame based on the variation ratio threshold
    df_filtered = df[df['Variation_Ratio'] <= threshold]

    return df_filtered

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great circle distance between two points on the earth
    R = 6371  # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def angle_difference(angle1, angle2):
    # Calculate the difference between two angles in a circular space
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

def filter_incorrect_angles(df, proximity_threshold, angle_threshold):
    # Initialize a column to mark outliers
    df['Outlier'] = False

    # Convert DataFrame columns to numpy arrays for efficient computation
    lats = df['Latitude'].to_numpy()
    lons = df['Longitude'].to_numpy()
    headings = df['TrueHeading'].to_numpy()

    for index in range(len(df)):
        # Calculate distances and heading differences to all other points
        distances = haversine(lats[index], lons[index], lats, lons)
        heading_diffs = angle_difference(headings[index], headings)

        # Find nearby ships within proximity_threshold and not the same ship
        nearby_mask = (distances < proximity_threshold) & (distances > 0)
        nearby_headings = heading_diffs[nearby_mask]

        # Calculate the average heading difference for nearby ships
        if nearby_headings.size > 0 and np.mean(nearby_headings) > angle_threshold:
            df.at[index, 'Outlier'] = True

    # Filter out the outliers
    return df[df['Outlier'] == False].drop(columns='Outlier')
def rotate_point(x, y, angle):
    """Rotate a point around the origin by an angle in degrees."""
    rad = np.radians(angle)
    x_new = x * np.cos(rad) - y * np.sin(rad)
    y_new = x * np.sin(rad) + y * np.cos(rad)
    return x_new, y_new

def get_rectangle_points(df):
    points_list = []
    
    for index, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        a, b, c, d = row['DimensionA'], row['DimensionB'], row['DimensionC'], row['DimensionD']
        heading = row['TrueHeading']

        # Calculate length and width of the ship
        length = a + b
        width = c + d

        # Calculate positions for 9 points (3 on each side: left, middle, right)
        for i in [-1, 0, 1]:  # -1 for left, 0 for middle, 1 for right
            for j in [-1, 0, 1]:  # -1 for bottom, 0 for center, 1 for top
                x = i * length / 2
                y = j * width / 2

                # Rotate the point according to the true heading
                rotated_x, rotated_y = rotate_point(x, y, heading)

                # Adjust the position according to the ship's location
                final_lat = lat + (rotated_x / 111111)  # Approximate conversion from degrees to meters
                final_lon = lon + (rotated_y / (111111 * np.cos(np.radians(lat))))  # Adjust for longitude lines convergence

                points_list.append((final_lat, final_lon))

    return pd.DataFrame(points_list, columns=['Latitude', 'Longitude'])

# def rotate_point(x, y, angle):
#     """Rotate a point around the origin by an angle in degrees."""
#     rad = np.radians(angle)
#     x_new = x * np.cos(rad) - y * np.sin(rad)
#     y_new = x * np.sin(rad) + y * np.cos(rad)
#     return x_new, y_new


def generate_points_in_ship_area(df, num_points):
    points_list = []
    
    for index, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        a, b, c, d = row['DimensionA'], row['DimensionB'], row['DimensionC'], row['DimensionD']
        heading = row['TrueHeading']
        
        # Calculate length and width of the ship
        length = a + b
        width = c + d

        for _ in range(num_points):
            # Generate a random point within the rectangle
            random_x = np.random.uniform(-b, a)
            random_y = np.random.uniform(-c, d)

            # Rotate the point according to the true heading
            rotated_x, rotated_y = rotate_point(random_x, random_y, heading)

            # Adjust the position according to the ship's location
            final_lat = lat + (rotated_x / 111111)  # Approximate conversion from degrees to meters
            final_lon = lon + (rotated_y / (111111 * np.cos(np.radians(lat))))  # Adjust for longitude lines convergence

            # Create a new dictionary with modified latitude and longitude
            new_row = row.to_dict()
            new_row['Latitude'] = final_lat
            new_row['Longitude'] = final_lon
            points_list.append(new_row)
        
    return pd.DataFrame(points_list)


# import rasterio
# from rasterio.features import rasterize
# from shapely.geometry import mapping

def sample_points_polygon_vai(polygons, n_samples=100):
    """Uniformly sample points from a list of non-overlapping polygons."""
    # Calculate total area
    total_area = sum(poly.area for poly in polygons)
    samples = []
    
    for i,poly in enumerate(tqdm(polygons)):
        # Determine samples for this polygon based on its area
        n_poly_samples = max(1, round((poly.area / total_area) * n_samples))
        
        minx, miny, maxx, maxy = poly.bounds
        poly_samples = []

        while len(poly_samples) < n_poly_samples:
            x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                poly_samples.append([x, y])

        samples.extend(poly_samples)
    
    return np.array(samples)
# def bhattacharyya_distance(mean1, cov1, mean2, cov2):
#     """
#     Calculate the Bhattacharyya distance between two distributions with given means and covariance matrices.
    
#     Parameters:
#     - mean1: Mean vector of the first distribution.
#     - cov1: Covariance matrix of the first distribution.
#     - mean2: Mean vector of the second distribution.
#     - cov2: Covariance matrix of the second distribution.
    
#     Returns:
#     - The Bhattacharyya distance.
#     """
#     cov_mean = (cov1 + cov2) / 2
#     mean_diff = mean2 - mean1
    
#     # First term of the Bhattacharyya distance
#     term1 = 0.125 * np.dot(np.dot(mean_diff.T, np.linalg.inv(cov_mean)), mean_diff)
    
#     # Second term of the Bhattacharyya distance
#     term2 = 0.5 * np.log(det(cov_mean) / np.sqrt(det(cov1) * det(cov2)))
    
#     return term1 + term2
def calculate_bhattacharyya_distance_vai(samples1, samples2):
    """Calculate the Bhattacharyya distance between two sets of samples."""
    mean1, mean2 = np.mean(samples1, axis=0), np.mean(samples2, axis=0)
    print(mean1)
    cov1, cov2 = np.cov(samples1.T), np.cov(samples2.T)
    print(cov1)
    cov_mean = (cov1 + cov2) / 2

    diff_mean = mean2 - mean1
    term1 = 0.25 * np.dot(np.dot(diff_mean.T, np.linalg.inv(cov_mean)), diff_mean)
    term2 = 0.5 * np.log(np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    
    return term1 + term2
from scipy.linalg import det, inv, sqrtm
def bhattacharya_medium(mu1, cov1, mu2, cov2): 
    """
    Calculate the Bhattacharyya distance between two multivariate normal distributions.

    Parameters:
    - mu1, mu2: Mean vectors of the distributions.
    - cov1, cov2: Covariance matrices of the distributions.

    Returns:
    - The Bhattacharyya distance.
    """
    # Intermediate matrix for the distance calculation
    cov_mean = (cov1 + cov2) / 2
    print(np.array(cov_mean).shape)
    # First term: Mahalanobis distance part
    diff = mu2 - mu1
    term1 = 1/8 * np.dot(np.dot(diff.T, inv(cov_mean)), diff)
    
    # Second term: Covariance determinant part
    term2 = 1/2 * np.log(det(cov_mean) / np.sqrt(det(cov1) * det(cov2)))
    
    return term1 + term2
from scipy.sparse import dok_matrix,lil_matrix





def assign_points_to_cells(grid, points, min_x, min_y, resolution):
    """
    Update a pre-defined sparse grid with the frequency of points in each cell.
    """
    grid = grid.copy()
    
    # Determine cell indices for each point
    cell_indices_x = np.floor((points[:,0] - min_x) / resolution).astype(int)
    cell_indices_y = np.floor((points[:,1] - min_y) / resolution).astype(int)
    
    # Ensure indices are within grid bounds
    cell_indices_x = np.clip(cell_indices_x, 0, grid.shape[1] - 1)
    cell_indices_y = np.clip(cell_indices_y, 0, grid.shape[0] - 1)
    
    # Update frequencies in the sparse grid
    for x, y in zip(cell_indices_x, cell_indices_y):
        grid[y, x] += 1  # Sparse matrices handle this indexing efficiently
    
    return grid
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix

def bhatta_mc(gmmp,gmmq,samplesq,space_bounds,type='gmm'):
    if type == 'gmm':
        scoresp = gmmp.score_samples(samplesq)
        probability_p = np.exp(scoresp)
        
        scoresq = gmmq.score_samples(samplesq)
        probability_q = np.exp(scoresq)
    else:
        polygonsp = gmmp
        polygonsq = gmmq
        probability_p = np.array(calculate_probabilities(polygonsp, samplesq, space_bounds))
        probability_q = np.array(calculate_probabilities(polygonsq, samplesq, space_bounds))

    # print(probability_p.shape)
    # Use np.where to safely handle division by zero
    # ratios = np.where(probability_q > 0, np.sqrt(probability_p / probability_q), 0)
    min_x, max_x, min_y, max_y  = space_bounds 

    port_area = (np.sqrt(max_x**2+min_x**2))*(np.sqrt(max_y**2+min_y**2))
    mean =  np.mean(np.sqrt(probability_p * probability_q))
    # print(mean)
    BC = port_area * mean
    # BC = np.sum(ratios)/ probability_p.shape[0] 
    BD = -np.log(BC)   # Check for positive BC to avoid log(0)
    
    # print('BC:', BC)
    # print('BD:', BD)
    return BD
from shapely.strtree import STRtree
def calculate_probabilities(polygons, points, space_bounds):
    """
    Calculate the probability of each point being inside any of the given polygons.
    
    Parameters:
    - polygons: A list of shapely.geometry.Polygon objects.
    - points: A list of coordinate tuples or shapely.geometry.Point objects.
    - space_bounds: The bounds of the space (min_x, min_y, max_x, max_y) containing the polygons.
    
    Returns:
    - A list of probabilities corresponding to each point.
    """
    # Calculate the total area of the space
    total_area = (space_bounds[2] - space_bounds[0]) * (space_bounds[3] - space_bounds[1])
    
    # Create an STRtree for efficient spatial queries
    tree = STRtree(polygons)

    # Create an aggregate polygon to quickly exclude points outside all polygons
    aggregate_polygon = unary_union(polygons) if len(polygons) > 1 else polygons[0]
    bounding_box = box(*aggregate_polygon.bounds)

    # List to store the probability for each point
    probabilities = []

    # Iterate over each point to calculate its probability
    for point in points:
        # Ensure the point is a shapely.geometry.Point object
        if not isinstance(point, Point):
            point = Point(point)
        
        # Default probability if the point is outside the bounding box
        point_probability = 0

        # Check if the point is in the bounding box of the aggregate polygon
        if bounding_box.contains(point):
            # Use spatial index to find potential polygons containing the point
            potential_polygons = tree.query(point)

            for polygon in potential_polygons:
                if polygons[polygon].contains(point):
                    # Calculate probability using the area of the polygon that contains the point
                    point_probability = polygons[polygon].area / total_area
                    break  # Assuming a point can be inside only one polygon

        probabilities.append(point_probability)
    
    return probabilities
# def bhatta_mc_vai(sample1,sample2,samplesq):
#     scoresp = gmmp.score_samples(samplesq)
#     probability_p = np.exp(scoresp)
    
#     scoresq = gmmq.score_samples(samplesq)
#     probability_q = np.exp(scoresq)
    
#     # Use np.where to safely handle division by zero
#     ratios = np.where(probability_q > 0, np.sqrt(probability_p / probability_q), 0)
    
#     BC = np.sum(ratios)/ scoresp.shape[0] 
#     BD = -np.log(BC)   # Check for positive BC to avoid log(0)
    
#     print('BC:', BC)
#     print('BD:', BD)
#     return BD
def Bhattacharyya_discrete(freq_grid_set1, freq_grid_set2):
    # Ensure inputs are in CSR format for efficient element-wise operations
    probs1 = csr_matrix(freq_grid_set1)
    probs2 = csr_matrix(freq_grid_set2)

    # Element-wise multiplication of the two frequency grids
    product = probs1.multiply(probs2)

    # Taking the square root of the element-wise product
    # Since scipy sparse matrices do not support sqrt directly, convert to COO for element-wise operations
    product_sqrt = np.sqrt(product.tocoo().data)

    # Sum the square root of the product to get the Bhattacharyya coefficient (BC)
    BC = np.sum(product_sqrt)

    # It's essential to normalize BC by the total number of samples to ensure it's within [0, 1]
    # Assuming freq_grid_set1 and freq_grid_set2 are already normalized to represent probabilities
    # If they represent raw frequencies, you would need to adjust BC accordingly

    # Calculate Bhattacharyya distance (ensure BC is not zero to avoid log(0))
    BD = -np.log(BC) if BC > 0 else float('inf')

    print('BC:', BC)
    print('BD:', BD)
    return BD

def create_uniform_grid(bounds, cell_size):
    """
    Create a uniform grid within given bounds with a specified cell size using a sparse matrix.
    """
    min_x, max_x, min_y, max_y = bounds
    
    # Calculate the number of cells in each dimension
    num_cells_x = int(np.ceil((max_x - min_x) / cell_size))
    num_cells_y = int(np.ceil((max_y - min_y) / cell_size))
    
    # Adjust max bounds to fit an integer number of cells
    adjusted_max_x = min_x + num_cells_x * cell_size
    adjusted_max_y = min_y + num_cells_y * cell_size
    adjusted_bounds = (min_x, adjusted_max_x, min_y, adjusted_max_y)
    
    # Initialize a sparse grid
    grid = lil_matrix((num_cells_y, num_cells_x), dtype=int)
    
    return grid, adjusted_bounds
# from scipy.stats import wasserstein_distance
from scipy.sparse import isspmatrix
def calculate_emd_vai(prob_grid1, prob_grid2):
    # Convert sparse matrix to dense if necessary
    if isspmatrix(prob_grid1):
        prob1 = prob_grid1.toarray().flatten()
    else:
        prob1 = prob_grid1.flatten()
    
    if isspmatrix(prob_grid2):
        prob2 = prob_grid2.toarray().flatten()
    else:
        prob2 = prob_grid2.flatten()

    # Ensure both distributions sum to 1
    prob1 /= np.sum(prob1)
    prob2 /= np.sum(prob2)

    # Generate the cost matrix
    X, Y = np.meshgrid(range(prob_grid1.shape[1]), range(prob_grid1.shape[0]))
    indices = np.stack([Y.flatten(), X.flatten()], axis=1)
    M = ot.dist(indices, indices, metric='euclidean')

    # Compute the EMD
    emd_distance = ot.emd2(prob1, prob2, M)

    return emd_distance


#########################################
################# CLASSES #################


def regularize_covariance(cov_matrix, reg_val=1e-6):
    # Add a small value to the diagonal elements
    return cov_matrix + np.eye(cov_matrix.shape[0]) * reg_val
# def ensure_positive_semi_definite(cov_matrix, epsilon=1e-6):
#     try:
#         np.linalg.cholesky(cov_matrix)
#         return cov_matrix
#     except np.linalg.LinAlgError:
#         return cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
def ensure_positive_semi_definite(cov_matrix, epsilon=1e-6):
    # Compute the symmetric part of the matrix
    sym_matrix = (cov_matrix + cov_matrix.T) / 2

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)

    # Ensure eigenvalues are above a threshold
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  
# def ensure_positive_semi_definite(cov_matrix, epsilon=1e-6):
#     if np.linalg.det(cov_matrix) <= 0:
#         cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
#     return cov_matrix

class CustomGaussianMixture_enforce(GaussianMixture):

    def __init__(self, n_components=1, fixed_width=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.fixed_width = fixed_width

    def _m_step(self, X, log_resp):
        super()._m_step(X, log_resp)
        self._enforce_fixed_width()
    
    def _enforce_fixed_width(self):
        for k in range(self.n_components):
            # Directly adjust the covariance matrices
            self.covariances_[k][0, 0] = self.fixed_width**2
            # Ensure the covariance matrix is still positive definite
            self.covariances_[k] = ensure_positive_semi_definite(self.covariances_[k])   
class CustomGaussianMixture(GaussianMixture):
    def __init__(self, n_components=1, fixed_width=None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.fixed_width = fixed_width
        self.lagrange_multipliers = np.zeros(n_components)

    
    def _m_step(self, X, log_resp):
        # Override the M-step to include Lagrangian optimization
        self._update_parameters_lagrangian(X, log_resp)

    def _compute_lagrangian(self, X, means, covariances, weights):
        # Compute the log-likelihood
        log_likelihood = self._compute_log_likelihood(X, means, covariances, weights)

        # Compute the constraint term
        constraint_term = self._compute_constraint(covariances)

        # Combine them to form the Lagrangian
        lagrangian = log_likelihood - constraint_term
        return lagrangian

    def _compute_lagrangian(self, X, means, covariances, weights):
        # Compute the Lagrangian, which is the log-likelihood minus the constraint term
        log_likelihood = self._compute_log_likelihood(X, means, covariances, weights)
        constraint_term = self._compute_constraint(covariances)
        return log_likelihood - constraint_term 

    def _compute_log_likelihood(self, X, means, covariances, weights):
        n_samples, _ = X.shape
        log_likelihood = 0
        for i in range(n_samples):
            component_density = 0
            for k in range(self.n_components):
                component_density += weights[k] * multivariate_normal.pdf(X[i], mean=means[k], cov=covariances[k])
            log_likelihood += np.log(component_density)
        return log_likelihood

    def _compute_constraint(self, covariances):
        constraint_term = 0
        for k in range(self.n_components):
            # Assuming we are constraining the variance along the x-axis (index 0)
            actual_variance = covariances[k][0, 0]
            deviation = actual_variance - self.fixed_width**2
            constraint_term += self.lagrange_multipliers[k] * deviation
        return constraint_term

    def _update_parameters_lagrangian(self, X, log_resp):
        learning_rate = 0.01  # Learning rate for the gradient ascent
        max_iterations = 100  # Maximum number of iterations for convergence
        tolerance = 1e-6      # Tolerance for convergence check

        previous_lagrangian = None

        for iteration in tqdm(range(max_iterations), desc="Lagrangian optimization",colour='green'):
            # Compute responsibilities
            resp = np.exp(log_resp - log_resp.max(axis=1)[:, np.newaxis])
            resp /= resp.sum(axis=1)[:, np.newaxis]

            # Update weights
            self.weights_ = self._estimate_weights(resp)

            # Update means
            self.means_ = self._estimate_means(X, resp)

            # Update covariances and Lagrange multipliers
            for k in range(self.n_components):
                # Update covariance
                self.covariances_[k] = self._estimate_covariances(resp, X, k)

                # Update Lagrange multiplier
                deviation = self.covariances_[k][0, 0] - self.fixed_width**2
                self.lagrange_multipliers[k] += learning_rate * deviation

            # Compute the current value of the Lagrangian
            current_lagrangian = self._compute_lagrangian(X, self.means_, self.covariances_, self.weights_)

            # Check for convergence
            if previous_lagrangian is not None and abs(previous_lagrangian - current_lagrangian) < tolerance:
                break

            previous_lagrangian = current_lagrangian

            
    def _estimate_weights(self, resp):
        n_samples = resp.shape[0]
        weights = np.sum(resp, axis=0) / n_samples
        return weights

    def _estimate_means(self, X, resp):
        weighted_sum = np.dot(resp.T, X)
        weights_sum = np.sum(resp, axis=0)[:, np.newaxis]
        means = weighted_sum / weights_sum
        return means

    def _estimate_covariances(self, resp, X, k):
        diff = X - self.means_[k]
        weighted_sum = np.dot(resp[:, k] * diff.T, diff)
        covariance = weighted_sum / np.sum(resp, axis=0)[k]

        # Adjusting for the fixed width constraint
        covariance[0, 0] = self.fixed_width**2


        # Regularize the covariance matrix
        covariance = regularize_covariance(covariance)

        #Ensure the covariance matrix is positive semi-definite
        covariance = ensure_positive_semi_definite(covariance)
        return covariance
 

class CustomGMM:
    def __init__(self, n_components, fixed_variance=1, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.fixed_variance = fixed_variance  # Fixed variance for one of the dimensions
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        
        self.rotations_ = None  # Rotation matrices for each component
        self.scalings_ = None  # Scaling factors for each component (other dimension)
    def predict(self, X):
        """
        Assign each sample in X to the most likely Gaussian component.
        :param X: Data array of shape (n_samples, n_features).
        :return: Array of shape (n_samples,) containing the index of the most likely component.
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    # def _initialize_parameters(self, X):
    #     self.weights_ = np.ones(self.n_components) / self.n_components
    #     kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X)
    #     self.means_ = kmeans.cluster_centers_
    #     self.rotations_ = [np.eye(2) for _ in range(self.n_components)]
    #     self.scalings_ = [np.array([self.fixed_variance, 1.0]) for _ in range(self.n_components)]
    #     self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
    
    def score_samples(self, X):
        """
        Calculate the log likelihood of the data X under the model.
        :param X: Data to be scored, shape (n_samples, n_features).
        :return: Log likelihood of the data.
        """
        log_likelihood = 0
        for k in range(self.n_components):
            cov = self._construct_covariance_matrix(k)
            rv = multivariate_normal(self.means_[k], cov)
            log_likelihood += np.log(np.dot(rv.pdf(X), self.weights_[k]))
        return log_likelihood
    
    def sample(self, n_samples=1):
        """
        Generate random samples from the Gaussian Mixture Model.
        :param n_samples: Number of samples to generate.
        :return: Samples generated from the GMM.
        """
        # Normalize weights to sum to 1
        normalized_weights = self.weights_ / np.sum(self.weights_)

        # Randomly choose a component for each sample
        chosen_components = np.random.choice(self.n_components, size=int(n_samples), p=normalized_weights)
        # print(n_samples)
        # Generate samples
        samples = np.zeros((int(n_samples), self.means_.shape[1]))
        for i, k in enumerate(chosen_components):
            mean = self.means_[k]
            cov = self._construct_covariance_matrix(k)
            samples[i] = np.random.multivariate_normal(mean, cov)

        return samples
    def _initialize_parameters(self, X):
        self.weights_ = np.ones(self.n_components) / self.n_components
        kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(X)
        self.means_ = kmeans.cluster_centers_
        self.rotations_ = [np.eye(2) for _ in range(self.n_components)]
        self.scalings_ = [np.array([self.fixed_variance, 1.0]) for _ in range(self.n_components)]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])
    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihood = 0
 
        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            new_log_likelihood = self._compute_log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) <= self.tol:
                break
            log_likelihood = new_log_likelihood
        
        return self
    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            cov = self._construct_covariance_matrix(k)
            rv = multivariate_normal(self.means_[k], cov)
            responsibilities[:, k] = rv.pdf(X)
        responsibilities *= self.weights_
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    # ... [Other methods remain unchanged]
 
    def _m_step(self, X, responsibilities):
        Nk = responsibilities.sum(axis=0)  # Effective number of points per component
 
        # Update weights
        self.weights_ = Nk / X.shape[0]
 
        # Update means
        self.means_ = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
 
        # Update rotations and scalings while keeping one variance fixed
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov_k = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            self._update_covariance_parameters(k, cov_k)
            self.covariances_[k] = cov_k
    def _update_covariance_parameters(self, k, cov_matrix):
        # Perform eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
 
        # Enforce fixed variance on one of the axes
        eigenvalues[0] = self.fixed_variance

        # Update scaling factors (other dimension) and rotation matrix
        self.scalings_[k] = np.array([self.fixed_variance, eigenvalues[1]])
        self.rotations_[k] = eigenvectors
 
    def _construct_covariance_matrix(self, k):
        # Construct the covariance matrix from the rotation and scaling
        scaling_matrix = np.diag(self.scalings_[k])
        return self.rotations_[k] @ scaling_matrix @ self.rotations_[k].T
        # return self.covariances_
    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            cov = self._construct_covariance_matrix(k)
            rv = multivariate_normal(self.means_[k], cov)
            log_likelihood += np.sum(np.log(np.dot(rv.pdf(X), self.weights_[k])))
        return log_likelihood



def train_and_adjust_gmm(data, n_components, fixed_width,map_obj=None, gmm=None):
    """
    Train a GMM and then adjust the covariance matrices to have a fixed width 
    in the dimension with the smallest variance, while preserving the lengths.

    :param data: Input data for training the GMM.
    :param n_components: Number of components in the GMM.
    :param fixed_width: Desired fixed width (standard deviation) for the smallest dimension.
    :return: Trained and adjusted GMM.
    """
    # Train a standard GMM
    if not gmm:
        gmm = GaussianMixture(n_components=n_components,tol=0.1,max_iter=1000,n_init=50)
        gmm.fit(data)
    # plot_gmm(data, gmm, title='Standard GMM with {} Components'.format(n_components))
        
    # map_obj2 = plot_singles_contours_with_labels(gmm,data,stds=3,samples=100,layers=3,m=map_obj)

    # map_obj2.save(f'figures/test_dbscan/normal.html')
        
    # Adjust the covariance matrices
    for k in range(gmm.n_components):
        # Eigenvalue decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(gmm.covariances_[k])

        # Find the index of the smallest eigenvalue
        smallest_index = np.argmin(eigenvalues)

        # Adjust only the smallest eigenvalue to have the desired fixed width
        # while keeping other eigenvalues (lengths in other dimensions) unchanged
        eigenvalues[smallest_index] = fixed_width ** 2

        # Reconstruct the covariance matrix
        gmm.covariances_[k] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return gmm

def adjust_covariance_for_fixed_width(cov_matrix, fixed_width, dimension=0):
    # Eigenvalue decomposition to get eigenvalues and rotation matrix
    eigenvalues, rotation_matrix = np.linalg.eigh(cov_matrix)

    # Rotate the axis to align with the principal axes of the component
    aligned_dimension = np.argmax(np.abs(rotation_matrix[:, dimension]))

    # Adjust the eigenvalue corresponding to the aligned dimension
    eigenvalues[aligned_dimension] = fixed_width ** 2

    # Reconstruct the covariance matrix
    adjusted_cov_matrix = rotation_matrix @ np.diag(eigenvalues) @ rotation_matrix.T
    return adjusted_cov_matrix
def fit_bayesian_gmm(data, n_components, fixed_width):
    # Initialize the Bayesian GMM
    # Note: scikit-learn's implementation doesn't allow for direct setting of priors on covariance.
    # You would need a more flexible implementation (like PyMC or custom code) for informative priors.
    bayesian_gmm = BayesianGaussianMixture(n_components=n_components,
                                           covariance_type='full',
                                           weight_concentration_prior_type='dirichlet_process',
                                           weight_concentration_prior=100)

    # Fit the model
    bayesian_gmm.fit(data)

    # After fitting, you might need to manually adjust the covariances to respect the fixed width
    for k in range(n_components):
        bayesian_gmm.covariances_[k][0, 0] = fixed_width**2
        bayesian_gmm.covariances_[k] = ensure_positive_semi_definite(bayesian_gmm.covariances_[k])

    return bayesian_gmm
########################################
################# MISC #################













































