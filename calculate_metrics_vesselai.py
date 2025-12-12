
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from shapely import geometry, wkt
from help_utils import *

def calculate_centers(df):
    df.sort_values(['MMSI', 'timestamp'], inplace=True)
    berth_visits = df[df.NavigationalStatus==5].groupby('berth_num')
    lon = berth_visits.Longitude.apply(pd.Series.median).values
    lat = berth_visits.Latitude.apply(pd.Series.median).values
    center_coords = list(zip(lat,lon))
    return center_coords

def dbscan_clusters(gdf):
    coords = calculate_centers(gdf)
    epsilon = 0.05 /6371.0088
    db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    clusters = pd.DataFrame.from_dict({'lat':  [c[0] for c in coords], 'lon':[c[1] for c in coords], 'cluster': db.labels_})
    poly = make_polygons(clusters)
    return poly

def make_polygons(clusters):
    clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
    clusters.reset_index(drop=True, inplace=True)
    clusters['geometry'] = [geometry.Point(xy) for xy in zip(clusters['lon'], clusters['lat'])]
    poly_clusters = gpd.GeoDataFrame()
    gb = clusters.groupby('cluster')
    for y in gb.groups:
        df0 = gb.get_group(y).copy()
        point_collection = geometry.MultiPoint(list(df0['geometry']))
        # point_collection.envelope
        convex_hull_polygon = point_collection.convex_hull
        poly_clusters = poly_clusters._append(pd.DataFrame(data={'cluster_id':[y],'geometry':[convex_hull_polygon]}))
    poly_clusters.reset_index(inplace=True)
    poly_clusters.crs = 'epsg:4326'
    
    return poly_clusters

def validate_polygons(polygons):
    validation_data = cfg.VALIDATION_DATA
    print(polygons.dtypes)
    polygons.geometry = polygons.geometry.buffer(0.0005)
    for poly in polygons.geometry:
        intersect = False
        for line in validation_data.geometry:
            if poly.intersects(line):
                intersect= True
        assert intersect

def select_ship_types(df):
    vessel_types = []
    for types in cfg.VESSEL_TYPES:
        vessel_types = vessel_types + (list(range(types[0], types[1]+1)))
    return df[df.shiptype.isin(vessel_types)]

if __name__ == "__main__":
        
    ship_types = range(70,90)
    period = sys.argv[1]
    port_kind = sys.argv[2]
    f = open(f'vesselai_results/metrics/{period}_bhattacharyya.csv', 'a')
    f.write('port,bhattacharyya\n')
    
    if period == '3days':
        min_date = pd.to_datetime(f'{2023}{10:02d}03 00:00:00', format='%Y%m%d %H:%M:%S')
        max_date = pd.to_datetime(f'{2023}{10:02d}05 23:59:59', format='%Y%m%d %H:%M:%S')
    elif period == '1week':
        min_date = pd.to_datetime(f'{2023}{10:02d}03 00:00:00', format='%Y%m%d %H:%M:%S')
        max_date = pd.to_datetime(f'{2023}{10:02d}10 23:59:59', format='%Y%m%d %H:%M:%S')
    elif period == '2weeks':
        min_date = pd.to_datetime(f'{2023}{10:02d}03 00:00:00', format='%Y%m%d %H:%M:%S')
        max_date = pd.to_datetime(f'{2023}{10:02d}16 23:59:59', format='%Y%m%d %H:%M:%S')
    elif period == '1month':
        min_date = pd.to_datetime(f'{2023}{10:02d}01 00:00:00', format='%Y%m%d %H:%M:%S')
        max_date = pd.to_datetime(f'{2023}{10:02d}30 23:59:59', format='%Y%m%d %H:%M:%S')
    else:
        print('Wrong period')
        exit()

    POI=(min_date,max_date)
    port_coords = pickle.load(open(os.path.join(f'data/port_birth_coords','port_coords'), 'rb'))
    label_name_dict = {'Limassol':'limassol','Gdansk':'gdansk','Algeciras':'algeciras','Aukland':'auckland','Southampton':'southampton','Livorno':'livorno','Singapore':'singapore','Antwerp':'antwerp','Busan':'busan','Ambarli':'ambarli','cape town':'cape','Los Angeles':'los'}
    

    for port in tqdm(ports_coords.keys()):
        if port_kind in ports_coords.keys():
            if port != port_kind:
                continue
        elif port_kind == 'all':
            if port == 'Ambarli':
                continue
        elif port_kind == 'small':
            if port == 'Ambarli' or port == 'Antwerp' or port == 'Singapore'  :
                continue
        elif port_kind == 'large':
            if  port != 'Antwerp' and port != 'Singapore'  :
                continue
        else:
            print('Wrong port kind')
            exit()
        print(port,'######################################################################')
        
        ais_data = prepare_port_data(port,ship_types,POI=POI,spyre = False,beefy=False,entopy=False,all=True)

        port_coord = port_coords[port]

        rects_b = pickle.load(open(f'labels/{label_name_dict[port]}.pkl', 'rb'))
        ais_data = filter_polygon(ais_data,port_coord)
        ais_data.drop(ais_data[(ais_data.speed>3) & (ais_data.NavigationalStatus==5)].index, inplace=True)
        ais_data = ais_data.loc[:,~ais_data.columns.duplicated()].copy()
        mmsi_counts = ais_data['MMSI'].value_counts()

        # 2. Sort MMSIs by their message count
        sorted_mmsis = mmsi_counts.sort_values(ascending=False).index.tolist()

        # 3. Split the sorted list of MMSIs using list slicing
        even_mmsis = sorted_mmsis[::2]
        odd_mmsis = sorted_mmsis[1::2]
        # 4. Split the original dataframe
        even_df = ais_data[ais_data['MMSI'].isin(even_mmsis)]
        odd_df = ais_data[ais_data['MMSI'].isin(odd_mmsis)]
        print('after split: ' , even_df.shape,odd_df.shape)

        even_df = even_df.drop_duplicates(['MMSI', 'timestamp'])
        even_df = even_df.sort_values(['MMSI', 'timestamp'])
        even_df['prev_mmsi'] = even_df.MMSI.shift()
        even_df['new_berth'] = ((even_df.NavigationalStatus.diff()!=0) | (even_df.MMSI!=even_df.prev_mmsi))
        even_df['berth_num'] = even_df.new_berth.cumsum()
        even_df.reset_index(inplace=True)
        # print(even_df)
        odd_df = odd_df.drop_duplicates(['MMSI', 'timestamp'])
        odd_df = odd_df.sort_values(['MMSI', 'timestamp'])
        odd_df['prev_mmsi'] = odd_df.MMSI.shift()
        odd_df['new_berth'] = ((odd_df.NavigationalStatus.diff()!=0) | (odd_df.MMSI!=odd_df.prev_mmsi))
        odd_df['berth_num'] = odd_df.new_berth.cumsum()
        odd_df.reset_index(inplace=True)

        even_poly = dbscan_clusters(even_df)
        odd_poly = dbscan_clusters(odd_df)
        m = folium.Map(location=[even_df.Latitude.mean(), even_df.Longitude.mean()], zoom_start=12)
        for _, row in even_poly.iterrows():
            if row['cluster_id'] == -1:
                continue
            try:
                exterior_coords = [(lon, lat) for lat, lon in row['geometry'].exterior.coords]
            except Exception as e:
                print(e)
                continue                    
    # Add the polygon to the map
            folium.vector_layers.Polygon(locations=exterior_coords,
                                popup=f'Cluster ID: {row["cluster_id"]}',
                                color='blue',
                                fill=True,
                                fill_color='blue').add_to(m)
        for  rect in rects_b:
            # Ensure the first and last points are the same for a polygon
            rect = [(i[0],i[1]) for i in list(rect.exterior.coords)]
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect, color='red', fill=True, fill_opacity=0.2).add_to(m)
        # m = plotBoundries(m)
        m.save(f'vesselai_results/{port}_{period}_vesselai_even.html')
        pickle.dump(even_poly, open(f'vesselai_results/{port}_{period}_vesselai_even.pkl', 'wb'))
        
        
        m = folium.Map(location=[odd_df.Latitude.mean(), odd_df.Longitude.mean()], zoom_start=12)
        for _, row in odd_poly.iterrows():
            if row['cluster_id'] == -1:
                continue
            try:
                exterior_coords = [(lon, lat) for lat, lon in row['geometry'].exterior.coords]
            except Exception as e:
                print(e)
                continue                    
    # Add the polygon to the map
            folium.vector_layers.Polygon(locations=exterior_coords,
                                popup=f'Cluster ID: {row["cluster_id"]}',
                                color='blue',
                                fill=True,
                                fill_color='blue').add_to(m)
        for  rect in rects_b:
            # Ensure the first and last points are the same for a polygon
            rect = [(i[0],i[1]) for i in list(rect.exterior.coords)]
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect, color='red', fill=True, fill_opacity=0.2).add_to(m)
        # m = plotBoundries(m)
        m.save(f'vesselai_results/{port}_{period}_vesselai_odd.html')
        pickle.dump(even_poly, open(f'vesselai_results/{port}_{period}_vesselai_odd.pkl', 'wb'))
        
        for polygon in even_poly['geometry'].tolist():
            if isinstance(polygon, str):
                print('even: ',polygon)
        for polygon in odd_poly['geometry'].tolist():
            if isinstance(polygon, str):
                print('odd: ',polygon)
        even_poly = even_poly[even_poly['cluster_id']!=-1]
        even_poly = [Polygon([(lon, lat) for lat, lon in polygon.exterior.coords])  for polygon in even_poly['geometry'].tolist() if isinstance(polygon, Polygon) ]
        odd_poly = odd_poly[odd_poly['cluster_id']!=-1]
        odd_poly = [Polygon([(lon, lat) for lat, lon in polygon.exterior.coords])  for polygon in odd_poly['geometry'].tolist() if isinstance(polygon, Polygon) ]
        
        
        
        even_rotated_rectangles = []
        for pol in even_poly:
            hull_points = np.array(pol.exterior.coords)
            rect = min_area_bounding_rect(hull_points)
            even_rotated_rectangles.append( rect)
            
        odd_rotated_rectangles = []
        for pol in odd_poly:
            hull_points = np.array(pol.exterior.coords)
            rect = min_area_bounding_rect(hull_points)
            odd_rotated_rectangles.append( rect)
            




        m = folium.Map(location=[even_df.Latitude.mean(), even_df.Longitude.mean()], zoom_start=12)
        for rect in even_rotated_rectangles:
            # Ensure the first and last points are the same for a polygon
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect.tolist(), color='blue', fill=True, fill_opacity=0.5).add_to(m)
        # m = plotBoundries(m)
        for  rect in rects_b:
            # Ensure the first and last points are the same for a polygon
            rect = [(i[0],i[1]) for i in list(rect.exterior.coords)]
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect, color='red', fill=True, fill_opacity=0.2).add_to(m)
        m.save(f'vesselai_results/{port}_{period}_vesselai_rects_even.html')
        pickle.dump(even_rotated_rectangles, open(f'vesselai_results/{port}_{period}_vesselai_rects_even.pkl', 'wb'))

        m = folium.Map(location=[odd_df.Latitude.mean(), odd_df.Longitude.mean()], zoom_start=12)    
        for rect in odd_rotated_rectangles:
            # Ensure the first and last points are the same for a polygon
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect.tolist(), color='blue', fill=True, fill_opacity=0.5).add_to(m)
        # m = plotBoundries(m)
        for  rect in rects_b:
            # Ensure the first and last points are the same for a polygon
            rect = [(i[0],i[1]) for i in list(rect.exterior.coords)]
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect, color='red', fill=True, fill_opacity=0.2).add_to(m)
        m.save(f'vesselai_results/{port}_{period}_vesselai_rects_odd.html')
        pickle.dump(odd_rotated_rectangles, open(f'vesselai_results/{port}_{period}_vesselai_rects_odd.pkl', 'wb'))
    


        even_rotated_rectangles = [Polygon(rect) for rect in even_rotated_rectangles]
        odd_rotated_rectangles = [Polygon(rect) for rect in odd_rotated_rectangles]
        
        
        if len(even_rotated_rectangles) == 0 or len(odd_rotated_rectangles) == 0:
            continue
        all_points = np.vstack([
            np.array(poly.exterior.coords) for poly in even_rotated_rectangles + odd_rotated_rectangles
        ])
        scaler = StandardScaler()
        all_points_scaled = scaler.fit_transform(all_points)
        def reconstruct_polygons_from_scaled_points(original_polygons, scaled_points, start_idx):
            reconstructed_polygons = []
            current_idx = start_idx
            for poly in original_polygons:
                n_points = len(poly.exterior.coords)
                # Use the correct variable passed as an argument
                scaled_poly_coords = scaled_points[current_idx:current_idx + n_points]
                reconstructed_polygons.append(Polygon(scaled_poly_coords))
                current_idx += n_points
            return reconstructed_polygons
        start_idx_for_odd = sum(len(poly.exterior.coords) for poly in even_rotated_rectangles)

        # Now pass all_points_scaled as an argument
        scaled_even_rotated_rectangles = reconstruct_polygons_from_scaled_points(
            even_rotated_rectangles, all_points_scaled, 0
        )

        scaled_odd_rotated_rectangles = reconstruct_polygons_from_scaled_points(
            odd_rotated_rectangles, all_points_scaled, start_idx_for_odd
        )
        n_samples = 100000
        min_x = min(port_coord, key=lambda x: x[0])[0]
        max_x = max(port_coord, key=lambda x: x[0])[0]

        min_y = min(port_coord, key=lambda x: x[1])[1]
        max_y = max(port_coord, key=lambda x: x[1])[1]
        min_x,min_y = scaler.transform([[min_x,min_y]])[0]
        max_x,max_y = scaler.transform([[max_x,max_y]])[0]
        bounds = (min_x, max_x, min_y, max_y )
        n_samples = 100000
        print('sampling...')
        bhs = []
        fs = open(f'vesselai_results/metrics/{period}_{port}_samples.csv', 'a')
        fs.write('bhattacharyya\n')
        for i in tqdm(range(100),colour='red'):
            x_samples = np.random.uniform(min_x, max_x, size=n_samples)
            y_samples = np.random.uniform(min_y, max_y, size=n_samples)
            samples =  np.column_stack((x_samples, y_samples))

            # print('calculating bhattacharyya...')
            b = bhatta_mc(scaled_even_rotated_rectangles,scaled_odd_rotated_rectangles,samples,bounds,type='polys') 
            fs.write(f'{b}\n')
            if b == np.inf:
                print('it was inf')
                continue

            bhs.append(b )
        if len(bhs) == 0:
            continue
        bh = np.mean(np.array(bhs))
        f = open(f'vesselai_results/metrics/{period}_bhattacharyya.csv', 'a')
        f.write(f'{port},{bh}\n')
        f.close()
        fs.close()
