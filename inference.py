from help_utils import *




def compute_geohash(row,precision=9):
    return geohash2.encode(row['Latitude'], row['Longitude'], precision)
def geohash_lat_lon(geohash):
    lat, lon = geohash2.decode(geohash)
    return pd.Series([lat, lon], index=['geohash_latitude', 'geohash_longitude'])






if __name__ == '__main__':
    
    prob_threshold = 0.7
    ship_types = range(70,90)
    period = sys.argv[2]
    train_type = sys.argv[1]
    port_kind = sys.argv[3]
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

        
        print('here')
        if not os.path.exists(f'labels/{label_name_dict[port]}.pkl'):
            continue
        
        print('here')
        rects_b = pickle.load(open(f'labels/{label_name_dict[port]}.pkl', 'rb'))
        
        ais_data = prepare_port_data(port,ship_types,POI=POI,spyre = False,beefy=False,entopy=False,all=True)

        port_coord = port_coords[port]


        ais_data = filter_polygon(ais_data,port_coord)
        ais_data = filter_speed(ais_data, 0.1)
        filtered_df = ais_data[(ais_data['DimensionA'] != 0) & (ais_data['DimensionB'] != 0) & 
                        (ais_data['DimensionC'] != 0) & (ais_data['DimensionD'] != 0)]
        ais_data = filtered_df[filtered_df['TrueHeading'] != 511]
        ais_data_interpolated = interpolate(ais_data)
        ais_data_interpolated = filter_heading_changes(ais_data_interpolated, threshold=10)
        scaler = StandardScaler()
        scaler = scaler.fit(ais_data[['Latitude', 'Longitude']])
        train_type = sys.argv[1] 
        best_params = file_to_dict(f'results/{train_type}_{period}/best_params_{port}.txt')
        eps = best_params['eps']
        min_points = best_params['min_points']
        tol = 0.00001
        n_clusters = best_params['n_clusters']

        ais_data_clustered = ais_data_interpolated.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
        data = ais_data_clustered[ais_data_clustered['Cluster']!=-1]
        data = generate_points_in_ship_area (data,20)

        if train_type == 'geohash':
            data['geohash'] = data.apply(compute_geohash, axis=1)
            
            data[['geohash_latitude', 'geohash_longitude']] = data['geohash'].apply(geohash_lat_lon)

            # Merge to get final DataFrame
            final = data[['geohash', 'geohash_latitude', 'geohash_longitude']]
            final = final.drop_duplicates(subset='geohash', keep='last')
            
            final = final.rename(columns={'geohash_latitude': 'Latitude', 'geohash_longitude': 'Longitude'})
            data = final[[ 'Latitude', 'Longitude']]
        else:
            data = data[['Latitude', 'Longitude']]

        data_trans = scaler.transform(data)

        d = data_trans.shape[1]  # number of dimensions
        N = data_trans.shape[0]  # number of data points
        gmm = GaussianMixture(n_components=n_clusters,max_iter=1000,tol=tol,verbose=1,n_init=5).fit(data_trans)
        pickle.dump(gmm, open(f'models/gmm_{port}_{train_type}_{period}.model', 'wb'))
        rects,confs = compute_unscaled_rotated_rectangles_withconf(data, data_trans, gmm, scaler, threshold=prob_threshold)

        map_obj = plot_singles_contours_with_labels(gmm,data_trans,scaler,stds=3,samples=1000,layers=5,m=None)
        map_obj.save(f'evaluation/figures/{train_type}_{port}_{period}_precision_recall_{prob_threshold}_contours.html')
        
        first_rect = list(rects.values())[0]
        avg_lat = np.mean(first_rect[:, 0])
        avg_lon = np.mean(first_rect[:, 1])
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        for rect_id, rect in rects.items():
            # Ensure the first and last points are the same for a polygon
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect.tolist(), color='blue', fill=True, fill_opacity=0.5).add_to(m)
        for  rect in rects_b:
            # Ensure the first and last points are the same for a polygon
            rect = [(i[0],i[1]) for i in list(rect.exterior.coords)]
            if not np.array_equal(rect[0], rect[-1]):
                rect = np.vstack([rect, rect[0]])  # Append the first point at the end
            folium.Polygon(locations=rect, color='red', fill=True, fill_opacity=0.3).add_to(m)
        m.save(f'evaluation/figures/{train_type}_{port}_{period}_precision_recall_{prob_threshold}_rects.html')
        rects = [Polygon(rect) for rect in rects.values()]
        if not os.path.exists(f'evaluation/{train_type}_{period}'):
            os.makedirs(f'evaluation/{train_type}_{period}') 
        pickle.dump(rects, open(f'evaluation/{train_type}_{period}/{port}_{prob_threshold}_preds.pkl', 'wb'))
        pickle.dump(confs, open(f'evaluation/{train_type}_{period}/{port}_{prob_threshold}_confs.pkl', 'wb'))