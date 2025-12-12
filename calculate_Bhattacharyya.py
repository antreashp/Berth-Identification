from help_utils import *
import warnings
warnings.filterwarnings("ignore")



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

    f = open(f'metrics/{period}_{train_type}_bhattacharyya.csv', 'a')
    f.write('port,train_type,bhattacharyya\n')
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

    # print('here')
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

        if not os.path.exists(f'labels/{label_name_dict[port]}.pkl'):
            continue
        

        best_params = file_to_dict(f'results/{train_type}_{period}/best_params_{port}.txt')
        eps = best_params['eps']#-0.000005
        min_points = best_params['min_points']#+5
        tol = 0.00001
        n_clusters = best_params['n_clusters']
        port_coord = port_coords[port]
        rects_b = pickle.load(open(f'labels/{label_name_dict[port]}.pkl', 'rb'))
        if os.path.exists(f'data/preprocessed_data/{port}_{train_type}_{period}_even.pkl'):
            data_even = pickle.load(open(f'data/preprocessed_data/{port}_{train_type}_{period}_even.pkl','rb'))
            data_odd = pickle.load(open(f'data/preprocessed_data/{port}_{train_type}_{period}_odd.pkl','rb'))
            scaler = pickle.load(open(f'data/preprocessed_data/{port}_{train_type}_{period}_scaler.pkl','rb'))
        else:
                
            ais_data = prepare_port_data(port,ship_types,POI=POI,spyre = False,beefy=False,entopy=False,all=True)



            ais_data = filter_polygon(ais_data,port_coord)
            # print(ais_data)
            ais_data = filter_speed(ais_data, 0.1)
            filtered_df = ais_data[(ais_data['DimensionA'] != 0) & (ais_data['DimensionB'] != 0) & 
                            (ais_data['DimensionC'] != 0) & (ais_data['DimensionD'] != 0)]
            ais_data = filtered_df[filtered_df['TrueHeading'] != 511]
            ais_data = ais_data.loc[:,~ais_data.columns.duplicated()].copy()
            # print(ais_data.columns)
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




            # #INTERPOLATE DATA
            print('interpolating...')
            even_interpolated = interpolate(even_df)
            odd_interpolated = interpolate(odd_df)

            print('after interpolatiopn: ' , even_interpolated.shape,odd_interpolated.shape)
            even_interpolated = filter_heading_changes(even_interpolated, threshold=10)
            odd_interpolated = filter_heading_changes(odd_interpolated, threshold=10)
            scaler = StandardScaler()
            scaler = scaler.fit(ais_data[['Latitude', 'Longitude']].to_numpy())
 

            print('DBSCAN Even...')
            even_interpolated_clustered = even_interpolated.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
            data_even = even_interpolated_clustered[even_interpolated_clustered['Cluster']!=-1]
            

            print('DBSCAN odd...')
            odd_interpolated_clustered = odd_interpolated.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
            data_odd = odd_interpolated_clustered[odd_interpolated_clustered['Cluster']!=-1]
                    
            print('Generating points in ship area...')
            data_even = generate_points_in_ship_area(data_even, 20)
            data_odd =  generate_points_in_ship_area(data_odd,  20)
            if sys.argv[1] == 'geohash':
                print('geohashing...')
                data_even['geohash'] = data_even.apply(compute_geohash, axis=1)
                data_odd['geohash'] = data_odd.apply(compute_geohash, axis=1)

                data_even[['geohash_latitude', 'geohash_longitude']] = data_even['geohash'].apply(geohash_lat_lon)
                data_odd[['geohash_latitude', 'geohash_longitude']] = data_odd['geohash'].apply(geohash_lat_lon)
                # Merge to get final DataFrame
                final_even = data_even[['geohash', 'geohash_latitude', 'geohash_longitude']]
                final_odd = data_odd[['geohash', 'geohash_latitude', 'geohash_longitude']]
                final_odd = final_odd.drop_duplicates(subset='geohash', keep='last')
                final_even = final_even.drop_duplicates(subset='geohash', keep='last')

                data_even = final_even[[ 'geohash_latitude', 'geohash_longitude']].to_numpy()
                data_odd = final_odd[[ 'geohash_latitude', 'geohash_longitude']].to_numpy()
            else:
                data_even = data_even[['Latitude', 'Longitude']].sample(frac=1, replace=False, random_state=1)
                data_odd = data_odd[['Latitude', 'Longitude']].sample(frac=1, replace=False, random_state=1)
            print('Scaling...')
            data_even = scaler.transform(data_even)
            data_odd = scaler.transform(data_odd)
            pickle.dump(data_even, open(f'data/preprocessed_data/{port}_{train_type}_{period}_even.pkl', 'wb'))
            pickle.dump(data_odd, open(f'data/preprocessed_data/{port}_{train_type}_{period}_odd.pkl', 'wb'))
            pickle.dump(scaler, open(f'data/preprocessed_data/{port}_{train_type}_{period}_scaler.pkl', 'wb'))

        if os.path.exists(f'models/gmm_{port}_{train_type}_{period}_even_eval.model'):
            print('loading fitted models...')
            gmm_even = pickle.load(open(f'models/gmm_{port}_{train_type}_{period}_even_eval.model', 'rb'))
            gmm_odd = pickle.load(open(f'models/gmm_{port}_{train_type}_{period}_odd_eval.model', 'rb'))

        else:
            print('fitting models...')
            gmm_even = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.00001, n_init=5,verbose=0,random_state=1)
            gmm_even.fit(data_even)
            gmm_odd = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.00001, n_init=5,verbose=0,random_state=1)
            gmm_odd.fit(data_odd)
            print('saving fitted models...')
            pickle.dump(gmm_even, open(f'models/gmm_{port}_{train_type}_{period}_even_eval.model', 'wb'))
            pickle.dump(gmm_odd, open(f'models/gmm_{port}_{train_type}_{period}_odd_eval.model', 'wb'))


        min_x = min(port_coord, key=lambda x: x[0])[0]
        max_x = max(port_coord, key=lambda x: x[0])[0]

        min_y = min(port_coord, key=lambda x: x[1])[1]
        max_y = max(port_coord, key=lambda x: x[1])[1]


        min_x,min_y = scaler.transform([[min_x,min_y]])[0]
        max_x,max_y = scaler.transform([[max_x,max_y]])[0]
        
        bounds = (min_x, max_x, min_y, max_y )  # Example bounds
        
        n_samples = 100000

        fs = open(f'metrics/{period}_{train_type}_{port}_samples.csv', 'a')
        fs.write('bhattacharyya\n')
        print('sampling...')
        bhs = []
        for i in tqdm(range(200),colour='red'):
            x_samples = np.random.uniform(min_x, max_x, size=n_samples)
            y_samples = np.random.uniform(min_y, max_y, size=n_samples)
            samples =  np.column_stack((x_samples, y_samples))

            # print('calculating bhattacharyya...')
            b = bhatta_mc(gmm_even,gmm_odd,samples,bounds,type='gmm')
            # print(b)
            bhs.append( b)
            fs.write(f'{b}\n')

        
        bh = np.mean(np.array(bhs))
        f = open(f'metrics/{period}_{train_type}_bhattacharyya.csv', 'a')
        f.write(f'{port},{train_type},{bh}\n')

        f.close()
        fs.close()
        