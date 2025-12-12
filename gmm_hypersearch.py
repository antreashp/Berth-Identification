from help_utils import *
import optuna
from optuna.samplers import TPESampler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


def compute_geohash(row,precision=9):
    return geohash2.encode(row['Latitude'], row['Longitude'], precision)
def geohash_lat_lon(geohash):
    lat, lon = geohash2.decode(geohash)
    return pd.Series([lat, lon], index=['geohash_latitude', 'geohash_longitude'])




if __name__ == '__main__':

    ship_types = range(70,90)   #work on cargo and tanker vessels
    period = sys.argv[2]        #[3days,1week,2weeks,1month]
    train_type = sys.argv[1]    #[geohash,custom]
    port_kind = sys.argv[3]     #[all,small,large,<name of port>]


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
        print()
        print(port,'######################################################################')
        if os.path.exists(f'data/clean_data/{train_type}_{period}_even_{port}.pkl') and os.path.exists(f'data/clean_data/{train_type}_{period}_odd_{port}.pkl'):
            print('Loading data from existing pickle...')
            even_interpolated = pickle.load(open(f'data/clean_data/{train_type}_{period}_even_{port}.pkl','rb'))
            odd_interpolated = pickle.load(open(f'data/clean_data/{train_type}_{period}_odd_{port}.pkl','rb'))
            scaler = pickle.load(open(f'data/clean_data/{train_type}_{period}_scaler_{port}.pkl','rb'))
            
        else:
            print('Loading data from scratch...')
            ais_data = prepare_port_data(port,ship_types,POI=POI,spyre = False,beefy=False,entopy=False,all=True)

            print('records: ' , ais_data.shape)
            port_coord = port_coords[port]
            
            # print(ais_data.shape)

            ais_data = filter_polygon(ais_data,port_coord)
            print('after AOI: ' , ais_data.shape)
            
            ais_data = filter_speed(ais_data, 0.1)
            print('after speed: ' , ais_data.shape)
            
            filtered_df = ais_data[(ais_data['DimensionA'] != 0) & (ais_data['DimensionB'] != 0) & 
                            (ais_data['DimensionC'] != 0) & (ais_data['DimensionD'] != 0)]
            print('after dimensions: ' , ais_data.shape)
            
            # Further filter out records where TrueHeading is 511
            ais_data = filtered_df[filtered_df['TrueHeading'] != 511]
            print('after 511 heading: ' , ais_data.shape)
            
            scaler = StandardScaler()
            scaler = scaler.fit(ais_data[['Latitude', 'Longitude']])
            x = np.linspace(float(ais_data['Longitude'].min()), float(ais_data['Longitude'].max()))
            y = np.linspace(float(ais_data['Latitude'].min()), float(ais_data['Latitude'].max()))
            

            #COUNT MMSIS    
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
            even_interpolated = interpolate(even_df)
            odd_interpolated = interpolate(odd_df)
            print('after interpolatiopn: ' , even_interpolated.shape,odd_interpolated.shape)
            
            even_interpolated = filter_heading_changes(even_interpolated, threshold=10)
            odd_interpolated = filter_heading_changes(odd_interpolated, threshold=10)
            
            print('after heading: ' , even_interpolated.shape,odd_interpolated.shape)
            
            print(even_interpolated.shape,odd_interpolated.shape)
            pickle.dump(even_interpolated,open(f'data/clean_data/{train_type}_{period}_even_{port}.pkl','wb'))
            pickle.dump(odd_interpolated,open(f'data/clean_data/{train_type}_{period}_odd_{port}.pkl','wb'))
            pickle.dump(scaler,open(f'data/clean_data/{train_type}_{period}_scaler_{port}.pkl','wb'))
        best_params = {'eps':0,'min_points':0,'n_clusters':0,'amdl':0,'kl':9999,'mdl_even':0,'mdl_odd':0}
        def objective(trial):
            global best_params

            try:
                eps = trial.suggest_float('eps', 5, 70, log=True)
                
                ms_per_radian = 6371.0088 * 1000
                eps = eps / ms_per_radian
                # eps = 8.535528180393294e-06
                if port_kind == 'small':
                    min_start = 5
                    max_start = 25
                    min_end = 26
                    max_end = 50
                elif port_kind=='large' or port_kind=='Singapore' or port_kind=='Antwerp':
                    min_start = 50
                    max_start = 150
                    min_end = 151
                    max_end = 230
                else:
                    min_start = 5
                    max_start = 25
                    min_end = 26
                    max_end = 50
                    
                min_points = trial.suggest_int('min_points', 2, 25, log=False)
                n_clusters_start =min_start
                n_clusters_end =max_end
                
                even_interpolated_clustered = even_interpolated.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
                data_even = even_interpolated_clustered[even_interpolated_clustered['Cluster']!=-1]

                # #INTERPOLATE DATA
                odd_interpolated_clustered = odd_interpolated.groupby('MMSI').apply(lambda x: apply_dbscan(x,eps=eps,min_samples=min_points, metric='haversine'))
                data_odd = odd_interpolated_clustered[odd_interpolated_clustered['Cluster']!=-1]
                
                data_even = generate_points_in_ship_area(data_even, 10)
                data_odd =  generate_points_in_ship_area(data_odd,  10)

                #GEOHASH
                if sys.argv[1] == 'geohash':
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
                
                data_even = scaler.transform(data_even)
                data_odd = scaler.transform(data_odd)
                
                d_even = data_even.shape[1]  # number of dimensions
                N_even = data_even.shape[0]  # number of data points
                d_odd = data_odd.shape[1]  # number of dimensions
                N_odd = data_odd.shape[0]  # number of data points

                MDLs = []
                clusters_range = range(n_clusters_start, n_clusters_end)
                
                for n_clusters in clusters_range:
                    #Fit the GMMs to splits with the same number of components
                    gmm_even = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.0010, n_init=1,max_iter=200,verbose=0,random_state=1)
                    gmm_even.fit(data_even)
                    gmm_odd = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=0.0010, n_init=1,max_iter=200,verbose=0,random_state=1)
                    gmm_odd.fit(data_odd)


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


                #retrain the GMMs with the optimal number of components
                optimal_clusters_mdl = clusters_range[np.argmin(MDLs)]
                gmm_even = GaussianMixture(n_components=optimal_clusters_mdl, covariance_type='full', tol=0.00010, n_init=3,max_iter=100,verbose=0,random_state=1)
                gmm_even.fit(data_even)
                
                gmm_odd = GaussianMixture(n_components=optimal_clusters_mdl, covariance_type='full', tol=0.00010, n_init=3,max_iter=100,verbose=0,random_state=1)
                gmm_odd.fit(data_odd)

                # Compute the symmetric KL divergence
                kl_div = kl_symm(gmm_even, gmm_odd, n_samples=1e3)
                if best_params['kl']> kl_div and kl_div >= 0:
                    best_params['eps'] = eps
                    best_params['min_points'] = min_points
                    best_params['amdl'] = np.min(MDLs)
                    best_params['kl'] = kl_div
                    best_params['n_clusters'] = optimal_clusters_mdl
                    best_params['mdl_even'] = MDL_even
                    best_params['mdl_odd'] = MDL_odd

                    if not os.path.exists(f'results/{train_type}_{period}'):
                        os.makedirs(f'results/{train_type}_{period}')
                    with open(f'results/{train_type}_{period}/best_params_{port}.txt','w') as f:
                        f.write(f"eps:{best_params['eps']},min_points:{best_params['min_points']},amdl:{best_params['amdl']},kl:{best_params['kl']},n_clusters:{best_params['n_clusters']},mdl_odd:{best_params['mdl_even']},mdl_odd:{best_params['mdl_odd']}")
                    
            except Exception as e:
                print(e)
                return 200
            return kl_div
        sampler = TPESampler(seed=10, n_startup_trials=30)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=100,n_jobs=10)
        if not os.path.exists(f'results/{train_type}_{period}'):
            os.makedirs(f'results/{train_type}_{period}')
        with open(f'results/{train_type}_{period}/best_params_{port}.txt','w') as f:
            f.write(f"eps:{best_params['eps']},min_points:{best_params['min_points']},amdl:{best_params['amdl']},kl:{best_params['kl']},n_clusters:{best_params['n_clusters']},mdl_odd:{best_params['mdl_even']},mdl_odd:{best_params['mdl_odd']}")
        pickle.dump(study,open(f'results/{train_type}_{period}/best_params_{port}_study','wb'))






