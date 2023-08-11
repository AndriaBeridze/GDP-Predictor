import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class our_model():
    def __init__(self):
        # Define model, involves data loading, preprocessing, and training
        
        # Load dataset
        path = 'cleaned_GDP.csv'
        self.data = pd.read_csv(path)

        # Preprocessing, data is preprocessed, but let's randomize to train
        # rearrange the rows of self.data
        self.data = self.data.sample(frac=1)
        # split into features and target
        target = self.data['GDP ($ per capita)']
        features = self.data[['Phones (per 1000)', 'Infant mortality (per 1000 births)', 'Birthrate', 'Deathrate', 'Net migration', 'Coastline (coast/area ratio)', 'Agriculture', 'Industry', 'Service', 'Arable (%)', 'Crops (%)']]

        # Training
        self.my_rf = RandomForestRegressor(criterion = 'friedman_mse', max_depth = 40, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 90)
        self.my_rf.fit(features, target)


    def predict(self, phones, infant_mortality, birthrate, deathrate, net_migration, coastline, agriculture, industry, service, arable, crops):
        # make a prediction using trained model and return prediction
        
        # are all of these numeric
        # yes

        # make a dataframe from our input features
        data = [phones, infant_mortality, birthrate, deathrate, net_migration, coastline, agriculture, industry, service, arable, crops]
        df = pd.DataFrame([data], columns = ['Phones (per 1000)', 'Infant mortality (per 1000 births)', 'Birthrate', 'Deathrate', 'Net migration', 'Coastline (coast/area ratio)', 'Agriculture', 'Industry', 'Service', 'Arable (%)', 'Crops (%)'])
        
        prediction = self.my_rf.predict(df)
        return prediction

