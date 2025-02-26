from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras import layers, models

def get_model_for_target(target_variable, model_type='linear'):
    """
    Return an *unfitted* model object based on target_variable and model_type.
    All models here are scikit-learn-like, except for 'lstm'/'gru' where we
    only return a string or a partially built structure so we can finalize it later.
    """
    # Example differences by target:
    if target_variable == 'FC':
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR(kernel='rbf', C=0.1, epsilon=0.01)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.005, num_leaves=31,
                                     min_data_in_leaf=50, random_state=42)
        elif model_type in ['lstm','gru']:
            # We'll build the actual Keras model structure in code that knows X's shape
            return model_type
    
    elif target_variable == 'FCH4':
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=200, random_state=42)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.005, num_leaves=31,
                                     min_data_in_leaf=50, random_state=42)
        elif model_type in ['lstm','gru']:
            return model_type

    raise ValueError("Invalid target variable or model type specified.")

def build_neural_net(model_type, input_dim):
    """
    Builds a simple sequential neural network (LSTM or GRU).
    """
    model = models.Sequential()
    if model_type == 'lstm':
        model.add(layers.LSTM(16, input_shape=(1, input_dim)))
    elif model_type == 'gru':
        model.add(layers.GRU(16, input_shape=(1, input_dim)))

    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
