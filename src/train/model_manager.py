import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras import layers, models, callbacks, optimizers

def get_model_for_target(target_variable, model_type='linear'):
    """
    Returns a model instance based on the target variable and model type.
    """
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
            return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.005, num_leaves=31, min_data_in_leaf=50, random_state=42)
   
    elif target_variable == 'FCH4':
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=300, random_state=42)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.005, num_leaves=31, min_data_in_leaf=50, random_state=42)
    else:
        raise ValueError("Invalid target variable or model type.")

def build_neural_net(model_type, input_dim):
    """
    Builds a simple sequential neural network using LSTM or GRU.
    """
    model = models.Sequential()
    if model_type == 'lstm':
        model.add(layers.LSTM(12, input_shape=(1, input_dim)))
    elif model_type == 'gru':
        model.add(layers.GRU(12, input_shape=(1, input_dim)))
    model.add(layers.Dense(1))
    return model

def prepare_sequence_data(X, y):
    """
    Reshapes X for sequence models (adds a time dimension).
    """
    if X.ndim == 2:
        X = np.expand_dims(X, axis=1)
    return X, y

def train_and_evaluate_model(target_variable, model_type, X_train, y_train, X_test, y_test):
    """
    Trains the specified model and returns predictions.
    Supports traditional ML models as well as neural network models (LSTM/GRU).
    """
    if model_type in ['lstm', 'gru']:
        X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train)
        X_test_seq, y_test_seq = prepare_sequence_data(X_test, y_test)
        model = build_neural_net(model_type, X_train_seq.shape[-1])
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        early_stop = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
        lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr: 0.001 * np.exp(-0.01 * epoch))
        model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=300, batch_size=512, callbacks=[early_stop, lr_scheduler], verbose=0)
        y_pred = model.predict(X_test_seq).flatten()
        return y_pred
    else:
        # Traditional ML branch
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        X_test_scaled = scaler_X.transform(X_test)
        model = get_model_for_target(target_variable, model_type)
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

def train_final_model(target_variable, model_type, best_features, site_list, feature_map):
    """
    Trains a final model on all available data using the best selected features.
    """
    from data_loader import process_site_data
    data = process_site_data(site_list, best_features, target_variable)
    X = data[best_features]
    y = data[target_variable]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    model = get_model_for_target(target_variable, model_type)
    model.fit(X_scaled, y_scaled)
    
    # Save scalers with the model for later prediction
    model.feature_scaler = scaler_X
    model.target_scaler = scaler_y
    return model

def save_trained_model(model, filename):
    """
    Saves the trained model to a file using joblib.
    """
    joblib.dump(model, filename)
