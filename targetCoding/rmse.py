print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, stregr_clf.predict(X_valid))))
    
