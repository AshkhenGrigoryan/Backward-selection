# Backward-selection    
all_features = list(data.columns)
current_selected_features = all_features[:]
removed_features = []
number_of_best_features = 5
rss_scores = {}

def get_test_rss(features, data_train, y_train, data_test, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(data_train[features].values, y_train.to_numpy())
    predictions = lin_reg.predict(data_test[features].values) 
    return RSS(predictions, y_test.to_numpy())
    
while len(current_selected_features) != number_of_best_features:
    rss_scores[len(current_selected_features)] = get_test_rss(current_selected_features, data_train, y_train, data_test, y_test)
    X2 = sm.add_constant(data_train[current_selected_features].values)
    est_ = sm.OLS(y_train, X2)
    est2 = est_.fit()
    # print(est2.summary())
    p_values = est2.summary2().tables[1]['P>|t|'][1:]
    highest_p_feature = p_values.argmax()
    current_selected_features.pop(highest_p_feature)
    
rss_scores[len(current_selected_features)] = get_test_rss(current_selected_features, data_train, y_train, data_test, y_test)
