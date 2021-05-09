#Defite Train Data

X = df.drop(['Close', 'High', 'Low', 'Volume'], axis=1)
y = df['Close']

X = X[max(windows)+1:]
y = y[max(windows)+1:]
X.fillna(value=0, inplace=True)
X.replace([np.inf, -np.inf], 0, inplace=True)


scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle = False)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train, test_size=0.2, shuffle = True)
print("Train", X_train.shape, y_train.shape)
print("Valid", X_valid.shape, y_valid.shape)
print("Test", X_test.shape, y_test.shape)


# https://catboost.ai/docs/search/?query=catboostregressor
model = CatBoostRegressor(iterations=20_000,
                          verbose=100, 
                          # boosting_type = 'Ordered',
                          early_stopping_rounds=200,
                          loss_function = 'RMSE',
                          custom_metric = 'MAE',
                          task_type = 'CPU'
                          )

model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model = True)
model.save_model('base_model_CPU.model')


model_use = CatBoostRegressor()
model_use.load_model('base_model_CPU.model')

y_pred = model_use.predict(X_train)
mse_train = mean_squared_error(y_train,y_pred)
print('mean_squared_error train', math.sqrt(mse_train))
mae_train = mean_absolute_error(y_train,y_pred)
print('mean_absolute_error train', mae_train)

y_pred = model_use.predict(X_valid)
mse_valid = mean_squared_error(y_valid,y_pred)
print('mean_squared_error valid', math.sqrt(mse_valid))
mae_valid = mean_absolute_error(y_valid,y_pred)
print('mean_absolute_error valid', mae_valid)

y_pred = model_use.predict(X_test)
mse_test = mean_squared_error(y_test,y_pred)
print('mean_squared_error test', math.sqrt(mse_test))
mae_test = mean_absolute_error(y_test,y_pred)
print('mean_absolute_error test', mae_test)


X_test_unscaled = scaler.inverse_transform(X_test)
X_test_unscaled = pd.DataFrame(X_test_unscaled, columns=X_test.columns)


X_test_unscaled['Pred_Close'] = y_pred
X_test_unscaled['Close'] = y_test.values

X_test_unscaled = X_test_unscaled[['Open', 'Close', 'Pred_Close']]


X_test_unscaled['Diff'] = X_test_unscaled['Close'] - X_test_unscaled['Open']
X_test_unscaled['Change'] = X_test_unscaled['Diff'] / X_test_unscaled['Open']
X_test_unscaled['Buy'] = X_test_unscaled['Pred_Close'] > X_test_unscaled['Close']



#https://matplotlib.org/

figure(num=None, figsize = (12, 10), dpi=80, facecolor='silver', edgecolor='gray')

plt.subplot(2, 1, 1)
plt.plot(X_test_unscaled[['Close','Pred_Close']])
plt.xlabel('time (d1)')
plt.ylabel('$')
plt.legend(['Close', 'Prediction'])
plt.title('Close vs Predicted Close')
plt.grid(True)

plt.tight_layout()
plt.show()