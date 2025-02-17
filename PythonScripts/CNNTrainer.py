from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from results_generator import generate_results


class CNNTrainer:

    def __init__(self, *args):
        self.X_train, self.X_test, self.Y_train, self.Y_test = args
        self.model = Sequential()

    def train_model(self):
        accuracy_nn3 = []
        le = LabelEncoder()
        l1_reg = 0.0001 * 5
        l2_reg = 0.0001 * 5
        regularizer = l1_l2(l1=l1_reg, l2=l2_reg)

        y_train_enc = le.fit_transform(self.Y_train)
        y_test_enc = le.transform(self.Y_test)
        X_train, X_val, y_train_enc, y_val_enc = train_test_split(self.X_train, y_train_enc, test_size=0.2, random_state=0)

        self.model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizer, bias_regularizer=regularizer))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
        self.model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizer, bias_regularizer=regularizer))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(X_train, y_train_enc, epochs=500, batch_size=len(X_train), verbose=1, validation_data=(X_val, y_val_enc), callbacks=[early_stopping])

        y_pred_train = (self.model.predict(X_train) > 0.5).astype("int32")
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        y_prob_train = self.model.predict(X_train)
        y_prob = self.model.predict(self.X_test)

        results = generate_results(self.Y_test, y_pred)
        return results
