class WrangleHelper:
        """A class to help wrangle data"""
        def split_data(X, y, test_size=0.2, val_size=len(X_test), 
                random_state=42, shuffle=True, stratify=None):
                from sklearn.model_selection import train_test_split

                # Splite dataset into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=test_size, random_state=random_state, 
                        shuffle=shuffle, stratify=stratify)

                # Split training dataset into train and validation sets
                X_train, X_val, y_train, y_val = train_test_split(X_train, 
                        y_train, test_size=val_size, 
                        random_state=random_state, shuffle=shuffle, 
                        stratify=stratify)

                return X_train, X_val, X_test, y_train, y_val, y_test

        def ordinal_encode(X_train, X_val, X_test):
                import category_encoders as ce  

                # Make a copy of each DataFrame to avoid warning
                X_train = X_train.copy()
                X_val = X_val.copy()
                X_test = X_test.copy()

                # Fit the encoder on the DataFrames
                encoder = ce.OrdinalEncoder()
                X_train = encoder.fit_transform(X_train)
                X_val = encoder.transform(X_val)
                X_test = encoder.transform(X_test)

                return X_train, X_val, X_test        

