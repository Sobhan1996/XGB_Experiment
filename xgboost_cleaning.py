import xgboost as xgb
import pickle
import pandas as pd
import sys

sys.path.append('../')


def delete_missing_data(df):
    return df.dropna()


def flatten_data_and_store_it(df, consecutive_rows, file_path):
    num_of_rows = df.shape[0]
    output_df = pd.DataFrame()
    for i in range(0, int(num_of_rows/consecutive_rows)-1):
        if df.iloc[(i + 1) * consecutive_rows - 1, 0] - df.iloc[i * consecutive_rows, 0]:
            window = df.iloc[i*consecutive_rows: (i+1)*consecutive_rows, :]
            flattened_window = window.values.flatten()
            if i == 0:
                flattened_window_df = pd.DataFrame([flattened_window])
                output_df = output_df.append(flattened_window_df).reset_index(drop=True)
            else:
                output_df.loc[i] = flattened_window
    output_df.to_csv(file_path)
    return output_df


def run_xgboost(train_df, test_df, cols, num_cols=[], dump_fpath=None, **xgb_kwargs):
    df_all_preds = pd.DataFrame()
    for col in cols:
        if col in num_cols:
            mod = xgb.XGBRegressor(**xgb_kwargs)
        else:
            mod = xgb.XGBClassifier(**xgb_kwargs)

        train_idxs = train_df[col].notnull()
        df_X = pd.get_dummies(train_df[[c for c in train_df.columns if c != col]])
        df_Y = train_df[col]

        df_train_X, df_train_Y = df_X[train_idxs], df_Y[train_idxs]

        mod.fit(df_train_X, df_train_Y)
        if dump_fpath:
            with open(dump_fpath, 'wb') as f:
                pickle.dump(mod, f)

        test_df_X = pd.get_dummies(test_df[[c for c in test_df.columns if c != col]])
        preds = mod.predict(test_df_X)
        df_preds = pd.DataFrame(preds, columns=['inferred_val'])
        df_preds = df_preds.reset_index().rename({'index': 'tid'}, axis=1)
        df_preds['attribute'] = col
        df_all_preds = pd.concat([df_all_preds, df_preds], axis=0).reset_index(drop=True)

    return df_all_preds


def xgboost_print_mae_mre(num_of_consecutive_rows, target_data_point, target_column, source_file, flattened_file,
                          new_source, create_train_test_data, creat_num_cols, read_dataset, stock_dataset):
    if new_source:
        miss_df = read_dataset(source_file)
        full_df = delete_missing_data(miss_df)
        flattened_data_df = flatten_data_and_store_it(full_df, num_of_consecutive_rows, flattened_file)
    else:
        flattened_data_df = pd.read_csv(flattened_file)
    train_data_df, test_data_df = create_train_test_data(flattened_data_df.copy())

    orig_col_size = int(train_data_df.shape[1] / num_of_consecutive_rows)
    num_cols = creat_num_cols(num_of_consecutive_rows, orig_col_size)
    target_cols = [str(target_data_point * orig_col_size + target_column)]
    all_preds_df = run_xgboost(train_data_df, test_data_df, target_cols, num_cols)
    if stock_dataset:
        observed_cols = [0]
        num_of_cols = flattened_data_df.shape[1]
        for i in range(0, num_of_cols):
            if i % 3 == 2:
                observed_cols.append(i)
        test_data_df = flattened_data_df.copy()
        test_data_df.drop(test_data_df.columns[observed_cols], axis=1, inplace=True)
        new_cols = []
        for i in range(0, test_data_df.shape[1]):
            new_cols.append(str(i))
        test_data_df.columns = new_cols
        test_data_df = test_data_df.iloc[int(test_data_df.shape[0]*8/10):test_data_df.shape[0], :]

    joined_df = pd.concat([test_data_df[target_cols].reset_index(drop=True), all_preds_df['inferred_val']], axis=1)
    joined_df['distance'] = joined_df.apply(lambda row: abs(row[target_cols] - row.inferred_val), axis=1)
    test_data_size = joined_df.shape[0]
    mae = joined_df['distance'].sum() / test_data_size
    mre = joined_df['distance'].sum() / joined_df[target_cols].sum()

    print(joined_df)
    print('MAE: ' + str(mae))
    print('MRE: ' + str(mre[target_cols[0]]))


def create_train_test_stock(flattened_data_df):
    truth_cols = []
    num_of_cols = flattened_data_df.shape[1]
    for i in range(0, num_of_cols):
        if i % 3 == 0:
            truth_cols.append(i)
    flattened_data_df.drop(flattened_data_df.columns[truth_cols], axis=1, inplace=True)
    new_cols = []
    for i in range(0, flattened_data_df.shape[1]):
        new_cols.append(str(i))
    flattened_data_df.columns = new_cols
    train_data_df = flattened_data_df.iloc[0:int(flattened_data_df.shape[0]*8/10), :]
    test_data_df = flattened_data_df.iloc[int(flattened_data_df.shape[0]*8/10):flattened_data_df.shape[0], :]
    return train_data_df, test_data_df


def create_num_cols_stock(num_of_consecutive_rows, orig_col_size):
    num_cols = []
    for i in range(0, num_of_consecutive_rows * orig_col_size):
        num_cols.append(str(i))
    return num_cols


def read_dataset_stock(source_file):
    df = pd.read_csv(source_file)
    return df


def create_train_test_air_quality(flattened_data_df):
    train_data_df = flattened_data_df.loc[flattened_data_df['2'].isin([1, 2, 4, 5, 7, 8, 10, 11])]
    test_data_df = flattened_data_df.loc[flattened_data_df['2'].isin([3, 6, 9, 12])]
    return train_data_df, test_data_df


def create_num_cols_air_quality(num_of_consecutive_rows, orig_col_size):
    num_cols = []
    for i in range(0, num_of_consecutive_rows * orig_col_size):
        if i % 13 != 9:
            num_cols.append(str(i))
    return num_cols


def read_dataset_air_quality(source_file):
    df = pd.read_csv(source_file)
    return df


def create_train_test_human_activity(flattened_data_df):
    train_data_df = flattened_data_df.iloc[0:int(flattened_data_df.shape[0]*8/10), :]
    test_data_df = flattened_data_df.iloc[int(flattened_data_df.shape[0]*8/10):flattened_data_df.shape[0], :]
    return train_data_df, test_data_df


def create_num_cols_human_activity(num_of_consecutive_rows, orig_col_size):
    num_cols = []
    for i in range(0, num_of_consecutive_rows * orig_col_size):
        if i % 8 != 2 or i % 8 != 3 or i % 8 != 0:
            num_cols.append(str(i))
    return num_cols


def read_dataset_human_activity(source_file):
    df = pd.read_csv(source_file)
    return df


if __name__ == '__main__':
    xgboost_print_mae_mre(10, 4, 5, './PRSA_data_2010.1.1-2014.12.31.csv', './flattened_data.csv', 0,
                          create_train_test_air_quality, create_num_cols_air_quality, read_dataset_air_quality, 0)
    # xgboost_print_mae_mre(10, 4, 1, 'stock10k.data', './stock_flattened_data.csv', 0,
    #                       create_train_test_stock, create_num_cols_stock, read_dataset_stock, 1)
    # xgboost_print_mae_mre(5, 4, 6, './ConfLongDemo_JSI.txt', './human_flattened_data.csv', 0,
    #                       create_train_test_human_activity, create_num_cols_human_activity, read_dataset_human_activity,
    #                       0)
