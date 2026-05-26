def transform_features(df):

    # Encode key
    list_of_keys = df['key'].unique()

    for i in range(len(list_of_keys)):
        df.loc[df['key'] == list_of_keys[i], 'key'] = i

    # Encode mode
    df.loc[df["mode"] == 'Major', "mode"] = 1
    df.loc[df["mode"] == 'Minor', "mode"] = 0

    # Encode time_signature
    list_of_time_signatures = df['time_signature'].unique()

    for i in range(len(list_of_time_signatures)):
        df.loc[df['time_signature'] == list_of_time_signatures[i], 'time_signature'] = i

    # Binary target
    df.loc[df['popularity'] < 57, 'popularity'] = 0
    df.loc[df['popularity'] >= 57, 'popularity'] = 1

    return df
