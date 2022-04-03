import numpy as np
import pandas as pd 
import pickle


# load onehot encoder
with open("preprocessors/onehot_encoder_v4.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# load scaler
with open("preprocessors/standard_scaler_v4.pkl", "rb") as f:
    scaler = pickle.load(f)


# TODO
COLUMNS_TO_REMOVE = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber", 'YearsSinceLastPromotion', 'JobSatisfaction', 'HourlyRate', 'MonthlyRate'

]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = ["BusinessTravel", "Department", "EducationField",
    "JobRole", "MaritalStatus", "Gender", "OverTime"

]


def preprocess(sample: dict) -> np.ndarray:
    sample_df = pd.DataFrame(sample, index=[0])

    sample_df = create_features(sample_df)
    sample_df = encode_columns(sample_df)
    sample_df = drop_columns(sample_df)

    scaled_sample_values = scale(sample_df.values)
    scaled_sample_values = scaled_sample_values.reshape(1, -1)
    return scaled_sample_values







def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # create MeanAttritionYear feature
    df["MeanAttritionYear"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)

    # create YearsAtCompanyCat
    bins = pd.IntervalIndex.from_tuples([(-1, 5), (5, 10), (10, 15), (15, 100)])
    cat_YearsAtCompany = pd.cut(df["YearsAtCompany"].to_list(), bins)
    cat_YearsAtCompany.categories = [0, 1, 2, 3]
    df["YearsAtCompanyCat"] = cat_YearsAtCompany
    #
    # create  cat_StockOption
    bins = pd.IntervalIndex.from_tuples([(-1, 0), (0, 3)])
    cat_StockOption = pd.cut(df["StockOptionLevel"].to_list(), bins)
    cat_StockOption.categories = [0, 1]
    df["cat_StockOption"] = cat_StockOption

    # create cat_YearsInCurrentRole
    bins = pd.IntervalIndex.from_tuples([(-1, 4), (4, 30)])
    cat_YearsInCurrentRole = pd.cut(df["YearsInCurrentRole"].to_list(), bins)
    cat_YearsInCurrentRole.categories = [0, 1]
    df["cat_YearsInCurrentRole"] = cat_YearsInCurrentRole
    #
    # create cat_YearsSinceLastPromotion
    bins = pd.IntervalIndex.from_tuples([(-1, 2), (2, 20)])
    cat_YearsSinceLastPromotion = pd.cut(df["YearsSinceLastPromotion"].to_list(), bins)
    cat_YearsSinceLastPromotion.categories = [0, 1]
    df["cat_YearsSinceLastPromotion"] = cat_YearsSinceLastPromotion

    # create cat_YearsWithCurrManager
    bins = pd.IntervalIndex.from_tuples([(-1, 5), (5, 30)])
    cat_YearsWithCurrManager = pd.cut(df["YearsWithCurrManager"].to_list(), bins)
    cat_YearsWithCurrManager.categories = [0, 1]
    df["cat_YearsWithCurrManager"] = cat_YearsWithCurrManager

    # create IncomePercentHike
    #df["IncomePercentHike"] = df["PercentSalaryHike"] * df["MonthlyIncome"]
    #
    # create PerformanceSalaryHike
    df["PerformanceSalaryHike"] = df["PercentSalaryHike"] * df["PerformanceRating"]

    # create AgeJobLevel
    df["AgeJobLevel"] = df["Age"] / df["JobLevel"]
    #
    # create WorkYearJobLevel
    df["WorkYearJobLevel"] = df["TotalWorkingYears"] / df["JobLevel"]

    # create cat_FirstCompanyOrNot
    bins = pd.IntervalIndex.from_tuples([(-1, 0), (0, 10)])
    cat_FirstCompanyOrNot = pd.cut(df["NumCompaniesWorked"].to_list(), bins)
    cat_FirstCompanyOrNot.categories = [0, 1]
    df["cat_FirstCompanyOrNot"] = cat_FirstCompanyOrNot
    #
    # create cat_WorkTenYears
    bins = pd.IntervalIndex.from_tuples([(-1, 10), (10, 50)])
    cat_WorkTenYears = pd.cut(df["TotalWorkingYears"].to_list(), bins)
    cat_WorkTenYears.categories = [0, 1]
    df["cat_WorkTenYears"] = cat_WorkTenYears

    # create NumCompEducaiton
    df["NumCompEducation"] = df["NumCompaniesWorked"] * df["Education"]
    #
    # create JobEnvSatisfaction
    df["JobEnvSatisfaction"] = df["EnvironmentSatisfaction"] * df["JobSatisfaction"]

    # create JobInvSatisfaction
    #df["JobInvSatisfaction"] = df["JobInvolvement"] * df["JobSatisfaction"]
    #
    # create IncomeJoblevel###############################
    df["IncomeJoblevel"] = df["MonthlyIncome"] * df["JobLevel"]

    # create MonthlyIncomeCat####################################

    # bins = pd.IntervalIndex.from_tuples([(-1, 7500), (7500, 12500), (12500, 30000)])
    # cat_MonthlyIncome = pd.cut(df["MonthlyIncome"].to_list(), bins)
    # cat_MonthlyIncome.categories = [0, 1, 2]
    # df["cat_MonthlyIncome"] = cat_MonthlyIncome


    #create CompanyCurrentRole
    df["CompanyCurrentRole"] = df["YearsAtCompany"] / (df["YearsInCurrentRole"] + 1)

    # create Age_cat
    bins = pd.IntervalIndex.from_tuples([(-1, 25), (25, 35), (35, 65)])
    cat_Age = pd.cut(df["Age"].to_list(), bins)
    cat_Age.categories = [0, 1, 2]
    df["cat_Age"] = cat_Age
    # TODO

    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_REMOVE)

def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe with one-hot encoded columns
    encoded_df = pd.DataFrame(onehot_encoder.transform(df[COLUMNS_TO_ONEHOT_ENCODE]).toarray())
    # set new column names
    column_names = onehot_encoder.get_feature_names(COLUMNS_TO_ONEHOT_ENCODE)
    encoded_df.columns = column_names
    # drop raw columns, and add one-hot encoded columns instead
    df = df.drop(columns=COLUMNS_TO_ONEHOT_ENCODE, axis=1)
    df = df.join(encoded_df)

    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
