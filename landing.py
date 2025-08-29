import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool
import os
from glob import glob



st.set_page_config(page_title="House Prices: CatBoost", page_icon="🏠", layout="wide")
# --- Responsive SVG CSS ---
st.markdown(
    """
    <style>
      .svg-fi-wrapper { width: 100%; max-width: 100%; overflow: hidden; }
      .svg-fi-wrapper svg { max-width: 100% !important; height: auto !important; display: block; margin: 0 auto; }
      .svg-fi-caption { font-size: 0.85rem; color: #666; text-align:center; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities
# -----------------------------

def rmlse(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def find_fi_images():
    patterns = [
        './*feature*importance*.png','./*feature*importance*.jpg',
        './*importance*.png','./*importance*.jpg',
        './fi*.png','./fi*.jpg',
        'images/*feature*importance*.png','images/*feature*importance*.jpg',
        'images/*importance*.png','images/*importance*.jpg',
        'images/fi*.png','images/fi*.jpg',
        'figs/*feature*importance*.png','figs/*feature*importance*.jpg',
        'assets/*feature*importance*.png','assets/*feature*importance*.jpg',
        'plots/*feature*importance*.png','plots/*feature*importance*.jpg',
    ]
    files = []
    for p in patterns:
        files.extend(glob(p))
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


# --- SVG display helper ---
def show_svg(path: str, caption: str | None = None, max_height: int = 480):
    """Render an SVG scaled to column width with a max height to avoid overlap."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                svg = f.read()
            # Inject responsive styles into the first <svg> tag
            import re
            if "<svg" in svg:
                svg = re.sub(r"<svg(.*?)>",
                             r'<svg\1 style="max-width:100%; height:auto; display:block;">',
                             svg, count=1, flags=re.IGNORECASE|re.DOTALL)
            wrapper = f'<div class="svg-fi-wrapper" style="max-height:{max_height}px;">{svg}</div>'
            cap = f'<div class="svg-fi-caption">{caption or os.path.basename(path)}</div>'
            st.markdown(wrapper + cap, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Не удалось отобразить {path}: {e}")
    else:
        st.info(f"Файл не найден: {path}")

# Columns as in the original script
NUMERIC_FEATURES = [
    'LotArea', 'GrLivArea', 'BsmtUnfSF', '1stFlrSF', 'TotalBsmtSF', 'SalePrice',
    'BsmtFinSF1', 'GarageArea', '2ndFlrSF', 'MasVnrArea', 'WoodDeckSF', 'OpenPorchSF',
    'BsmtFinSF2', 'EnclosedPorch', 'LotFrontage', 'ScreenPorch', 'TotRmsAbvGrd',
    'OverallQual', 'OverallCond', 'BedroomAbvGr', 'PoolArea', 'GarageCars', 'Fireplaces',
    'KitchenAbvGr', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BsmtHalfBath', 'YearBuilt',
    'YearRemodAdd', 'LowQualFinSF', 'MiscVal'
]

CAT_FEATURES = [
    'MSSubClass', 'YrSold', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'LotConfig', 'Neighborhood', 'LandSlope', 'Condition1', 'BldgType', 'HouseStyle',
    'RoofStyle', 'Exterior1st', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'HeatingQC', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'SaleType',
    'SaleCondition'
]

FOR_LOG = [
    'LotArea', 'GrLivArea', 'BsmtUnfSF', '1stFlrSF', 'TotalBsmtSF',
    'BsmtFinSF1', 'GarageArea', '2ndFlrSF', 'MasVnrArea',
    'WoodDeckSF', 'OpenPorchSF', 'BsmtFinSF2', 'EnclosedPorch',
    'LotFrontage', 'ScreenPorch'
]

TYPESTR = ['MSSubClass', 'YrSold']

FOR_DROP = ['Utilities', 'Condition2', 'RoofMatl', 'Exterior2nd', 'BsmtFinType2', 'Heating', 'GarageCond', 'GarageQual', 'Fence']

WORST_FEATURES = [
    'ScreenPorch','HasLotFrontage','HasMasVnr','Has2ndFlr','HasGarageYrBlt',
    'HasTotalBsmtSF','HasBsmtUnfSF','HasBsmtFinSF2','WoodDeckSF','EnclosedPorch',
    'BsmtHalfBath','HasBsmtFullBath','PoolQC','MiscFeature','KitchenAbvGr',
    'Street','MiscVal','PoolArea','3SsnPorch','LowQualFinSF'
]

PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'iterations': 1800,
    'depth': 5,
    'learning_rate': 0.017019709333813988,
    'l2_leaf_reg': 2.7788107429135454,
    'subsample': 0.9506241426228604,
    'colsample_bylevel': 0.9527719324719595,
    'random_state': 42,
    'verbose': 0,
    'thread_count': -1,
}


# -----------------------------
# Preprocessing (train)
# -----------------------------

def prepare_train(df: pd.DataFrame):
    df = df.copy()
    df = df.set_index('Id')

    # fill
    df["Alley"] = df["Alley"].fillna("NoAlley")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["BsmtQual"] = df["BsmtQual"].fillna("NoBsmt")
    df["BsmtCond"] = df["BsmtCond"].fillna("NoBsmt")
    df["BsmtExposure"] = df["BsmtExposure"].fillna("NoBsmt")
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna("NoBsmt")
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")
    df["GarageType"] = df["GarageType"].fillna("NoGarage")
    df["GarageFinish"] = df["GarageFinish"].fillna("NoGarage")
    df["PoolQC"] = df["PoolQC"].fillna("NoPool")
    df["MiscFeature"] = df["MiscFeature"].fillna("None")
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # log features
    for col in FOR_LOG:
        df[f"{col}_log"] = np.log1p(df[col])

    # target log
    df['SalePrice'] = np.log1p(df['SalePrice'])

    # str types
    for col in TYPESTR:
        df[col] = df[col].astype(str)

    # flags
    df["HasTotalBsmtSF"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasBsmtUnfSF"]   = (df["BsmtUnfSF"] > 0).astype(int)
    df["HasBsmtFinSF1"]  = (df["BsmtFinSF1"] > 0).astype(int)
    df["HasBsmtFinSF2"]  = (df["BsmtFinSF2"] > 0).astype(int)
    df["HasBsmtFullBath"] = (df["BsmtFullBath"] > 0).astype(int)

    df["HasGarageArea"] = (df["GarageArea"] > 0).astype(int)
    df["HasGarageYrBlt"] = (~df["GarageYrBlt"].isna()).astype(int)

    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

    df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)

    df["WoodDeckSF"]    = (df["WoodDeckSF"] > 0).astype(int)
    df["EnclosedPorch"] = (df["EnclosedPorch"] > 0).astype(int)
    df["ScreenPorch"]   = (df["ScreenPorch"] > 0).astype(int)
    df["3SsnPorch"]     = (df["3SsnPorch"] > 0).astype(int)

    df["HasLotFrontage"] = (~df["LotFrontage"].isna()).astype(int)

    df["PoolArea"]    = (df["PoolArea"] > 0).astype(int)
    df["MiscVal"]    = (df["MiscVal"] > 0).astype(int)
    df["LowQualFinSF"] = (df["LowQualFinSF"] > 0).astype(int)
    df["PavedDrive"] = (df["PavedDrive"] == "Y").astype(int)
    df["PoolQC"] = (df["PoolQC"] != "NoPool").astype(int)
    df["MiscFeature"] = (df["MiscFeature"] != "None").astype(int)

    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    df["CentralAir"] = (df["CentralAir"] == "Y").astype(int)

    df = df.drop(columns=FOR_DROP)

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values

    # drop worst features (top-20 least important for CatBoost)
    drop_cols = [c for c in WORST_FEATURES if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    X[cat_cols] = X[cat_cols].fillna("__MISSING__").astype(str)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    return df, X, y, cat_cols, cat_idx

# -----------------------------
# Preprocessing (test)
# -----------------------------

def prepare_test(test_df: pd.DataFrame, X_columns: pd.Index, train_df: pd.DataFrame):
    df = test_df.copy()
    df = df.set_index('Id')

    # same fills
    df["Alley"] = df["Alley"].fillna("NoAlley")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["BsmtQual"] = df["BsmtQual"].fillna("NoBsmt")
    df["BsmtCond"] = df["BsmtCond"].fillna("NoBsmt")
    df["BsmtExposure"] = df["BsmtExposure"].fillna("NoBsmt")
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna("NoBsmt")
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")
    df["GarageType"] = df["GarageType"].fillna("NoGarage")
    df["GarageFinish"] = df["GarageFinish"].fillna("NoGarage")
    df["PoolQC"] = df["PoolQC"].fillna("NoPool")
    df["MiscFeature"] = df["MiscFeature"].fillna("None")
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df['LotFrontage'] = df['LotFrontage'].fillna(train_df['LotFrontage'].median())

    for col in FOR_LOG:
        df[f"{col}_log"] = np.log1p(df[col])

    for col in TYPESTR:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["HasTotalBsmtSF"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasBsmtUnfSF"]   = (df["BsmtUnfSF"] > 0).astype(int)
    df["HasBsmtFinSF1"]  = (df["BsmtFinSF1"] > 0).astype(int)
    df["HasBsmtFinSF2"]  = (df["BsmtFinSF2"] > 0).astype(int)
    df["HasBsmtFullBath"] = (df["BsmtFullBath"] > 0).astype(int)

    df["HasGarageArea"] = (df["GarageArea"] > 0).astype(int)
    df["HasGarageYrBlt"] = (~df["GarageYrBlt"].isna()).astype(int)

    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

    df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)

    df["WoodDeckSF"]    = (df["WoodDeckSF"] > 0).astype(int)
    df["EnclosedPorch"] = (df["EnclosedPorch"] > 0).astype(int)
    df["ScreenPorch"]   = (df["ScreenPorch"] > 0).astype(int)
    df["3SsnPorch"]     = (df["3SsnPorch"] > 0).astype(int)

    df["HasLotFrontage"] = (~df["LotFrontage"].isna()).astype(int)

    df["PoolArea"]    = (df["PoolArea"] > 0).astype(int)
    df["MiscVal"]    = (df["MiscVal"] > 0).astype(int)
    df["LowQualFinSF"] = (df["LowQualFinSF"] > 0).astype(int)
    df["PavedDrive"] = (df["PavedDrive"] == "Y").astype(int)
    df["PoolQC"] = (df["PoolQC"] != "NoPool").astype(int)
    df["MiscFeature"] = (df["MiscFeature"] != "None").astype(int)

    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    df["CentralAir"] = (df["CentralAir"] == "Y").astype(int)

    df = df.drop(columns=FOR_DROP)

    # mirror worst-features drop
    drop_cols = [c for c in WORST_FEATURES if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # fill cats like train
    present_cats = [c for c in CAT_FEATURES if c in df.columns]
    if present_cats:
        df[present_cats] = df[present_cats].fillna("__MISSING__").astype(str)
    for c in TYPESTR:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # align columns
    missing_in_test = [c for c in X_columns if c not in df.columns]
    for c in missing_in_test:
        df[c] = 0
    extra_in_test = [c for c in df.columns if c not in X_columns]
    if extra_in_test:
        df = df.drop(columns=extra_in_test)

    df = df[X_columns]

    # final NaN check
    if df.isna().any().any():
        # fill by train statistics for any leftovers
        fillmap = {
            c: (train_df[c].median() if pd.api.types.is_numeric_dtype(train_df[c])
                else (train_df[c].mode(dropna=True).iloc[0] if not train_df[c].mode(dropna=True).empty else "__MISSING__"))
            for c in df.columns if c in train_df.columns
        }
        df = df.fillna(fillmap)

    return df

# -----------------------------
# Training (cached)
# -----------------------------

@st.cache_data(show_spinner=False)
def load_train():
    return pd.read_csv('train.csv')

@st.cache_resource(show_spinner=False)
def train_model(train_df: pd.DataFrame):
    # prepare
    df_prep, X, y, cat_cols, cat_idx = prepare_train(train_df)
    model = CatBoostRegressor(**PARAMS)
    pool = Pool(X, y, cat_features=cat_idx)
    model.fit(pool, verbose=False)

    # train score
    y_pred_log = model.predict(pool)
    score = rmlse(np.expm1(y), np.expm1(y_pred_log))

    return model, X.columns, cat_idx, df_prep, score

# -----------------------------
# UI
# -----------------------------

st.title("🏠 House Prices — CatBoost")

# --- EDA summary at top ---
train_df = load_train()
n_rows, n_cols = train_df.shape
num_cols_count = train_df.select_dtypes(include='number').shape[1]
obj_cols_count = n_cols - num_cols_count

eda_md = f"""
**Датасет:** {n_rows} строк, {n_cols} колонок  
**Типы:** числовых — {num_cols_count}, объектных — {obj_cols_count}

**Шаги препроцессинга:**
1. Целевая: `SalePrice` → log1p.
2. Заполнение пропусков:
   - `Alley`→`NoAlley`, `MasVnrType`→`None`,
   - `BsmtQual/BsmtCond/BsmtExposure/BsmtFinType1`→`NoBsmt`,
   - `Electrical`→`SBrkr`,
   - `FireplaceQu`→`NoFireplace`,
   - `GarageType/GarageFinish`→`NoGarage`,
   - `PoolQC`→`NoPool`, `MiscFeature`→`None`,
   - `GarageYrBlt`→0, `MasVnrArea`→0,
   - `LotFrontage`→ медиана train.
3. Лог-признаки: {', '.join(FOR_LOG)} → `*_log`.
4. Типы: `MSSubClass`, `YrSold` → строки (`str`).
5. Булевы флаги: `HasTotalBsmtSF`, `HasBsmtUnfSF`, `HasBsmtFinSF1`, `HasBsmtFinSF2`, `HasBsmtFullBath`,
   `HasGarageArea`, `HasGarageYrBlt`, `Has2ndFlr`, `HasMasVnr`, `WoodDeckSF`, `EnclosedPorch`, `ScreenPorch`,
   `3SsnPorch`, `HasLotFrontage`, `PoolArea`, `MiscVal`, `LowQualFinSF`, `PavedDrive`, `PoolQC`, `MiscFeature`,
   `HasFireplace`, `CentralAir`.
6. Удалённые признаки: {', '.join(FOR_DROP)}.
7. Выравнивание test под train: добавляем недостающие столбцы (=0), лишние удаляем, остаточные NaN → (число: медиана, категория: мода).
8. Удалили топ-20 наименее важных признаков (по FI CatBoost) — для устойчивости и компактности.
"""

with st.expander("Краткий EDA (что сделано для сабмита)", expanded=True):
    st.markdown(eda_md)


# --- Feature Importance SVG block ---
with st.expander("Важность признаков (Feature Importance)", expanded=False):
    show_svg("featimp_top20.svg", caption="Top-20 важных признаков (CatBoost)", max_height=480)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    show_svg("featimp_bottom20.svg", caption="Bottom-20 важных признаков (CatBoost)", max_height=480)

    st.markdown("В рамках анализа важности признаков удалили **топ-20 наименее важных** для CatBoost, затем переобучили модель на урезанном наборе признаков.")
    st.code(
        """
worst_features = [
    'ScreenPorch','HasLotFrontage','HasMasVnr','Has2ndFlr','HasGarageYrBlt',
    'HasTotalBsmtSF','HasBsmtUnfSF','HasBsmtFinSF2','WoodDeckSF','EnclosedPorch',
    'BsmtHalfBath','HasBsmtFullBath','PoolQC','MiscFeature','KitchenAbvGr',
    'Street','MiscVal','PoolArea','3SsnPorch','LowQualFinSF'
]
# train: X = X.drop(columns=[c for c in worst_features if c in X.columns])
# test:  df = df.drop(columns=[c for c in worst_features if c in df.columns])
        """,
        language="python",
    )

st.caption("Загрузите test.csv или используйте файл из папки проекта. На выходе — submission.csv")

with st.spinner("Загрузка и обучение модели..."):
    train_df = load_train()
    model, X_columns, cat_idx, df_prep, train_rmlse = train_model(train_df)

col1, col2 = st.columns(2)
col1.metric("RMLSE на train (full pool)", f"{train_rmlse:.6f}")
col2.write("")

uploaded = st.file_uploader("Загрузите test.csv", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    test_raw = pd.read_csv(uploaded)
    st.success("Используется загруженный файл")
else:
    test_raw = pd.read_csv('test.csv')
    st.info("Файл не загружен. Используется test.csv из папки проекта")

if st.button("Сгенерировать submission.csv"):
    try:
        test_X = prepare_test(test_raw, X_columns, df_prep)
        # predict
        y_test_log = model.predict(test_X)
        y_test = np.expm1(y_test_log)

        # build submission
        if 'Id' not in test_raw.columns:
            st.error("В test.csv отсутствует колонка 'Id'.")
        else:
            submission = pd.DataFrame({"Id": test_raw["Id"].values, "SalePrice": y_test})
            csv_bytes = submission.to_csv(index=False).encode('utf-8')

            # Save to disk and provide download
            with open('submission.csv', 'wb') as f:
                f.write(csv_bytes)

            st.download_button(
                label="Скачать submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
            )

            st.dataframe(submission.head(10))
            st.success("Готово: submission.csv")
    except Exception as e:
        st.error(f"Ошибка: {e}")
